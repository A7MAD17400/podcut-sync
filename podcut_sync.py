"""Align podcast camera recordings and export an editable timeline."""

import argparse
import csv
import itertools
import json
import math
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np


SAMPLE_RATE = 4000
MIN_MATCH = 0.18


@dataclass(frozen=True)
class Recording:
    path: Path
    audio_start: float
    audio_length: float
    audio_rate: int
    channels: int
    video_start: float = 0.0
    video_length: float = 0.0
    fps: Fraction = Fraction(30)
    width: int = 0
    height: int = 0


@dataclass(frozen=True)
class Match:
    offset: float
    score: float
    anchors: int
    drift_ppm: float | None


@dataclass(frozen=True)
class Camera:
    recording: Recording
    match: Match

    @property
    def offset(self):
        return self.match.offset + self.recording.video_start - self.recording.audio_start


def run_media(command):
    process = subprocess.run(command, capture_output=True, check=False)
    if process.returncode:
        error = process.stderr.decode(errors="replace").strip()
        raise ValueError(error or f"{command[0]} failed")
    return process.stdout


def probe(path, need_video):
    path = path.resolve(strict=True)
    data = json.loads(run_media([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path),
    ]))
    streams = data.get("streams", [])
    audio = next((stream for stream in streams if stream["codec_type"] == "audio"), None)
    if audio is None:
        raise ValueError(f"{path.name} needs an audio track for waveform matching")
    video = next((stream for stream in streams if stream["codec_type"] == "video"), None)
    if need_video and video is None:
        raise ValueError(f"{path.name} has no video track")

    def duration(stream):
        value = stream.get("duration") or data.get("format", {}).get("duration")
        if value in (None, "N/A"):
            raise ValueError(f"Cannot determine duration for {path.name}")
        seconds = float(value)
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError(f"Invalid duration for {path.name}")
        return seconds

    video = video if need_video else None
    return Recording(
        path=path, audio_start=float(audio.get("start_time", 0)),
        audio_length=duration(audio), audio_rate=int(audio["sample_rate"]),
        channels=int(audio["channels"]),
        video_start=float(video.get("start_time", 0)) if video else 0.0,
        video_length=duration(video) if video else 0.0,
        fps=Fraction(video["avg_frame_rate"]) if video else Fraction(30),
        width=int(video["width"]) if video else 0,
        height=int(video["height"]) if video else 0,
    )


def waveform(path):
    raw = run_media([
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", "0:a:0",
        "-af", "highpass=f=90,lowpass=f=1600", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-f", "s16le", "pipe:1",
    ])
    return np.frombuffer(raw, dtype="<i2")


def best_peak(search, template):
    """Find a waveform match and its separation from competing peaks."""
    template = template.astype(np.float64)
    template -= template.mean()
    energy = np.dot(template, template)
    if energy < len(template) * 25**2 or len(search) < len(template):
        return None

    signal = search.astype(np.float64)
    size = len(template)
    fft_size = 1 << (len(signal) + size - 2).bit_length()
    product = np.fft.rfft(signal, fft_size) * np.fft.rfft(template[::-1], fft_size)
    numerator = np.fft.irfft(product, fft_size)[size - 1:len(signal)]
    sums = np.concatenate(([0.0], np.cumsum(signal)))
    squares = np.concatenate(([0.0], np.cumsum(signal * signal)))
    variance = squares[size:] - squares[:-size] - (sums[size:] - sums[:-size]) ** 2 / size
    valid = variance >= size * 25**2
    scores = np.zeros(len(numerator))
    np.divide(np.abs(numerator), np.sqrt(energy * np.maximum(variance, 0)), out=scores, where=valid)
    peak = int(np.argmax(scores))
    score = float(min(scores[peak], 1.0))
    guard = SAMPLE_RATE // 20
    background = max(np.max(scores[:max(0, peak - guard)], initial=0),
                     np.max(scores[peak + guard:], initial=0))
    if score < MIN_MATCH or score < 1.15 * background:
        return None
    return peak, score


def align(reference, candidate, max_offset):
    size = min(SAMPLE_RATE * 3, len(reference) // 2, len(candidate) // 2)
    if size < SAMPLE_RATE:
        return None
    radius = round(max_offset * SAMPLE_RATE)
    anchors = []
    for start in np.linspace(0, len(candidate) - size, 7, dtype=int):
        lower = max(0, start - radius)
        upper = min(len(reference), start + radius + size)
        match = best_peak(reference[lower:upper], candidate[start:start + size])
        if match:
            lag, score = match
            anchors.append(((start + size / 2) / SAMPLE_RATE,
                            (lower + lag - start) / SAMPLE_RATE, score))
    if not anchors:
        return None

    # Find a consistent line through the anchors before estimating clock drift.
    points = np.asarray(anchors)
    seconds, lags, scores = points.T
    models = [(0.0, lag) for lag in lags]
    for a, b in itertools.combinations(points, 2):
        if b[0] - a[0] >= 4:
            slope = (b[1] - a[1]) / (b[0] - a[0])
            if abs(slope) < 0.005:
                models.append((slope, a[1] - slope * a[0]))
    masks = [np.abs(lags - (slope * seconds + offset)) <= 0.012 for slope, offset in models]
    keep = max(masks, key=lambda mask: (mask.sum(), scores[mask].sum()))
    if keep.sum() == 1 and len(anchors) > 1:
        return None

    drift = None
    offset = float(np.median(lags[keep]))
    if keep.sum() >= 3 and np.ptp(seconds[keep]) >= 8:
        slope, offset = np.polyfit(seconds[keep], lags[keep], 1)
        drift = float(slope * 1e6)
    return Match(float(offset), float(np.median(scores[keep])), int(keep.sum()), drift)


def read_cues(path, names, duration, fps):
    cues = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"start", "end", "camera"}.issubset(reader.fieldnames or []):
            raise ValueError("Cue CSV needs start, end, and camera columns")
        for row in reader:
            try:
                start = float(row["start"] or "")
                end = float(row["end"] or "")
            except ValueError as error:
                raise ValueError(f"Invalid time on CSV line {reader.line_num}") from error
            camera = (row["camera"] or "").strip()
            if (not 0 <= start < end <= duration or camera not in names
                    or round(start * fps) == round(end * fps)):
                raise ValueError(f"Invalid cue on CSV line {reader.line_num}")
            cues.append((round(start * fps), round(end * fps), camera))
    return cues


def rough_cut(coverage, cues, length):
    boundaries = {0, length}
    for start, end in coverage.values():
        boundaries.update((start, end))
    for start, end, _ in cues:
        boundaries.update((start, end))
    cuts = []
    markers = []
    current = None
    timeline = sorted(boundaries)
    for start, end in zip(timeline, timeline[1:]):
        available = [name for name, (lo, hi) in coverage.items() if lo <= start < hi]
        requested = {name for lo, hi, name in cues if lo <= start < hi}
        if len(requested) > 1:
            markers.append((start, end, "Overlapping speaker cues"))
        if len(requested) == 1 and not requested.issubset(available):
            markers.append((start, end, "Requested camera unavailable"))
        if not available:
            markers.append((start, end, "No camera coverage"))
            current = None
            continue
        if len(requested) == 1 and requested.issubset(available):
            current = next(iter(requested))
        elif current not in available:
            current = available[0]
        if cuts and cuts[-1][1:] == (start, current):
            cuts[-1] = (cuts[-1][0], end, current)
        else:
            cuts.append((start, end, current))
    return cuts, markers


def add(parent, tag, value):
    ET.SubElement(parent, tag).text = str(value)


def rate(parent, fps):
    item = ET.SubElement(parent, "rate")
    base = round(fps)
    if fps not in (base, Fraction(base * 1000, 1001)) or base < 1:
        raise ValueError(f"Unsupported frame rate for XML export: {fps}")
    add(item, "timebase", base)
    add(item, "ntsc", "FALSE" if fps == base else "TRUE")


def add_clip(track, recording, fps, start, end, offset, item_id, channel=None):
    audio = channel is not None
    source_fps = fps if audio else recording.fps
    duration = recording.audio_length if audio else recording.video_length
    source_frames = round(duration * source_fps)
    source_in = max(0, round((start / fps - offset) * source_fps))
    source_out = min(source_frames, round((end / fps - offset) * source_fps))
    if source_out <= source_in:
        return
    item = ET.SubElement(track, "clipitem", id=f"clip-{item_id}")
    add(item, "name", recording.path.name)
    add(item, "enabled", "TRUE")
    add(item, "duration", source_frames)
    rate(item, source_fps)
    for tag, value in (("start", start), ("end", end), ("in", source_in), ("out", source_out)):
        add(item, tag, value)
    source = ET.SubElement(item, "file", id=f"file-{item_id}")
    add(source, "name", recording.path.name)
    add(source, "pathurl", recording.path.as_uri())
    add(source, "duration", source_frames)
    rate(source, source_fps)
    media = ET.SubElement(source, "media")
    stream = ET.SubElement(media, "audio" if audio else "video")
    characteristics = ET.SubElement(stream, "samplecharacteristics")
    if audio:
        add(characteristics, "samplerate", recording.audio_rate)
        add(stream, "channelcount", recording.channels)
        track_ref = ET.SubElement(item, "sourcetrack")
        add(track_ref, "mediatype", "audio")
        add(track_ref, "trackindex", channel)
    else:
        rate(characteristics, source_fps)
        add(characteristics, "width", recording.width)
        add(characteristics, "height", recording.height)


def export_timeline(name, reference, cameras, fps, cues):
    length = round(reference.audio_length * fps)
    coverage = {}
    for camera in cameras:
        start = max(0, round(camera.offset * fps))
        end = min(length, round((camera.offset + camera.recording.video_length) * fps))
        if start < end:
            coverage[camera.recording.path.name] = (start, end)
    if not coverage:
        raise ValueError("No aligned video overlaps the reference audio")

    cuts, markers = rough_cut(coverage, cues, length) if cues else ([], [])
    root = ET.Element("xmeml", version="5")
    sequence = ET.SubElement(root, "sequence", id="podcut-sequence")
    add(sequence, "name", name)
    add(sequence, "duration", length)
    rate(sequence, fps)
    media = ET.SubElement(sequence, "media")
    video = ET.SubElement(media, "video")
    display = ET.SubElement(ET.SubElement(video, "format"), "samplecharacteristics")
    rate(display, fps)
    add(display, "width", cameras[0].recording.width)
    add(display, "height", cameras[0].recording.height)
    camera_by_name = {camera.recording.path.name: camera for camera in cameras}
    if cues:
        cut_track = ET.SubElement(video, "track")
        for index, (start, end, camera_name) in enumerate(cuts, 1):
            camera = camera_by_name[camera_name]
            add_clip(cut_track, camera.recording, fps, start, end, camera.offset, f"cut-{index}")
    for index, (camera_name, (start, end)) in enumerate(coverage.items(), 1):
        camera = camera_by_name[camera_name]
        track = ET.SubElement(video, "track")
        add_clip(track, camera.recording, fps, start, end, camera.offset, f"angle-{index}")
        if cues:
            add(track, "enabled", "FALSE")
        if camera.match.anchors == 1 or camera.match.score < 0.30:
            markers.append((start, min(end, start + 1), f"Check sync for {camera_name}"))
        drift = camera.match.drift_ppm
        if drift is not None and abs(drift) * camera.recording.video_length / 1e6 >= 1 / fps:
            markers.append((start, min(end, start + 1), f"Check clock drift for {camera_name}"))
    audio = ET.SubElement(media, "audio")
    for channel in range(1, reference.channels + 1):
        track = ET.SubElement(audio, "track")
        add_clip(track, reference, fps, 0, length, 0, f"audio-{channel}", channel=channel)
    markers.sort()
    for start, end, label in markers:
        marker = ET.SubElement(sequence, "marker")
        add(marker, "name", label)
        add(marker, "in", start)
        add(marker, "out", end)
    ET.indent(root)
    header = b'<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE xmeml>\n'
    return header + ET.tostring(root, encoding="utf-8") + b"\n", markers, coverage


def write_outputs(xml_path, xml, report_path, report):
    created = []
    try:
        for path, payload in ((xml_path, xml), (report_path, report)):
            with path.open("xb") as stream:
                created.append(path)
                stream.write(payload)
    except OSError:
        for path in created:
            path.unlink(missing_ok=True)
        raise


def run(args):
    if any(shutil.which(tool) is None for tool in ("ffmpeg", "ffprobe")):
        raise ValueError("Install FFmpeg and add ffmpeg and ffprobe to PATH")
    if not math.isfinite(args.max_offset) or args.max_offset <= 0:
        raise ValueError("--max-offset must be a positive number of seconds")
    if args.output.suffix.lower() != ".xml":
        raise ValueError("The output filename must end in .xml")
    report_path = args.output.with_suffix(".json")
    if args.output.exists() or report_path.exists():
        raise ValueError("Choose a new output name. The XML or JSON file already exists")
    recordings = [probe(path, need_video=True) for path in args.cameras]
    names = [item.path.name for item in recordings]
    if len(set(names)) != len(names):
        raise ValueError("Use distinct camera filenames")
    reference = probe(args.audio, need_video=False) if args.audio else recordings[0]
    if reference.channels not in (1, 2):
        raise ValueError("Use a mono or stereo master audio recording")
    fps = args.fps or recordings[0].fps
    if fps <= 0:
        raise ValueError("The timeline frame rate must be positive")
    if round(reference.audio_length * fps) < 1:
        raise ValueError("The reference audio is shorter than one timeline frame")
    cues = read_cues(args.cues, names, reference.audio_length, fps) if args.cues else []
    ref_wave = waveform(reference.path)
    cameras = []
    skipped = []
    for item in recordings:
        if item.path == reference.path:
            result = Match(0.0, 1.0, 0, 0.0)
        else:
            result = align(ref_wave, waveform(item.path), args.max_offset)
        if result is None:
            skipped.append(item.path.name)
            print(f"Skipped {item.path.name}: no reliable waveform match", file=sys.stderr)
        else:
            cameras.append(Camera(item, result))
    if not cameras:
        raise ValueError("No camera could be aligned to the reference")
    xml, markers, coverage = export_timeline(args.output.stem, reference, cameras, fps, cues)
    report = {
        "reference": reference.path.name,
        "timeline_fps": str(fps),
        "cameras": [{**asdict(camera.match), "file": camera.recording.path.name,
                     "video_offset_seconds": camera.offset,
                     "included": camera.recording.path.name in coverage} for camera in cameras],
        "unmatched": skipped,
        "review_markers": [{"start_seconds": float(start / fps), "end_seconds": float(end / fps),
                            "note": note} for start, end, note in markers],
    }
    write_outputs(args.output, xml, report_path, (json.dumps(report, indent=2) + "\n").encode())
    print(f"Exported {len(coverage)} of {len(recordings)} camera angles to {args.output}")
    print(f"Sync report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cameras", nargs="+", type=Path, help="Camera clips with scratch audio")
    parser.add_argument("--audio", type=Path, help="Master recording. Defaults to the first camera")
    parser.add_argument("--cues", type=Path, help="Optional CSV of start,end,camera speaker cues")
    parser.add_argument("--max-offset", type=float, default=60, help="Search radius in seconds (default: 60)")
    parser.add_argument("--fps", type=Fraction, help="Timeline frame rate, e.g. 30000/1001")
    parser.add_argument("--output", "-o", type=Path, required=True, help="New timeline XML filename")
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError, ArithmeticError) as error:
        parser.exit(1, f"PodCut Sync: {error}\n")


if __name__ == "__main__":
    main()
