# PodCut Sync

PodCut Sync is an early command line tool for editing podcasts or interviews recorded on several cameras. It matches camera scratch audio against a master recording and exports an editable Final Cut Pro 7 XML timeline. The analysis runs locally.

## What it does

- Places matching camera angles on separate video tracks and uses the master audio on the timeline.
- Estimates recording drift from multiple waveform matches when there is enough evidence. The report gives the estimated drift in parts per million, and the timeline flags angles that may drift by a frame or more. Placement uses a fixed offset, so drift correction is manual in this version.
- Optionally builds a rough cut from camera cues you supply, keeps the synchronized angles on disabled tracks, and adds review markers for overlapping cues, missing camera coverage or uncertain alignment.
- Writes a JSON report with offsets, match scores, skipped files, and review markers.

## Setup

Install Python 3.10 or newer and [FFmpeg](https://ffmpeg.org/download.html). Make sure both `ffmpeg` and `ffprobe` are available on your command line. From the project folder, run:

```sh
python -m pip install -r requirements.txt
```

## Sync two cameras

```sh
python podcut_sync.py wide.mp4 guest.mp4 --audio master.wav --output interview.xml
```

The camera files need scratch audio that overlaps with the master. If you omit `--audio`, the first camera is the reference. Use `--max-offset 120` if the start times may differ by up to two minutes. The XML references the original media by absolute path. Keep those files available or relink them when importing the timeline. Use a new output filename each time.

Import `interview.xml` into an editor supporting Final Cut Pro 7 XML, also called `xmeml`. Import behavior, including markers and disabled tracks, varies by editor. Check the cuts and sync in your editor before using the timeline in production.

## Generate a rough cut from speaker cues

Create `speakers.csv` with times in seconds relative to the master audio and the exact filename of the preferred camera on each row.

```csv
start,end,camera
0,12,wide.mp4
12,25,guest.mp4
23,30,wide.mp4
```

```sh
python podcut_sync.py wide.mp4 guest.mp4 --audio master.wav --cues speakers.csv --output rough-cut.xml
```

In this example, the cut switches to the guest at 12 seconds, keeps the current angle during the overlap from 23 to 25 seconds, and marks that overlap for review. The tool does not identify speakers or create a transcript in this release.

## Current limits

Use one continuous, preferably constant frame rate clip per camera. Recordings with little shared audio, strong reverberation, or periodic sound can be difficult to align. Angles with an ambiguous match are skipped. This is a prototype and has been exercised with generated recordings. Feedback from real editing sessions would help guide the next version.

## License

MIT. See [LICENSE](LICENSE).
