# Video Transcriber

A small, local command-line tool for producing a timestamped transcript from a
video or audio file. It uses the open-source
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) implementation and
downloads the selected Whisper model into the normal local model cache on its
first run.

## Requirements

- [Pixi](https://pixi.sh/)
- Internet access on the first run, to install the environment and download a
  model

The media file is not uploaded to a service. Transcription runs locally.

## Usage

From this repository:

```console
pixi install
pixi run transcribe -- /path/to/video.mp4 --output transcripts/video.txt
```

The input can also be an HTTP(S) URL:

```console
pixi run transcribe -- \
  'https://example.org/meeting.mp4' \
  --output transcripts/meeting.txt
```

The default model is `small`, which is a reasonable starting point for short
meetings. Use `--model medium` or `--model large-v3` when accuracy matters more
than runtime. The first use of each model downloads it once and then reuses the
local cache.

Useful options:

```console
pixi run transcribe -- video.mp4 --model medium --language en
pixi run transcribe -- video.mp4 --format srt --output transcripts/video.srt
pixi run transcribe -- video.mp4 --format json --output transcripts/video.json
```

Text output contains one timestamped paragraph per detected segment. SRT and
JSON output are intended for downstream tools.

## Development

Run the small standard-library test suite with:

```console
pixi run test
```
