#!/usr/bin/env python3
"""Create a timestamped transcript from a local or remote media file."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Iterator


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe a video or audio file with faster-whisper."
    )
    parser.add_argument("input", help="Local media path or HTTP(S) URL")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path; defaults to <input-stem>.<format>",
    )
    parser.add_argument(
        "--format",
        choices=("text", "srt", "json"),
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--model",
        default="small",
        help="Whisper model name (default: small)",
    )
    parser.add_argument(
        "--language",
        help="Language code, such as en; omit to detect it automatically",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device (default: auto)",
    )
    parser.add_argument(
        "--compute-type",
        default="auto",
        help="faster-whisper compute type (default: auto)",
    )
    return parser.parse_args(argv)


def is_url(value: str) -> bool:
    parsed = urllib.parse.urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def media_path(source: str) -> Iterator[Path]:
    """Yield a local source path and clean up a temporary URL download."""

    if not is_url(source):
        path = Path(source).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Media file does not exist: {path}")
        yield path
        return

    suffix = Path(urllib.parse.urlparse(source).path).suffix or ".media"
    with tempfile.TemporaryDirectory(prefix="video-transcriber-") as directory:
        path = Path(directory) / f"input{suffix}"
        print(f"Downloading {source}", file=sys.stderr)
        urllib.request.urlretrieve(source, path)
        yield path


def seconds_to_timestamp(seconds: float, *, decimal: str = ".") -> str:
    """Format seconds as HH:MM:SS.mmm for text, JSON, and SRT output."""

    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}{decimal}{milliseconds:03d}"


def segment_dict(segment: Any) -> dict[str, Any]:
    return {
        "start": round(float(segment.start), 3),
        "end": round(float(segment.end), 3),
        "text": segment.text.strip(),
    }


def collect_segments(segments: Iterable[Any]) -> list[dict[str, Any]]:
    return [item for segment in segments if (item := segment_dict(segment))["text"]]


def text_output(segments: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"[{seconds_to_timestamp(segment['start'])}] {segment['text']}"
        for segment in segments
    ) + "\n"


def srt_output(segments: list[dict[str, Any]]) -> str:
    entries = []
    for index, segment in enumerate(segments, start=1):
        start = seconds_to_timestamp(segment["start"], decimal=",")
        end = seconds_to_timestamp(segment["end"], decimal=",")
        entries.append(f"{index}\n{start} --> {end}\n{segment['text']}\n")
    return "\n".join(entries)


def json_output(
    segments: list[dict[str, Any]], *, model: str, language: str | None
) -> str:
    return json.dumps(
        {"model": model, "language": language, "segments": segments},
        ensure_ascii=False,
        indent=2,
    ) + "\n"


def default_output(source: str, output_format: str) -> Path:
    name = Path(urllib.parse.urlparse(source).path).stem or "transcript"
    suffix = {"text": ".txt", "srt": ".srt", "json": ".json"}[output_format]
    return Path(f"{name}{suffix}")


def transcribe(args: argparse.Namespace) -> tuple[list[dict[str, Any]], Any]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as error:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "faster-whisper is not installed; run `pixi install` first"
        ) from error

    for path in media_path(args.input):
        device = "cpu" if args.device == "auto" else args.device
        compute_type = args.compute_type
        if compute_type == "auto":
            compute_type = "int8" if device == "cpu" else "float16"

        model = WhisperModel(args.model, device=device, compute_type=compute_type)
        segments, info = model.transcribe(
            str(path),
            language=args.language,
            beam_size=5,
            vad_filter=True,
        )
        return collect_segments(segments), info

    raise RuntimeError("Could not resolve the media input")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output or default_output(args.input, args.format)
    try:
        segments, info = transcribe(args)
        output.parent.mkdir(parents=True, exist_ok=True)
        if args.format == "text":
            rendered = text_output(segments)
        elif args.format == "srt":
            rendered = srt_output(segments)
        else:
            rendered = json_output(
                segments,
                model=args.model,
                language=getattr(info, "language", args.language),
            )
        output.write_text(rendered, encoding="utf-8")
    except (FileNotFoundError, OSError, RuntimeError, urllib.error.URLError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    detected = getattr(info, "language", None)
    print(
        f"Wrote {len(segments)} segments to {output}"
        + (f" (language: {detected})" if detected else ""),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
