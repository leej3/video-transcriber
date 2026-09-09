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


def positive_int(value: str) -> int:
    count = int(value)
    if count < 2:
        raise argparse.ArgumentTypeError("speaker count must be at least 2")
    return count


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
    parser.add_argument(
        "--no-diarize",
        action="store_true",
        help="omit local speaker clustering and use the original transcript format",
    )
    parser.add_argument(
        "--speakers",
        type=positive_int,
        default=2,
        metavar="N",
        help="number of generic speakers to separate (default: 2)",
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


def _decode_mono_audio(path: Path, target_rate: int = 16_000) -> tuple[Any, int]:
    """Decode a media file to mono audio without invoking an external service."""

    import av
    import numpy as np
    from scipy.signal import resample_poly

    with av.open(str(path)) as container:
        stream = next(
            (item for item in container.streams if item.type == "audio"), None
        )
        if stream is None or stream.sample_rate is None:
            raise RuntimeError(f"Media file has no audio stream: {path}")

        chunks = []
        for frame in container.decode(stream):
            samples = frame.to_ndarray()
            if samples.ndim == 2:
                samples = samples.mean(axis=0)
            if np.issubdtype(samples.dtype, np.integer):
                samples = samples.astype(np.float32) / np.iinfo(samples.dtype).max
            else:
                samples = samples.astype(np.float32)
            chunks.append(samples)

    if not chunks:
        raise RuntimeError(f"Media file contains no decodable audio: {path}")

    audio = np.concatenate(chunks)
    source_rate = int(stream.sample_rate)
    if source_rate != target_rate:
        audio = resample_poly(audio, target_rate, source_rate).astype(np.float32)
    return audio, target_rate


def _segment_features(
    audio: Any, sample_rate: int, segments: list[dict[str, Any]]
) -> Any:
    """Create simple timbre features for each speech segment.

    This is intentionally a small local heuristic, not a pretrained diarization
    model. It is useful for separating distinct voices in a small conversation,
    but it should not be treated as identity verification.
    """

    import librosa
    import numpy as np

    features = []
    for segment in segments:
        start = max(0, int((segment["start"] - 0.15) * sample_rate))
        end = min(len(audio), int((segment["end"] + 0.15) * sample_rate))
        clip = audio[start:end]
        minimum_samples = int(0.5 * sample_rate)
        if len(clip) < minimum_samples:
            clip = np.pad(clip, (0, minimum_samples - len(clip)))

        mfcc = librosa.feature.mfcc(
            y=clip,
            sr=sample_rate,
            n_mfcc=13,
            n_fft=512,
            hop_length=160,
            n_mels=26,
        )
        delta = librosa.feature.delta(mfcc)
        features.append(
            np.concatenate(
                [
                    mfcc.mean(axis=1),
                    mfcc.std(axis=1),
                    delta.mean(axis=1),
                    delta.std(axis=1),
                ]
            )
        )
    return np.asarray(features, dtype=np.float32)


def diarize_segments(
    path: Path, segments: list[dict[str, Any]], speaker_count: int
) -> list[dict[str, Any]]:
    """Attach generic speaker labels using local acoustic-feature clustering."""

    if len(segments) < speaker_count:
        raise RuntimeError(
            f"Need at least {speaker_count} transcript segments for diarization"
        )

    try:
        import numpy as np
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "Diarization dependencies are not installed; run `pixi install` first"
        ) from error

    audio, sample_rate = _decode_mono_audio(path)
    features = _segment_features(audio, sample_rate, segments)
    durations = np.asarray(
        [segment["end"] - segment["start"] for segment in segments]
    )
    fit_indices = np.flatnonzero(durations >= 0.75)
    if len(fit_indices) < speaker_count:
        fit_indices = np.arange(len(segments))

    scaled = StandardScaler().fit_transform(features)
    clustering = KMeans(
        n_clusters=speaker_count,
        n_init=10,
        random_state=0,
    ).fit(scaled[fit_indices])
    labels = clustering.predict(scaled)

    return [
        {**segment, "speaker": f"SPEAKER_{int(label):02d}"}
        for segment, label in zip(segments, labels)
    ]


def text_output(segments: list[dict[str, Any]]) -> str:
    entries = []
    for segment in segments:
        speaker = segment.get("speaker")
        if speaker:
            timestamp = (
                f"{seconds_to_timestamp(segment['start'])}–"
                f"{seconds_to_timestamp(segment['end'])}"
            )
            entries.append(f"[{timestamp}] {speaker}\n{segment['text']}")
        else:
            entries.append(
                f"[{seconds_to_timestamp(segment['start'])}] {segment['text']}"
            )
    return "\n\n".join(entries) + "\n"


def srt_output(segments: list[dict[str, Any]]) -> str:
    entries = []
    for index, segment in enumerate(segments, start=1):
        start = seconds_to_timestamp(segment["start"], decimal=",")
        end = seconds_to_timestamp(segment["end"], decimal=",")
        speaker = f"{segment['speaker']}: " if segment.get("speaker") else ""
        entries.append(f"{index}\n{start} --> {end}\n{speaker}{segment['text']}\n")
    return "\n".join(entries)


def json_output(
    segments: list[dict[str, Any]], *, model: str, language: str | None
) -> str:
    speakers = sorted(
        {segment["speaker"] for segment in segments if "speaker" in segment}
    )
    payload: dict[str, Any] = {
        "model": model,
        "language": language,
        "segments": segments,
    }
    if speakers:
        payload["speakers"] = speakers
    return json.dumps(
        payload,
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
        collected = collect_segments(segments)
        if not args.no_diarize:
            collected = diarize_segments(path, collected, args.speakers)
        return collected, info

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
