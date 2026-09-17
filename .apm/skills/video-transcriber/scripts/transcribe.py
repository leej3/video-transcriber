#!/usr/bin/env python3
"""Versioned orchestration around yt-dlp, WhisperX, and pyannote."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

VERSION = "0.2.0"
PRESETS = {"fast": ("small", 1), "balanced": ("turbo", 5), "accurate": ("large-v3", 5)}
DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"


def positive_int(value: str) -> int:
    count = int(value)
    if count < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return count


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", help="Local audio/video or HTTP(S) URL")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--format", choices=("text", "srt", "json"), default="text")
    parser.add_argument("--quality", choices=PRESETS, default="balanced")
    parser.add_argument("--model", help="Override the preset's Whisper model")
    parser.add_argument("--language")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--compute-type", default="auto")
    parser.add_argument("--batch-size", type=positive_int, default=4)
    parser.add_argument("--no-diarize", action="store_true")
    parser.add_argument(
        "--speakers", type=positive_int, help="Exact count; default: estimate"
    )
    parser.add_argument("--min-speakers", type=positive_int)
    parser.add_argument("--max-speakers", type=positive_int)
    parser.add_argument(
        "--doctor", action="store_true", help="Check runtime without downloading models"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Check runtime and online model access; list all setup actions",
    )
    parser.add_argument(
        "--token-file",
        type=Path,
        help="Read HF credentials from a private local file (never print them)",
    )
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["--"]:
        args = args[1:]
    args = parser.parse_args(args)
    if not args.input and not (args.doctor or args.setup):
        parser.error("input is required unless --doctor or --setup is used")
    if args.speakers and (args.min_speakers or args.max_speakers):
        parser.error("use --speakers or speaker bounds, not both")
    if (
        args.min_speakers
        and args.max_speakers
        and args.min_speakers > args.max_speakers
    ):
        parser.error("--min-speakers must not exceed --max-speakers")
    return args


def is_url(value: str) -> bool:
    parsed = urllib.parse.urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


@contextmanager
def media_path(source: str):
    if not is_url(source):
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Media file does not exist: {path}")
        yield path
        return
    from yt_dlp import YoutubeDL

    with tempfile.TemporaryDirectory(prefix="video-transcriber-") as directory:
        options = {
            "format": "bestaudio/best",
            "noplaylist": True,
            "extract_flat": "in_playlist",
            "outtmpl": str(Path(directory) / "input.%(ext)s"),
            "socket_timeout": 30,
            "retries": 3,
            "quiet": True,
        }
        with YoutubeDL(options) as downloader:
            info = downloader.extract_info(source, download=False)
            if not info or info.get("_type") in {"playlist", "multi_video"}:
                raise RuntimeError("Provide a single video URL, not a playlist")
            info = downloader.process_ie_result(info, download=True)
            path = Path(downloader.prepare_filename(info))
        if not path.is_file():
            raise RuntimeError("Downloader did not produce a media file")
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
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


def default_output(source: str, output_format: str) -> Path:
    name = Path(urllib.parse.urlparse(source).path).stem or "transcript"
    suffix = {"text": ".txt", "srt": ".srt", "json": ".json"}[output_format]
    return Path(f"{name}{suffix}")


def versions():
    result = {"runner": VERSION, "python": platform.python_version()}
    for package in ("whisperx", "faster-whisper", "pyannote.audio", "torch", "yt-dlp"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def resolve_device(requested):
    if requested == "cpu":
        return "cpu"
    import ctranslate2
    import torch

    available = torch.cuda.is_available() and ctranslate2.get_cuda_device_count() > 0
    if requested == "cuda" and not available:
        raise RuntimeError(
            "CUDA unavailable to both Torch and CTranslate2; use --device cpu or check the NVIDIA driver"
        )
    return ("cuda" if available else "cpu") if requested == "auto" else requested


def doctor():
    report = {
        "versions": versions(),
        "system": platform.system(),
        "machine": platform.machine(),
        "ffmpeg": bool(shutil.which("ffmpeg")),
        "status": "ready",
        "model_access": "not_checked",
    }
    try:
        if not report["ffmpeg"]:
            raise RuntimeError("FFmpeg missing; run through the bundled Pixi manifest")
        from huggingface_hub import get_token
        from whisperx.alignment import align  # noqa: F401
        from whisperx.asr import load_model  # noqa: F401
        from whisperx.diarize import DiarizationPipeline  # noqa: F401
        from yt_dlp import YoutubeDL  # noqa: F401

        report["hf_credentials_present"] = bool(get_token())
        report["device"] = resolve_device("auto")
    except Exception as error:
        report.update(
            status="unavailable", error_type=type(error).__name__, detail=str(error)
        )
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "ready" else 1


def setup():
    """Collect setup failures without downloading weights or exposing credentials."""
    checks = {}
    actions = []
    checks["ffmpeg"] = "ready" if shutil.which("ffmpeg") else "missing"
    if checks["ffmpeg"] == "missing":
        actions.append("Run setup through the bundled Pixi launcher to install FFmpeg.")
    for module in ("whisperx.asr", "whisperx.alignment", "whisperx.diarize", "yt_dlp"):
        try:
            importlib.import_module(module)
            checks[module] = "ready"
        except Exception as error:
            checks[module] = type(error).__name__
    if any(
        checks.get(module) != "ready"
        for module in (
            "whisperx.asr",
            "whisperx.alignment",
            "whisperx.diarize",
            "yt_dlp",
        )
    ):
        actions.append("Repair the locked Pixi runtime; see the failed import checks.")
    try:
        checks["device"] = resolve_device("auto")
    except Exception as error:
        checks["device"] = type(error).__name__
        actions.append(
            "Repair Torch/CTranslate2 hardware detection in the locked runtime."
        )
    try:
        from huggingface_hub import get_token

        token = get_token()
        checks["credentials"] = "present" if token else "missing"
        if not token:
            actions.append(
                "Create a Hugging Face read token; supply HF_TOKEN, hf auth login, or --token-file /private/path. Do not paste it into chat."
            )
        request = urllib.request.Request(
            f"https://huggingface.co/{DIARIZATION_MODEL}/resolve/main/config.yaml",
            headers={"Authorization": f"Bearer {token}"} if token else {},
            method="HEAD",
        )
        try:
            with urllib.request.urlopen(request, timeout=30):
                checks["model_access"] = "ready"
        except urllib.error.HTTPError as error:
            checks["model_access"] = f"http_{error.code}"
            if error.code in (401, 403):
                actions.append(
                    f"Accept the conditions at https://huggingface.co/{DIARIZATION_MODEL} and allow this repository in any fine-grained token's read permissions."
                )
            else:
                actions.append(
                    "Hugging Face access failed; check service availability and retry setup."
                )
        except (urllib.error.URLError, TimeoutError):
            checks["model_access"] = "network_error"
            actions.append(
                "Check network/proxy access to huggingface.co, then retry setup."
            )
    except Exception as error:
        checks["credentials"] = type(error).__name__
        actions.append(
            "Repair huggingface_hub credential access in the locked runtime."
        )
    print(
        json.dumps(
            {
                "status": "action_required" if actions else "ready",
                "checks": checks,
                "actions": actions,
                "limits": "Checks Community-1 config access, not weight downloads or inference. ASR/alignment models download on first use; quality is not measured here.",
            },
            indent=2,
        )
    )
    return 1 if actions else 0


def atomic_write(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Sibling temporary file allows an atomic rename on the same filesystem.
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=".transcript-", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(contents)
            handle.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def output_paths(output):
    # Append to the full filename so names such as talk.txt and talk.srt cannot
    # accidentally overwrite one another's reports.
    data = output if output.suffix.lower() == ".json" else Path(str(output) + ".json")
    return data, Path(str(output) + ".run.json")


def speaker_turns(segments, language=None):
    """Render word-level speaker changes without inventing labels for gaps."""
    turns = []
    separator = "" if language in {"zh", "ja"} else " "
    for segment in segments:
        if not segment.get("words"):
            turns.append(segment)
            continue
        current = None
        for word in segment["words"]:
            speaker = word.get("speaker", "UNKNOWN")
            if current is None or current["speaker"] != speaker:
                current = {
                    "start": word.get("start", segment["start"]),
                    "end": word.get("end", segment["end"]),
                    "text": word["word"],
                    "speaker": speaker,
                }
                turns.append(current)
            else:
                current["text"] += separator + word["word"]
                current["end"] = word.get("end", segment["end"])
    return turns


def save_result(output, output_format, result, report):
    data_path, report_path = output_paths(output)
    payload = {**result, "schema_version": 1, "run": report}
    atomic_write(data_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    if output_format != "json":
        segments = result.get("segments", [])
        if report.get("diarization") == "complete":
            segments = speaker_turns(segments, result.get("language"))
        rendered = (
            srt_output(segments) if output_format == "srt" else text_output(segments)
        )
        atomic_write(output, rendered)
    atomic_write(report_path, json.dumps(report, indent=2) + "\n")


def run(args, output):
    # No transcript text, source URL, token, or local path enters the run report.
    report = {
        "schema_version": 1,
        "versions": versions(),
        "quality": args.quality,
        "system": platform.system(),
        "machine": platform.machine(),
        "status": "running",
        "stage": "preflight",
        "timings_seconds": {},
        "diarization": "disabled" if args.no_diarize else "pending",
    }
    result = {"segments": [], "language": args.language}
    completed_asr = False
    started = time.monotonic()

    def stage(name):
        report["stage"] = name
        print(f"[{name}]", file=sys.stderr)
        return time.monotonic()

    def finish(name, stage_start):
        report["timings_seconds"][name] = round(time.monotonic() - stage_start, 3)

    try:
        stage("input_validation")
        if not is_url(args.input) and not Path(args.input).expanduser().is_file():
            raise FileNotFoundError("Local input does not exist or is not a file")
        preflight_start = stage("preflight")
        if not shutil.which("ffmpeg"):
            raise RuntimeError("FFmpeg missing; use the bundled Pixi runtime")
        # Match setup/doctor initialization: Torch-first aborts with duplicate
        # libomp on the locked Apple Silicon runtime. Do not suppress OMP checks.
        importlib.import_module("whisperx.asr")
        import torch
        import whisperx
        from huggingface_hub import get_token
        from whisperx.diarize import DiarizationPipeline, assign_word_speakers

        device = resolve_device(args.device)
        compute_type = args.compute_type
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"
        model_name, beam_size = PRESETS[args.quality]
        model_name = args.model or model_name
        report.update(
            model=model_name,
            device=device,
            compute_type=compute_type,
            beam_size=beam_size,
            batch_size=args.batch_size,
            speakers=args.speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            diarization_model=None if args.no_diarize else DIARIZATION_MODEL,
        )
        print(
            f"{model_name} / {device} / {compute_type}; diarization={not args.no_diarize}",
            file=sys.stderr,
        )
        # Load gated weights before downloading media or paying the ASR cost.
        # Hold the diarizer on CPU until ASR/alignment release GPU memory.
        diarizer = None
        if not args.no_diarize:
            diarizer = DiarizationPipeline(
                model_name=DIARIZATION_MODEL, token=get_token(), device="cpu"
            )
        finish("preflight", preflight_start)
        acquisition_start = stage("input")
        with media_path(args.input) as path:
            audio = whisperx.load_audio(str(path))
        finish("input", acquisition_start)
        report["duration_seconds"] = round(len(audio) / 16000, 3)
        asr_start = stage("transcription")
        model = whisperx.load_model(
            model_name,
            device,
            compute_type=compute_type,
            language=args.language,
            vad_method="silero",
            asr_options={"beam_size": beam_size},
        )
        result = model.transcribe(
            audio, batch_size=args.batch_size, language=args.language
        )
        completed_asr = True
        finish("transcription", asr_start)
        report["status"] = "partial"
        save_result(output, args.format, result, report)
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        if result["segments"]:
            alignment_start = stage("alignment")
            align_model, metadata = whisperx.load_align_model(
                result["language"], device
            )
            aligned = whisperx.align(
                result["segments"], align_model, metadata, audio, device
            )
            result.update(aligned)
            finish("alignment", alignment_start)
            save_result(output, args.format, result, report)
            del align_model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            if diarizer is not None:
                diarization_start = stage("diarization")
                diarizer.model.to(torch.device(device))
                turns = diarizer(
                    audio,
                    num_speakers=args.speakers,
                    min_speakers=args.min_speakers,
                    max_speakers=args.max_speakers,
                )
                result = assign_word_speakers(turns, result)
                result["speaker_turns"] = turns[["start", "end", "speaker"]].to_dict(
                    "records"
                )
                report["diarization"] = "complete"
                finish("diarization", diarization_start)
        elif not args.no_diarize:
            report["diarization"] = "no_speech"
        report.update(status="complete", stage="complete")
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        save_result(output, args.format, result, report)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        report["status"] = "partial" if completed_asr else "failed"
        report["error_type"] = type(error).__name__
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        if completed_asr:
            save_result(output, args.format, result, report)
        else:
            atomic_write(output_paths(output)[1], json.dumps(report, indent=2) + "\n")
        print(f"error in {report['stage']}: {error}", file=sys.stderr)
        if completed_asr:
            print(
                f"Partial transcript preserved at {output}; requested pipeline did not complete.",
                file=sys.stderr,
            )
        elif not args.no_diarize and report["stage"] == "preflight":
            print(
                "Check --doctor, HF_TOKEN/login and Community-1 model access. Credentials stay outside the repository.",
                file=sys.stderr,
            )
        return 130 if isinstance(error, KeyboardInterrupt) else 1


def main(argv=None):
    args = parse_args(argv)
    if args.token_file:
        try:
            token = args.token_file.expanduser().read_text().strip()
            if not token or any(character.isspace() for character in token):
                raise ValueError("invalid token file")
            os.environ["HF_TOKEN"] = token
        except (OSError, ValueError):
            print(
                "error: --token-file must be a readable file containing one token",
                file=sys.stderr,
            )
            return 2
    if args.setup:
        return setup()
    if args.doctor:
        return doctor()
    output = (
        (args.output or default_output(args.input, args.format)).expanduser().resolve()
    )
    if (args.format == "json") != (output.suffix.lower() == ".json"):
        print(
            "error: --format json requires a .json output, and vice versa",
            file=sys.stderr,
        )
        return 2
    paths = {output, *output_paths(output)}
    if not is_url(args.input) and Path(args.input).expanduser().resolve() in paths:
        print("error: output must not replace the input", file=sys.stderr)
        return 2
    if not args.overwrite and any(path.exists() for path in paths):
        print(
            "error: output or sidecar exists; choose another name or use --overwrite",
            file=sys.stderr,
        )
        return 2
    try:
        return run(args, output)
    except OSError as error:
        print(f"error writing output: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
