#!/usr/bin/env python3
"""Pinned reference preparation, measured inference, scoring, and comparisons."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import unicodedata
import uuid
import wave
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / ".apm/skills/video-transcriber"
PROTOCOL = {
    "version": 1,
    "text": "Unicode NFKC, lowercase, delete Unicode punctuation, collapse whitespace; retain fillers; no number expansion",
    "word_order": "reference by (start, speaker, end); hypothesis in ASR order; overlap order can increase WER",
    "boundary_words": "include words whose midpoint is in the half-open excerpt, clip timings",
    "diarization": "AMI only_words; optimal speaker mapping; collar=0 seconds; include overlap; whole excerpt UEM",
    "timing": "absolute boundary error of correctly recognized 1:1 normalized words; excludes substitutions/deletions/insertions",
    "memory": "sampled sum of process-tree RSS every 0.1s; may double-count shared memory and miss brief peaks",
}


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def git(directory, *args):
    return subprocess.check_output(
        ["git", "-C", str(directory), *args], text=True
    ).strip()


def require_dataset(dataset):
    if not (dataset / ".datalad/config").is_file():
        raise ValueError(
            "Create a separate DataLad dataset first; see benchmarks/README.md"
        )
    if dataset == ROOT:
        raise ValueError("Benchmark data must be separate from the source repository")


def normalize(text):
    text = unicodedata.normalize("NFKC", text).lower()
    return " ".join(
        "".join(c for c in text if not unicodedata.category(c).startswith("P")).split()
    )


def reference_words(archive, meeting, start, end):
    words = []
    with zipfile.ZipFile(archive) as bundle:
        for speaker in "ABCD":
            tree = ET.fromstring(bundle.read(f"words/{meeting}.{speaker}.words.xml"))
            for word in tree.findall("w"):
                if (
                    word.get("punc") == "true"
                    or not word.get("starttime")
                    or not word.get("endtime")
                ):
                    continue
                a, b = float(word.get("starttime")), float(word.get("endtime"))
                if start <= (a + b) / 2 < end and word.text and normalize(word.text):
                    words.append(
                        {
                            "start": max(a, start) - start,
                            "end": min(b, end) - start,
                            "word": word.text,
                            "speaker": speaker,
                        }
                    )
    return sorted(words, key=lambda w: (w["start"], w["speaker"], w["end"]))


def reference_turns(path, start, end):
    turns = []
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if not fields or fields[0] != "SPEAKER":
            continue
        a = max(float(fields[3]), start)
        b = min(float(fields[3]) + float(fields[4]), end)
        if b > a:
            turns.append({"start": a - start, "end": b - start, "speaker": fields[7]})
    return turns


def prepare(dataset, manifest):
    require_dataset(dataset)
    spec = read(manifest)
    for source in spec["sources"]:
        target = dataset / source["path"]
        if target.is_symlink() and not target.exists():
            subprocess.run(
                ["datalad", "-C", str(dataset), "get", source["path"]], check=True
            )
        elif not target.exists():
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(dataset),
                    "annex",
                    "addurl",
                    "--file",
                    source["path"],
                    source["url"],
                ],
                check=True,
            )
        if sha(target) != source["sha256"]:
            raise ValueError(f"Source checksum mismatch: {source['path']}")
    target = dataset / "fixtures" / spec["id"]
    reference = target / "reference.json"
    audio = target / "audio.wav"
    if target.exists():
        existing = read(reference)
        if existing["manifest_sha256"] != sha(manifest) or existing[
            "audio_sha256"
        ] != sha(audio):
            raise ValueError(
                "Existing fixture differs; use a new fixture ID rather than overwrite"
            )
        return target
    target.mkdir(parents=True)
    start, end = spec["start_seconds"], spec["end_seconds"]
    with wave.open(str(dataset / spec["sources"][0]["path"]), "rb") as source:
        rate = source.getframerate()
        source.setpos(round(start * rate))
        frames = source.readframes(round((end - start) * rate))
        if (
            len(frames)
            != round((end - start) * rate)
            * source.getnchannels()
            * source.getsampwidth()
        ):
            raise ValueError("Excerpt extends past the audio")
        with wave.open(str(audio), "wb") as dest:
            dest.setparams(source.getparams())
            dest.writeframes(frames)
    words = reference_words(
        dataset / spec["sources"][1]["path"], spec["meeting"], start, end
    )
    turns = reference_turns(dataset / spec["sources"][2]["path"], start, end)
    write(
        reference,
        {
            "id": spec["id"],
            "manifest_sha256": sha(manifest),
            "audio_sha256": sha(audio),
            "duration_seconds": end - start,
            "words": words,
            "turns": turns,
            "license": spec["license"],
            "attribution": spec["attribution"],
        },
    )
    subprocess.run(
        ["git", "-C", str(dataset), "annex", "add", str(target.relative_to(dataset))],
        check=True,
    )
    return target


def score(reference, hypothesis):
    import jiwer
    from pyannote.core import Annotation, Segment, Timeline
    from pyannote.metrics.diarization import DiarizationErrorRate

    ref_words = reference["words"]
    hyp_words = [
        word for seg in hypothesis["segments"] for word in seg.get("words", [])
    ]
    ref_text = normalize(" ".join(w["word"] for w in ref_words))
    hyp_text = normalize(" ".join(seg["text"] for seg in hypothesis["segments"]))
    asr = jiwer.process_words(ref_text, hyp_text)
    cer = jiwer.cer(ref_text, hyp_text)

    def annotation(turns):
        result = Annotation(uri=reference["id"])
        for i, turn in enumerate(turns):
            if turn["end"] > turn["start"]:
                result[Segment(turn["start"], turn["end"]), i] = turn["speaker"]
        return result

    duration = reference["duration_seconds"]
    ref = annotation(reference["turns"])
    hyp = annotation(hypothesis.get("speaker_turns", []))
    metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    der = metric(ref, hyp, uem=Timeline([Segment(0, duration)]), detailed=True)
    errors = []
    # Score timestamp errors independently of segment text; words without times
    # still count in WER above but cannot contribute to boundary error.
    rw = [w for w in ref_words if len(normalize(w["word"]).split()) == 1]
    hw = [w for w in hyp_words if len(normalize(w["word"]).split()) == 1]
    alignment = jiwer.process_words(
        " ".join(normalize(w["word"]) for w in rw),
        " ".join(normalize(w["word"]) for w in hw),
    )
    matches = 0
    for chunk in alignment.alignments[0]:
        if chunk.type != "equal":
            continue
        for r, h in zip(
            rw[chunk.ref_start_idx : chunk.ref_end_idx],
            hw[chunk.hyp_start_idx : chunk.hyp_end_idx],
        ):
            if all(k in h for k in ("start", "end")):
                errors.extend([abs(r[k] - h[k]) for k in ("start", "end")])
                matches += 1
    return {
        "wer": asr.wer,
        "cer": cer,
        "word_counts": {
            "reference": len(asr.references[0]),
            "substitutions": asr.substitutions,
            "deletions": asr.deletions,
            "insertions": asr.insertions,
            "hits": asr.hits,
        },
        "der": float(der["diarization error rate"]),
        "der_components_seconds": {
            k: float(v) for k, v in der.items() if k != "diarization error rate"
        },
        "reference_speakers": len(ref.labels()),
        "hypothesis_speakers": len(hyp.labels()),
        "speaker_count_absolute_error": abs(len(ref.labels()) - len(hyp.labels())),
        "matched_timed_words": matches,
        "word_boundary_mae_seconds": statistics.mean(errors) if errors else None,
        "word_boundary_p95_seconds": sorted(errors)[
            min(len(errors) - 1, int(0.95 * len(errors)))
        ]
        if errors
        else None,
    }


def measure(command, directory):
    import psutil

    started = time.perf_counter()
    peak = 0
    with (directory / "process.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        root = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                processes = [root, *root.children(recursive=True)]
                rss = 0
                for child in processes:
                    try:
                        rss += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                peak = max(peak, rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(0.1)
        return process.returncode, time.perf_counter() - started, peak


def model_cache_inventory():
    # Record only this pipeline's public model families, never arbitrary user repos.
    hub = Path(
        os.environ.get(
            "HF_HUB_CACHE",
            Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub",
        )
    )
    entries = {}
    for family in (
        "models--pyannote--*",
        "models--Systran--faster-whisper-*",
        "models--mobiuslabsgmbh--faster-whisper-*",
    ):
        for repo in sorted(hub.glob(family)):
            for ref in sorted((repo / "refs").glob("*")):
                if ref.is_file():
                    entries[f"{repo.name}/refs/{ref.name}"] = ref.read_text().strip()
    torch_home = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache/torch"))
    for pattern in (
        "hub/checkpoints/wav2vec2*",
        "hub/snakers4_silero-vad_master/src/silero_vad/data/*",
    ):
        for path in sorted(torch_home.glob(pattern)):
            if path.is_file():
                entries[str(path.relative_to(torch_home))] = sha(path)
    return entries


def hardware():
    import psutil

    processor = platform.processor()
    if platform.system() == "Darwin":
        processor = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    elif platform.system() == "Linux" and Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                processor = line.split(":", 1)[1].strip()
                break
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": processor,
        "cpu_count": os.cpu_count(),
        "memory_bytes": psutil.virtual_memory().total,
    }


def run_benchmark(args):
    dataset = args.dataset.resolve()
    require_dataset(dataset)
    spec = read(args.manifest)
    fixture = dataset / "fixtures" / spec["id"]
    reference = read(fixture / "reference.json")
    if (
        reference["manifest_sha256"] != sha(args.manifest)
        or sha(fixture / "audio.wav") != reference["audio_sha256"]
    ):
        raise ValueError("Fixture checksum differs from the pinned reference")
    records = []
    for iteration in range(args.repeat):
        run_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + uuid.uuid4().hex[:8]
        )
        directory = dataset / "runs" / run_id
        directory.mkdir(parents=True)
        output = directory / "transcript.json"
        command = [
            sys.executable,
            str(SKILL / "scripts/launch.py"),
            str(fixture / "audio.wav"),
            "--quality",
            args.quality,
            "--language",
            spec["language"],
            "--format",
            "json",
            "--output",
            str(output),
        ]
        if args.token_file:
            command += ["--token-file", str(args.token_file.resolve())]
        record = {
            "schema_version": 1,
            "run_id": run_id,
            "fixture_id": spec["id"],
            "fixture_manifest_sha256": sha(args.manifest),
            "reference_sha256": sha(fixture / "reference.json"),
            "audio_sha256": reference["audio_sha256"],
            "protocol": PROTOCOL,
            "protocol_sha256": digest(PROTOCOL),
            "quality": args.quality,
            "cache_label": args.cache_label,
            "iteration": iteration + 1,
            "source_commit": git(ROOT, "rev-parse", "HEAD"),
            "source_dirty": bool(
                git(ROOT, "status", "--porcelain", "--untracked-files=normal")
            ),
            "runner_sha256": sha(SKILL / "scripts/transcribe.py"),
            "benchmark_sha256": sha(__file__),
            "runtime_lock_sha256": sha(SKILL / "pixi.lock"),
            "evaluation_lock_sha256": sha(ROOT / "pixi.lock"),
            "dataset_commit": git(dataset, "rev-parse", "HEAD"),
            "dataset_dirty": bool(git(dataset, "status", "--porcelain")),
            "hardware": hardware(),
            "metric_versions": {
                p: importlib.metadata.version(p)
                for p in ("jiwer", "pyannote.metrics", "psutil")
            },
            "model_cache_before": model_cache_inventory(),
        }
        print(f"Running {run_id} ({iteration + 1}/{args.repeat})", flush=True)
        code, elapsed, peak = measure(command, directory)
        record.update(
            returncode=code,
            wall_seconds=elapsed,
            peak_process_tree_rss_bytes=peak,
            real_time_factor=elapsed / reference["duration_seconds"],
            status="failed",
            metrics=None,
            model_cache_after=model_cache_inventory(),
        )
        # Preserve timing/failure evidence even if scoring itself fails.
        write(directory / "result.json", record)
        sidecar = Path(str(output) + ".run.json")
        if sidecar.exists():
            record["pipeline"] = read(sidecar)
        if (
            code == 0
            and output.exists()
            and record.get("pipeline", {}).get("status") == "complete"
        ):
            hypothesis = read(output)
            if hypothesis["run"]["diarization"] == "complete":
                try:
                    record["metrics"] = score(reference, hypothesis)
                    record["status"] = "complete"
                except Exception as error:
                    record["status"] = "scoring_failed"
                    record["scoring_error_type"] = type(error).__name__
        write(directory / "result.json", record)
        subprocess.run(
            [
                "git",
                "-C",
                str(dataset),
                "annex",
                "add",
                str(directory.relative_to(dataset)),
            ],
            check=True,
        )
        records.append(record)
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "status": record["status"],
                    "wall_seconds": elapsed,
                    "metrics": record["metrics"],
                },
                indent=2,
            ),
            flush=True,
        )
    return 0 if all(r["status"] == "complete" for r in records) else 1


def compare(paths):
    records = [read(p) for p in paths]
    keys = (
        "fixture_manifest_sha256",
        "reference_sha256",
        "audio_sha256",
        "protocol_sha256",
        "metric_versions",
        "hardware",
        "cache_label",
    )
    for key in keys:
        if any(r[key] != records[0][key] for r in records[1:]):
            raise ValueError(f"Incomparable runs: {key} differs")
    rows = []
    for r in records:
        rows.append(
            {
                k: r[k]
                for k in (
                    "run_id",
                    "source_commit",
                    "quality",
                    "status",
                    "wall_seconds",
                    "real_time_factor",
                    "peak_process_tree_rss_bytes",
                    "metrics",
                )
            }
        )
    groups = {}
    for r in records:
        key = (
            r["source_commit"],
            r["runner_sha256"],
            r["benchmark_sha256"],
            r["quality"],
        )
        groups.setdefault(key, []).append(r)
    summaries = []
    for key, group in groups.items():
        good = [r for r in group if r["status"] == "complete"]
        summaries.append(
            {
                "source_commit": key[0],
                "runner_sha256": key[1],
                "benchmark_sha256": key[2],
                "quality": key[3],
                "successful_runs": len(good),
                "failed_runs": len(group) - len(good),
                "median_wall_seconds": statistics.median(
                    r["wall_seconds"] for r in good
                )
                if good
                else None,
                "median_wer": statistics.median(r["metrics"]["wer"] for r in good)
                if good
                else None,
                "median_der": statistics.median(r["metrics"]["der"] for r in good)
                if good
                else None,
            }
        )
    # Failed runs stay visible, and are not averaged into success metrics.
    print(
        json.dumps(
            {"comparable_by": keys, "runs": rows, "summaries": summaries}, indent=2
        )
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("prepare", "run"):
        p = sub.add_parser(action)
        p.add_argument("--dataset", type=Path, default=ROOT / ".datasets/benchmarks")
        p.add_argument(
            "--manifest", type=Path, default=ROOT / "benchmarks/ami-mini.json"
        )
        if action == "run":
            p.add_argument(
                "--quality", choices=("fast", "balanced", "accurate"), default="fast"
            )
            p.add_argument("--repeat", type=int, default=1)
            p.add_argument(
                "--cache-label",
                choices=("cold", "warm", "uncontrolled"),
                default="uncontrolled",
            )
            p.add_argument("--token-file", type=Path)
    p = sub.add_parser("compare")
    p.add_argument("results", nargs="+", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.action == "prepare":
            print(prepare(args.dataset.resolve(), args.manifest))
        elif args.action == "run":
            if args.repeat < 1:
                parser.error("--repeat must be positive")
            return run_benchmark(args)
        else:
            compare(args.results)
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"benchmark error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
