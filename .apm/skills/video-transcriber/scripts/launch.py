#!/usr/bin/env python3
"""Materialize the exact runtime in a cache, keeping the skill deployable."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

RUNTIME_FILES = ("pixi.toml", "pixi.lock")


def prepare_runtime(skill, cache):
    files = {name: (skill / name).read_bytes() for name in RUNTIME_FILES}
    digest = hashlib.sha256()
    for name, content in files.items():
        digest.update(name.encode() + b"\0" + content + b"\0")
    destination = cache / digest.hexdigest()[:24]
    for name, content in files.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        # Immutable content-addressed files; concurrent identical launches write
        # identical bytes, and the atomic replace prevents partial manifest reads.
        if not target.exists() or target.read_bytes() != content:
            import tempfile

            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(content)
            try:
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
    return destination


def cache_root():
    override = os.environ.get("VIDEO_TRANSCRIBER_CACHE")
    if override:
        return Path(override).expanduser().resolve()
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library/Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "video-transcriber"


def main(argv=None):
    pixi = shutil.which("pixi")
    if not pixi:
        print("error: install Pixi before running this skill", file=sys.stderr)
        return 1
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["--"]:
        args = args[1:]
    # Pixi tasks run from their manifest directory. Resolve relative CLI paths
    # against the caller before switching to the cached workspace.
    for index, value in enumerate(args):
        if index > 0 and args[index - 1] in {"--output", "--token-file"}:
            args[index] = str(Path(value).expanduser().resolve())
        elif value.startswith(("--output=", "--token-file=")):
            args[index] = (
                value.split("=", 1)[0]
                + "="
                + str(Path(value.split("=", 1)[1]).expanduser().resolve())
            )
    # Let argparse distinguish input from option values using the same parser.
    import runpy

    skill = Path(__file__).resolve().parents[1]
    parser = runpy.run_path(str(skill / "scripts/transcribe.py"))["parse_args"]
    parsed = parser(args)
    if parsed.input:
        source = parsed.input
        if not source.startswith(("https://", "http://")):
            args[args.index(source)] = str(Path(source).expanduser().resolve())
        if parsed.output is None:
            helpers = runpy.run_path(str(skill / "scripts/transcribe.py"))
            output = helpers["default_output"](source, parsed.format).resolve()
            args.extend(["--output", str(output)])
    runtime = prepare_runtime(skill, cache_root())
    return subprocess.call(
        [
            pixi,
            "run",
            "--manifest-path",
            str(runtime / "pixi.toml"),
            "--locked",
            "python",
            str(skill / "scripts/transcribe.py"),
            *args,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
