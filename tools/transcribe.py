#!/usr/bin/env python3
"""Compatibility entry point; the portable skill owns the implementation."""

import runpy
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / ".apm"
    / "skills"
    / "video-transcriber"
    / "scripts"
    / "transcribe.py"
)
globals().update(
    {
        key: value
        for key, value in runpy.run_path(
            str(_SCRIPT), run_name="video_transcriber"
        ).items()
        if not key.startswith("__")
    }
)

if __name__ == "__main__":
    raise SystemExit(globals()["main"]())
