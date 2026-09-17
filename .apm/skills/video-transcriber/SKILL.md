---
name: video-transcriber
description: Transcribe a local audio/video file or supported video URL with timestamps and speaker diarization. Use for producing transcripts, choosing speed/quality settings, and diagnosing this bundled local transcription pipeline.
---

# Video transcriber

Use the deterministic CLI in this directory; do not reimplement inference in ad
hoc agent code.
Pixi and initial network access are prerequisites.
The bundled manifest and lock own Python, FFmpeg, and inference dependencies.
Resolve this directory from the loaded skill location, not the current project.
Use absolute paths for user inputs and outputs.

## Run

Run
`pixi exec --spec python=3.12 -- python <skill-dir>/scripts/launch.py --setup`
on first use in an environment or after a runtime failure.
It checks imports, FFmpeg, hardware, credentials, and online Community-1 access
without downloading model weights.
Present all returned human actions together as one concise setup checklist, then
rerun after the user completes it.
Use `--doctor` for offline import checks.
Setup success does not prove inference or transcription quality.

Run:

```console
pixi exec --spec python=3.12 -- python <skill-dir>/scripts/launch.py \
  '<absolute-file-or-url>' --quality balanced \
  --output '<absolute-output-directory>/transcript.txt'
```

Use `balanced` unless the user prioritizes speed or accuracy; choose `fast` or
`accurate` accordingly.
All presets include alignment and diarization.
They are starting configurations, not benchmarked quality guarantees.
Preserve explicit model, language, and device choices.
The CLI reports resolved settings.

Speaker count is estimated unless supplied.
Use `--speakers 1` for a known single speaker, or
`--min-speakers N --max-speakers N` for bounds.
Labels are anonymous; do not infer names from voices.
Do not disable diarization or reduce quality to recover a failure without the
user's agreement.
Ordinary dependency installation through the locked Pixi environment is part of
running this skill.

For first-time diarization, the user must accept the conditions at
[Community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
and supply credentials through `HF_TOKEN`, an existing Hugging Face login, or
`--token-file /private/path` (pass that option to setup and transcription).
Do not ask them to paste a token into chat, write it into this skill, or include
it in command arguments.
Cached models are reused; access is checked by loading the diarizer before
transcription.
Inference runs locally after downloads.

Return links to the transcript and JSON sidecar.
Inspect the `.run.json` status before claiming success.
JSON preserves word timings and speaker turns; text/SRT split on word-level
speaker changes.
`UNKNOWN` means no supported assignment.
A nonzero exit with `partial` means useful text was saved but the requested
pipeline did not complete.
Do not present it as a complete diarized transcript.

## Recover and improve

Read [operations.md](references/operations.md) for platform limits, artifacts,
and failure recovery.
Read [maintenance.md](references/maintenance.md) when fixing a demonstrated
issue or assessing an upstream upgrade.

Use run reports and user corrections as evidence.
Improve the canonical CLI, skill guidance, and regression coverage together
where relevant.
Never edit a managed deployed copy, rewrite transcript content as a software
fix, or turn one hardware-specific workaround into the global default.
If the canonical source is unavailable, provide a reproducible issue and a
concrete proposed change.
