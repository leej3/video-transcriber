# Operation and recovery

## Environment contract

The runtime targets Apple Silicon macOS, Linux x86-64, and Windows x86-64.
The lock resolves dependencies per platform; execution still needs validation on
each platform.
Intel macOS is excluded because this
WhisperX/PyTorch generation has no matching modern Torch wheel. Pixi cannot fix that upstream limitation. The project's lightweight development/test
environment can still run there.

FFmpeg comes from conda-forge; do not prescribe Homebrew, apt, or global pip as
the default repair.
Use the bundled manifest with `--locked`.
For resolution or binary failures, capture the platform, package versions, and
failing stage.
Test a version change in the canonical manifest and regenerate the lock; do not
silently run an unpinned installation.

`auto` selects CUDA only when Torch and CTranslate2 both detect it; otherwise it
uses CPU with int8 ASR.
Apple Silicon currently uses CPU, not Metal acceleration.
CUDA still needs a compatible host NVIDIA driver and runtime libraries;
detecting a GPU does not prove all inference kernels load.
The CPU path is the portable baseline.
Suggest an explicit CPU retry for CUDA failures.
MLX is a future candidate, not an implemented backend.

First use downloads dependencies and models; allow for disk, memory, and time.
Hugging Face model caches persist outside the disposable media directory.
WhisperX may also cache VAD/alignment resources through upstream libraries.
The dependency lock does not pin every downloaded model revision; reports record
model names and package versions, not full model-byte reproducibility.

## Inputs and outputs

Local files bypass downloading.
URLs use yt-dlp, audio-first, with bounded network
timeouts/retries and no playlist processing. Website support can change; an unsupported/authenticated
URL may need a user-provided local file.
Cookies and DRM bypass are not implemented.
Temporary URL downloads are deleted after media decoding.
Processing retains the decoded audio in memory: long recordings need enough RAM.
Existing captions are not currently reused.

Examples (prepend the Pixi command from SKILL.md):

```console
transcribe /absolute/meeting.mp4 --quality fast --speakers 2 --output /absolute/meeting.txt
transcribe 'https://example.org/meeting.mp4' --quality accurate --language en --output /absolute/meeting.srt --format srt
transcribe /absolute/meeting.wav --format json --output /absolute/meeting.json
```

A text/SRT result has `<output>.json` with structured transcript data and
`<output>.run.json` with configuration, versions, stage timings, and status.
JSON output uses the requested `.json` path plus `<output>.run.json`.
Run reports omit media paths, URLs, tokens, and transcript text.
Upstream console messages can contain source details; sanitize them before
sharing.
Output files are written atomically one at a time; the set is not transactional.
Existing outputs require `--overwrite`.
Concurrent runs should use distinct output names.
ASR and alignment checkpoints are saved to the same output paths, with `partial`
status, before later processing.
Automatic resume is not yet implemented; a retry reruns the pipeline.

## Failure response

- Environment/import/FFmpeg: run doctor and restore the locked Pixi environment.
- Gated model access: explain the model conditions and environment/login route;
  never print credentials.
  No ASR is attempted before diarizer loading succeeds.
- Unsupported URL: retain the original request and offer a local-file route.
- Out of memory: use a smaller explicit batch size first; request agreement
  before reducing model quality or changing device when the user specified it.
- Alignment/diarization failure: return the partial transcript and identify the
  failed stage.
  Unsupported alignment languages need an upstream-compatible alignment model or
  a future tested adapter, not fabricated word timings.
- Silence: successful empty ASR is a valid result; diarization reports
  `no_speech`.
  Silence still incurs model preflight; there is no hand-rolled silence
  detector.
- Incorrect speakers/text: preserve the source transcript, capture a sanitized
  example with permission, and distinguish recognition errors from attribution.

After a repeated identical failure, stop blind retries.
Keep artifacts and provide the smallest reproducible case and next action
supported by evidence.

## Runtime cache and deployment

Use `scripts/launch.py` rather than running Pixi directly inside the skill.
The launcher copies the exact manifest and lock into a content-addressed user
cache and invokes the skill's CLI with `pixi run --locked` there.
This avoids APM copying a local `.pixi` environment into the skill deployment.
Canonical and deployed copies share a runtime when their content matches.
`VIDEO_TRANSCRIBER_CACHE` can choose a different writable cache directory.
Old runtime directories can be removed when no run uses them; model caches have
their own upstream lifecycle.
Serialize environment setup: a concurrent reinstall was observed to trigger a
Pixi 0.76.2 stale-package uninstall failure during development.

## Initial setup

Use `launch.py --setup` before the first recording.
It returns a single list of required actions covering the runtime and online
Community-1 access.
A valid Hugging Face token can still receive 403 until model conditions are
accepted or fine-grained permissions allow the repository.
Present these steps together; do not ask the user for credentials in chat.
`--token-file` reads a private file without exposing its contents in command
arguments or reports.

On the tested Apple Silicon lock, importing Torch before the speech stack caused
a fatal duplicate-OpenMP abort.
The runner now initializes WhisperX ASR first, matching setup and doctor.
Do not set `KMP_DUPLICATE_LIB_OK` to suppress the runtime's correctness check.
Reproduce initialization and a tensor operation when changing native
dependencies; an import-only doctor cannot prove inference.
