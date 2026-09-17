# Co-evolving the CLI and skill

The improvement loop is evidence-driven and user-triggered or part of resolving
an actual task failure.
It is not a background updater or permission to publish recordings, install
arbitrary tools, or silently change the model selection.

1. Inspect the run report, user correction, and relevant environment facts.
   Distinguish dependency/import failures, URL acquisition, ASR, alignment,
   speaker attribution, and presentation.
   Preserve what was actually observed.
2. Reproduce the smallest useful failure.
   Prefer a synthetic or public fixture; obtain permission before retaining user
   media.
   Keep only sanitized operational observations in durable project records;
   never retain credentials or transcript excerpts merely to create usage
   evidence.
3. Change deterministic behavior in the canonical script and add an
   outcome-based regression test.
   Change the skill/reference guidance only when agent decisions or recovery
   should change.
   Keep the deployed package self-contained.
4. Test the affected path with the locked runtime.
   Record fixture, hardware, package/model settings, cold versus warm cache,
   outcome, and limitations.
   Unit tests and runtime imports do not establish transcription quality.
5. Redeploy through project APM and audit the generated tree.
   Record actual skill use separately from installation in the project's
   skill-management workflow.
   If Workshop is available, use it for consideration, membership, and usage
   evidence; the runtime itself must remain usable without Workshop or APM.

## Upstream review

Review upstream changes when a failure suggests a fixed bug, the user asks for
an upgrade, or a maintenance task is explicitly scheduled.
Consult primary sources and compatibility notes before changing pinned packages:

- [WhisperX](https://github.com/m-bain/whisperX): ASR, alignment, speaker
  assignment.
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper):
  inference/quantization.
- [pyannote](https://github.com/pyannote/pyannote-audio): speaker pipeline and
  model access.
- [yt-dlp](https://github.com/yt-dlp/yt-dlp): extractor fixes and runtime needs.
- [Parakeet MLX](https://github.com/senstella/parakeet-mlx): candidate Apple
  acceleration.

Compare changes against the same fixtures and hardware.
Track word error rate, diarization error rate,
runtime/media-duration ratio, and peak memory where reference data permits. Include single speakers, rapid changes, overlapping speech, noisy audio, multilingual material, mono/stereo,
silence, and long input.
Do not label a new tool best-in-class based only on its own benchmark.

Update the manifest and lock together.
Check available wheels, FFmpeg ABI, CUDA requirements, language coverage, model
licensing/access, output schema, and whether model downloads remain
reproducible.
Keep the working baseline until an alternative demonstrates a useful
improvement.
Add a backend only when it solves a measured need; keep presets explicit and CLI
overrides stable.
