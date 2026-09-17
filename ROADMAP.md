# Direction and next acceptance milestone

Continue as a thin, agent-independent skill plus deterministic CLI, contingent
on measured usefulness for our recordings.
Claude Code can operate or maintain the same CLI; changing the assistant does
not establish transcription accuracy or remove the need for a tested audio
pipeline.

## One pipeline first

Keep WhisperX (faster-whisper, forced alignment, Community-1 diarization).
Measure fast/balanced/accurate on the same clips before tuning defaults.
Apple Silicon currently runs CPU inference; measure wall time and peak memory
before adding an accelerated backend.
Pixi owns software resolution, including FFmpeg and the Apple Silicon PyPI
git-annex wheel; OS drivers, model permissions, and network access remain
external prerequisites.

Offer a single setup task with one checklist for credentials, model conditions,
and environment failures.
Setup checks access; actual inference remains a separate acceptance check.
Never silently remove diarization to bypass gating.

## Feedback from annextube

The
[review of the older published implementation](https://github.com/con/annextube/issues/31#issuecomment-5714772382)
correctly identified missing licensing and real inference evidence.
The local refactor replaces MFCC/K-means fixed-count clustering with
Community-1, but that architectural improvement alone is not quality evidence.

Annextube already retrieves captions and curates them through CaptionCurator.
Our boundary is fresh transcription when captions are absent or inadequate, plus
speaker labels and timed structured output.
Preserve the existing curation path.
Aligning curated text and generating a new transcript are distinct tasks; do not
overwrite the curated text with fresh ASR as an alignment substitute.

Before integration, demonstrate a stable JSON contract, explicit error status,
local-file and URL success, and a repeatable quality/speed comparison.

## Evaluation data and decision

Use the [DataLad registry](https://registry.datalad.org) for discovery and
[ReproTube](https://datasets.datalad.org/repronim/ReproTube/) for recordings
from our domain archived with annextube.
Clone metadata first and retrieve selected files.
Keep source commits, annex provenance, content checksums, and license
observations in the fixture manifest.
Do not bundle third-party media into this MIT repository.

The initial fixture is a speech smoke test.
It has no verified reference text or speaker annotations.
Archived auto-captions are comparison material, not ground truth.
The [reference benchmark](benchmarks/README.md) now starts with a pinned AMI
meeting excerpt carrying manual word and speaker references.
Expand it to single-speaker talks, interviews, technical vocabulary, and noisy
recordings.

Record WER against corrected text, diarization error with declared
overlap/collar conventions, speaker-count errors, timestamp errors, real-time factor, peak RAM, and failures. Separate installation/model
downloads from warm inference.
Include
silence/music to check hallucinations and URL retrieval failures. Keep routine unit tests offline; run real-model acceptance tests explicitly. Package locks do not pin model artifacts: record model revisions/checksums
and replace moving model/VAD references before claiming reproducible inference.

Continue investing if the pipeline provides useful speaker labels and acceptable
transcripts at an agreed latency on the user's hardware.
If it does not improve our caption workflow enough to justify maintenance,
retain the small wrapper and fixtures, and stop expanding it.
Choose that boundary from results, not from whether Codex or Claude happens to
drive the task.
