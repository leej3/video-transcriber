# Video Transcriber: project review

Reviewed 2026-09-10 against the goal: low-friction URL/file transcription,
adjustable speed versus quality, and useful speaker diarization using open
tools.

**Recommendation:** retain a tiny deterministic runner, replace the homegrown
diarization, and package the workflow as a self-contained skill if an agent is
the main interface.
The project is already small; its largest opportunity is better upstream
composition, rather than fewer lines of code.

## What works, and what needs attention

The single-file CLI, Pixi lock, local inference, model caching, VAD, and text,
SRT, and JSON exports are a sensible foundation.
Eight existing tests pass (`pixi run test`), but cover formatting and argument
defaults rather than transcription accuracy or speaker attribution.
No model benchmark was run.

| Priority | Finding in `tools/transcribe.py` | Recommended change |
| --- | --- | --- |
| High | `diarize_segments` clusters MFCC summaries with KMeans, assigning one speaker to each ASR segment. A segment can contain multiple speakers. | Replace this heuristic with pretrained diarization and reconcile speaker turns with word timestamps. |
| High | `--speakers` defaults to two and rejects one; fewer than two ASR segments makes the default pipeline fail, including silence or a short utterance. | Estimate speaker count by default; allow one, an exact count, or bounds. Handle empty/short audio explicitly. |
| High | `media_path` downloads a URL verbatim. A video webpage becomes HTML, not playable media. | Use a maintained media extractor for supported sites; retain local-file input and handle download errors clearly. |
| High | `--device auto` explicitly selects CPU; beam size is fixed at five. | Detect supported hardware and provide tested presets with explicit overrides. |
| Medium | `_decode_mono_audio` assumes array axes imply channels. Packed stereo can remain interleaved; averaging integer arrays also changes dtype before normalization. | Delegate decoding/resampling to an established audio loader instead of maintaining this path. |
| Medium | ASR runs before diarization and no transcript is saved if diarization fails. | Check dependencies/model access first; preserve completed ASR with an explicit partial-result status. |

## Recommended upstream stack

- **Input:** [yt-dlp](https://github.com/yt-dlp/yt-dlp) plus FFmpeg for
  supported web sources and audio extraction.
  Prefer audio-only downloads.
  Reusing existing captions can be a fast option, but captions do not
  necessarily provide speaker labels and must not silently satisfy a diarization
  request.
- **Default pipeline:** [WhisperX](https://github.com/m-bain/whisperX) already
  integrates faster-whisper, word alignment, and pyannote diarization.
  Wrapping this is the shortest credible replacement for the custom speaker
  code.
  Alignment adds dependencies and is language-dependent; overlapping speech
  remains difficult.
- **Diarization:**
  [pyannote Community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
  is an open-weight local option with automatic speaker counting and an
  exclusive diarization output useful for transcript reconciliation.
  Initial access requires accepting model conditions and a Hugging Face token;
  cached local execution can run offline.
  Make that a one-time setup step, not a surprise after ASR.
- **Hardware:** retain
  [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for
  CPU/CUDA. On Apple Silicon, benchmark [parakeet-mlx](https://github.com/senstella/parakeet-mlx)
  as an optional ASR backend; it does not replace diarization, and model
  language coverage must match the input.
  Avoid building a large backend framework before measuring.

These are strong integration candidates, not a demonstrated universal “best.”
Choose defaults using representative recordings on the target hardware.

## Make the common path simple

Offer one command with `--quality fast|balanced|accurate`, defaulting to
`balanced`, and diarization enabled in every preset.
Starting hypotheses are Whisper `small` for fast, `turbo` for balanced, and
`large-v3` for accurate; validate those choices rather than promising a
monotonic quality ranking.
Presets should also set
decoding/batching parameters, report the resolved model and device, and retain model/language/speaker
overrides.

Add one-time setup/diagnostics, stage progress, persistent model caches, and
resumable intermediate results.
Produce a readable transcript plus structured JSON with word times, speaker
turns, actual settings, versions, and stage timings.
Keep speaker labels anonymous unless the user supplies a mapping.

Before calling this reliable, test local files and webpage URLs, mono/stereo,
silence, a single speaker, rapid turns, overlap, and a long recording.
Measure word error rate, diarization error rate, total runtime relative to media
duration, and peak memory; distinguish first-run downloads from warm-cache
performance.

## Could it become a skill?

**Yes: a skill with bundled scripts and declared dependencies.** Instructions
can select presets, resolve inputs, invoke the runner, and return artifacts.
Executable code should own downloads, inference,
timestamp/speaker merging, errors, and serialization. A prose-only skill would leave these requirements to repeated agent improvisation and would not remove model/runtime
setup.

For this project, keep the canonical skill under `.apm/skills/` and deploy it
through the project's APM manifest and lock.
Include all scripts and operating references within the skill; retain a CLI
entry point for use without an agent.

Workshop discovery searched memory, all three registered local sources, ASM,
GitHub skill search, and Vercel.
Memory/local/GitHub returned no matches for the query.
Remote status was fetched without updating checkouts: scientific skills was one
upstream commit behind, CON skills had a newer remote revision than its pin, and
NiPreps matched.
Relevant consultation candidates were:

- [Interview transcription](https://github.com/jamditis/claude-skills-journalism):
  useful interview/transcript workflow guidance, but broader than this
  executable transcription requirement; potential adaptation reference.
- [Azure transcription skill](https://github.com/sickn33/antigravity-awesome-skills/tree/main/skills/azure-ai-transcription-py):
  catalog describes timestamps and diarization, but a hosted Azure SDK workflow
  does not meet the open/local objective.
- **New project-owned skill:** recommended because the required surface is small
  and the existing CLI/export code can be reused around upstream inference.

Discovery is not an implementation audit of those candidates.
No skill was adopted, installed, or created.
The next decision is whether to create this project-owned skill, adapt an
existing candidate, or keep the improved CLI alone.
