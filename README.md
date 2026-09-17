# Video Transcriber

A project-owned agent skill with a deterministic CLI for local audio/video and
supported video URLs.
It wraps yt-dlp, WhisperX, and pyannote Community-1 for transcription, word
alignment, and speaker diarization.
Inference runs locally.

## Run

Install [Pixi](https://pixi.sh/), then:

```console
pixi run setup
pixi run transcribe -- /absolute/video.mp4 --output transcripts/video.txt
pixi run transcribe -- 'https://example.org/video.mp4' --quality fast --output transcripts/remote.txt
```

First use installs the locked runtime and downloads models.
Diarization requires accepting the
[Community-1 model conditions](https://huggingface.co/pyannote/speaker-diarization-community-1)
and `HF_TOKEN` or an existing Hugging Face login.
Keep credentials outside Git.
Setup checks imports, FFmpeg, hardware, credentials, and Community-1 access,
then prints one list of required actions.
Use `pixi run setup -- --token-file /private/path/token` for file-based
credentials; pass the same option when transcribing.
It does not download weights or guarantee inference success.
`pixi run doctor` remains an offline runtime check.

`--quality fast|balanced|accurate` selects small/turbo/large-v3; balanced is the
default.
All include alignment and diarization.
These presets are starting points, not measured quality guarantees.
Override with `--model`, `--language`, `--device`, `--compute-type`, or
`--batch-size`.
Speaker count is estimated; use `--speakers N` (including one) or
`--min-speakers`/`--max-speakers` if known.
`--no-diarize` is an explicit opt-out.

Use `--format text|srt|json` and a matching output extension.
Text/SRT outputs include a structured JSON sidecar and a `.run.json` report.
Failures after ASR preserve partial results and exit nonzero.
Existing outputs require `--overwrite`.
Automatic resume is not yet implemented.

## Skill and runtime ownership

The canonical skill, scripts, references, Pixi manifest, and lock live together
under [.apm/skills/video-transcriber](.apm/skills/video-transcriber/SKILL.md).
The root CLI delegates to that implementation.
Root Pixi owns lightweight repository tests; the skill's separate Pixi
environment owns inference.

APM deploys the project-owned skill from `apm.yml` to `.agents/skills/`:

```console
apm install --target codex --no-policy --frozen
apm audit --ci --no-policy
```

`--no-policy` explicitly skips organization-policy discovery for this personal
project.
Edit canonical files, then regenerate deployment with `apm install`.
Do not edit generated copies.
The deployed skill works without this repository, APM, or Workshop once its
directory is present and Pixi is available.

The runtime targets Apple Silicon macOS, Linux x86-64, and Windows x86-64.
Pixi supplies FFmpeg and Python packages, but cannot supply an NVIDIA driver or
missing upstream wheels.
Modern Torch excludes Intel macOS from this runtime.
Apple Silicon currently uses CPU; Metal/MLX acceleration is a future candidate.
Cross-platform lock resolution does not imply every platform has been tested.

## Development

```console
pixi run test
pixi run format
```

The skill's
[maintenance procedure](.apm/skills/video-transcriber/references/maintenance.md)
connects real usage failures, CLI regression tests, agent guidance, upstream
reviews, and environment-specific evidence.
There is no automatic self-update.
Run reports omit source locations, credentials, and transcript content.

[REVIEW.md](REVIEW.md) records the initial assessment before this refactor.

[VALIDATION.md](VALIDATION.md) records what has actually been tested and the
remaining acceptance checks.

## Test data

The separate `data` environment provides DataLad and git-annex.
Apple Silicon uses the PyPI git-annex wheel and requires macOS 14 or later;
Linux uses the pinned conda-forge package.
No Homebrew installation is required.

```console
pixi run -e data datalad clone https://datasets.datalad.org/repronim/ReproTube .datasets/ReproTube
pixi run -e data datalad -C .datasets/ReproTube get -n DataLad
pixi run -e data datalad -C .datasets/ReproTube/DataLad get videos/2021/01/2021-01-04_What-is-DataLad/video.mkv
```

Use `-C` to enter the ignored dataset explicitly, then retrieve selected files.
The [fixture manifest](benchmarks/fixtures.json) records revisions and
checksums.
Fetched media retains its upstream terms and stays outside Git.
See [ROADMAP.md](ROADMAP.md) for the evaluation plan and annextube integration
scope.

## Licensing

Project code and documentation are MIT licensed.
`LICENSE` and `LICENSES/MIT.txt` contain the license; `REUSE.toml` records
copyright and SPDX annotations.
Run `pixi run license-check` to validate REUSE compliance.
The portable transcription skill includes its own license copy.
External skills, model weights, dependencies, and downloaded media retain their
own licenses; they are not relicensed by this project.

## Reference benchmarks

[The benchmark protocol](benchmarks/README.md) uses a pinned AMI meeting excerpt
with manual word transcripts and published diarization references.
Run `pixi run -e data benchmark prepare`, then `benchmark run` and
`benchmark compare` as documented there.
The separate DataLad dataset holds audio, references, logs, and full results;
only small selected summaries belong in this repository.
