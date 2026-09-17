# Validation evidence

2026-09-17, Apple M4, 32 GiB RAM, macOS, Pixi 0.76.2.

## Real local-file run

DataLad successfully cloned ReproTube, installed its DataLad subdataset without
bulk media, and retrieved the 40 MB “What is DataLad?”
recording from the archive.
[fixtures.json](benchmarks/fixtures.json) records both dataset revisions and the
content SHA256.
Third-party media remains ignored and is not relicensed MIT.

The complete local pipeline succeeded with `fast`, English, small/int8, CPU,
batch size 4, and automatic speaker count.
The 227.091-second input produced 29 aligned segments, 35 diarization turns, and
one anonymous speaker label.

| Stage | Seconds |
| --- | ---: |
| Preflight/model loading | 6.181 |
| Input decoding | 0.503 |
| Transcription | 17.131 |
| Alignment | 14.502 |
| Diarization | 296.260 |
| Total | 334.783 |

This is one smoke run, including model loading and some downloads, not a
warm-cache benchmark.
Total real-time factor was approximately 1.47.
Peak memory was not measured.
The sanitized [run evidence](benchmarks/smoke-apple-m4.json) records runtime
versions/settings.
Text and JSON outputs remain locally under `transcripts/datalad-smoke.txt`.

No verified reference transcript or speaker annotation exists for this fixture.
One detected speaker is an observation, not a diarization accuracy score.
Spot inspection found “DataLad” rendered as “data-led” and “DataNet”; this is
concrete motivation for domain-vocabulary evaluation and downstream curation.

## Setup and failures addressed

- Full runtime imports and online Community-1 access now pass through
  `pixi run setup`.
  The token first authenticated but received 403 GatedRepo; after the user
  enabled model access, the config probe and weight loading succeeded.
  Tokens are absent from reports and Git.
- Setup collects runtime, credential, and model-access actions into one
  checklist.
  Its success does not guarantee weight downloads, inference, or quality.
- The first transcription attempt aborted with duplicate `libomp` initialization
  on Apple Silicon.
  Initializing WhisperX ASR before Torch, matching setup and doctor, fixed the
  real retry.
  No duplicate-OpenMP suppression was enabled.
- An earlier installation tried to build PyAV 18 from source and failed without
  pkg-config.
  The runtime now resolves binary conda-forge PyAV 15.1.0 and FFmpeg 7.
- DataLad 1.6.2 and PyPI git-annex 10.20260601 run inside the project `data`
  environment on Apple Silicon; no Homebrew dependency was needed.

## Automated checks and remaining limits

- 25 standard-library tests pass, including pipeline
  failure/partial-output behavior, speaker formatting, URL/playlist handling,
  portable launcher paths, token-file path handling, and aggregated setup
  actions without secret leakage.
  Inference is mocked in unit tests; the real smoke run above is separate.
- The skill frontmatter validator and APM deployment integrity/drift audit pass.
  CON REUSE and K-Dense DataLad are pinned project dependencies.
- REUSE lint passes.
  Snapper formats project documentation.
- Speech locks resolve for Apple Silicon, Linux x86-64, and Windows x86-64; only
  Apple Silicon execution has been tested.
  Data tooling targets macOS 14+ Apple Silicon and Linux, with only the former
  exercised.
- Real URL acquisition,
  multiple/overlapping speakers, silence, multilingual accuracy, warm-cache timings, and balanced/accurate
  comparisons remain to test.
- Model artifacts and the upstream Silero VAD reference are not fully revision
  pinned.
  Dependency locks alone do not establish reproducible inference.

See [ROADMAP.md](ROADMAP.md) for the bounded evaluation and integration plan.
