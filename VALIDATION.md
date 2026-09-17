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

- 32 unique standard-library tests pass across the default and data
  environments, including pipeline
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

## Reference-based AMI pilot

The fixed `ami-es2004a-0000-0060-v1` fixture uses manual transcripts and the
published AMI `only_words` RTTM on a 60-second headset-mix excerpt.
The [protocol](benchmarks/README.md) defines normalization, overlap handling,
zero collar, timing diagnostics, and provenance requirements.
Full source media, derived references, transcripts, and logs are annexed in the
separate `.datasets/benchmarks` DataLad dataset.

Two warm-cache `fast` runs on Apple M4 at source commit `8f0a0b3` completed:

| Measure | Run 1 | Run 2 |
| --- | ---: | ---: |
| Wall time, seconds | 79.799 | 79.994 |
| Real-time factor | 1.330 | 1.333 |
| Sampled process-tree RSS, GiB | 3.59 | 3.95 |
| WER | 88.71% | 88.71% |
| DER, zero collar, overlap included | 47.63% | 47.63% |
| Reference / detected speakers | 3 / 2 | 3 / 2 |

Both runs omitted 55 of 62 normalized reference words.
Their seven matched words had a boundary MAE of 11.866 seconds.
This is a poor result, preserved as an actionable baseline.
Investigate ASR/VAD omissions and alignment behavior before drawing conclusions
about the model or optimizing speed.
A 60-second excerpt and two repetitions do not establish corpus-wide accuracy,
statistical significance, or generalization to unseen data.

The [selected small run summaries](benchmarks/results/) retain hashes, versions,
settings, metrics, and timing evidence without copying transcripts or media.
Compare them with
`pixi run -e data benchmark compare benchmarks/results/*.json`.
An initial uncontrolled-cache run took 151 seconds and stays in the DataLad
history; it is not mixed into the warm-cache speed comparison.
