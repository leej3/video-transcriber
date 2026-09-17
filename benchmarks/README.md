# Reference benchmarks

`ami-mini.json` pins a 60-second excerpt (0–60 s) of AMI ES2004a, a meeting in
the standard test partition.
It has manual transcripts, timed words, multiple speakers, and overlap.
This is a small regression fixture, not an AMI leaderboard score or evidence
that the model has never encountered the corpus.

Sources:

- [AMI manual annotations 1.6.2 and audio](https://groups.inf.ed.ac.uk/ami/download/),
  CC BY 4.0; credit the AMI Consortium and Carletta et al. (2006).
- [AMI only_words diarization setup](https://github.com/pyannote/AMI-diarization-setup),
  pinned to commit `67c2d539286e89f68952d5dcf83912bd9f01dfae`.
  The setup repository declares Apache-2.0; underlying AMI annotations retain
  their corpus terms.
  Credit Landini et al., VBx (2022), when using this setup.

## Store data separately

The source repository stores code, checksums, protocol documentation, and
selected small result summaries only.
Raw downloads, derived
audio/reference text, transcripts, logs, and full run records live in a separate DataLad dataset under ignored `.datasets/benchmarks/`,
or a user-selected external location.
The project MIT license does not relicense these files.

```console
pixi run -e data datalad create .datasets/benchmarks
pixi run -e data benchmark prepare
pixi run -e data datalad -C .datasets/benchmarks save -m 'chore(data): save pinned AMI fixture'
```

Run `create` only once.
Preparation downloads three sources (roughly 54 MB total) through git-annex,
verifies SHA256, cuts the WAV without re-encoding, and annexes its derived
fixture.
Existing content is verified and reused; missing annex content is retrieved
using DataLad.
Use a new fixture ID for a changed excerpt.
The source URLs are registered as annex remotes for retrieval.

To move the dataset between machines, publish its Git metadata and annexed
content to a configured Forgejo/annexjo sibling.
Raw upstream files have public retrieval URLs; derived fixtures are reproducible
with `prepare`, while past run outputs need a storage sibling for independent
recovery.
This project does not assume a Forgejo host, repository, or credentials.
Do not force-drop the only copy of generated results.
No media belongs in the source repository or PR.

## Run and compare

```console
pixi run -e data benchmark run --quality fast --cache-label uncontrolled --token-file /private/token
pixi run -e data benchmark run --quality fast --repeat 3 --cache-label warm --token-file /private/token
pixi run -e data benchmark compare .datasets/benchmarks/runs/RUN1/result.json .datasets/benchmarks/runs/RUN2/result.json
pixi run -e data datalad -C .datasets/benchmarks save -m 'test(benchmark): record reference evaluation'
```

Run `pixi run setup` first.
`--token-file` is optional when `HF_TOKEN` or a cached login is available.
Use `balanced` or `accurate` to compare quality presets with identical inputs;
diarization stays enabled and no reference speaker count is passed to inference.
Each repetition starts a fresh process and preserves an independent result.
No overwrite of earlier runs occurs.

The cache label is a declared experimental condition, not an automatic cache
purge or guarantee.
Prime each selected model before labeling runs `warm`; reserve `cold` for a
controlled empty cache.
Use `uncontrolled` otherwise.
Avoid other heavy workloads and record at least three warm repetitions before
claiming a speed change.
The first run may include Pixi installation and model downloads.
Full wall time includes launcher/startup; pipeline stage timings are reported
separately.
Scoring, hashing, and annex writes are outside timed inference.

`compare` refuses different
fixture/reference hashes, scoring protocols, metric versions, hardware, or cache labels. It preserves failed runs and reports per-revision/preset
medians for successful repetitions.
A lower number is better for WER, CER, DER, latency, RTF, and memory; compare
quality and speed together.
An observed difference on one short clip is not a significant improvement.

## Fixed scoring protocol v1

- WER and CER use jiwer.
  Apply Unicode NFKC and lowercase, delete Unicode punctuation, and collapse
  whitespace.
  Retain fillers; do not expand numbers or apply glossary corrections.
  Exclude AMI punctuation and non-word events.
- Reference words are ordered by start time, speaker, then end time.
  Include words whose midpoint falls inside the half-open excerpt.
  WER uses full ASR segment text, including words that lack alignment.
  Overlapping speakers have no unique word order, so this chronological WER is
  not cpWER or tcpWER.
- DER uses pyannote.metrics with optimal speaker permutation, zero collar,
  overlap included, and a UEM covering the complete excerpt including silence.
  References use the published `only_words` RTTM.
  Preserve missed speech, false alarm, confusion, and reference speaker-time
  denominators.
- Report absolute speaker-count error.
  Labels are anonymous and never compared literally between the reference and
  hypothesis.
- Timing
  MAE/p95 uses boundaries of correctly matched one-token words that have timestamps. Report the matched-word count; it is a conditional diagnostic, not an accuracy measure for deleted/substituted
  words.
  Corpus word times are alignment references, not sample-exact human boundary
  judgments.
- Sample summed process-tree RSS every 100 ms.
  This may double-count shared pages and miss short peaks; it is not OS-reported
  exact maximum memory.

Each run records source commit and dirty state,
CLI/scorer and lock hashes, fixture/reference hashes, package versions,
hardware, cache condition, stage and total timings, process exit status, and
model-cache inventory before/after.
The cache inventory records available public model refs and selected artifact
hashes, not proof of which exact bytes every library loaded.
Model revisions are not yet enforced by the CLI, so cache changes need
investigation.

Offline verification:

```console
pixi run test
pixi run -e data python -B -m unittest discover -s tests -p test_benchmark.py
```

The scorer tests cover perfect output, speaker-label permutation, overlap
misses, empty hypotheses, normalization, excerpt clipping, and incompatible
comparisons.
Keep automatic CI free of model downloads and credentials.
Explicit benchmark runs provide the real-model evidence.
