"""Offline scorer invariants; metric tests run in the data environment."""

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).parents[1] / "tools/benchmark.py"
spec = importlib.util.spec_from_file_location("benchmark", SOURCE)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class BenchmarkTests(unittest.TestCase):
    def test_comparison_keeps_failures_out_of_medians(self):
        common = dict.fromkeys(
            (
                "fixture_manifest_sha256",
                "reference_sha256",
                "audio_sha256",
                "protocol_sha256",
                "cache_label",
            ),
            "same",
        )
        common.update(
            metric_versions={},
            hardware={},
            source_commit="commit",
            runner_sha256="runner",
            benchmark_sha256="scorer",
            quality="fast",
            real_time_factor=1,
            peak_process_tree_rss_bytes=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index, (status, elapsed) in enumerate(
                (("complete", 10), ("complete", 20), ("failed", 100))
            ):
                path = Path(directory) / f"{index}.json"
                benchmark.write(
                    path,
                    dict(
                        common,
                        run_id=str(index),
                        status=status,
                        wall_seconds=elapsed,
                        metrics={"wer": 0.1, "der": 0.2}
                        if status == "complete"
                        else None,
                    ),
                )
                paths.append(path)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                benchmark.compare(paths)
            report = json.loads(output.getvalue())
            self.assertEqual(len(report["runs"]), 3)
            self.assertEqual(report["summaries"][0]["failed_runs"], 1)
            self.assertEqual(report["summaries"][0]["median_wall_seconds"], 15)

    def test_normalization_preserves_fillers_and_handles_unicode(self):
        self.assertEqual(
            benchmark.normalize("  UH, Café! Don’t   42. "), "uh café dont 42"
        )

    def test_reference_turns_clip_to_half_open_excerpt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.rttm"
            path.write_text(
                "SPEAKER sample 1 8 5 <NA> <NA> A <NA> <NA>\n"
                "SPEAKER sample 1 15 2 <NA> <NA> B <NA> <NA>\n"
            )
            self.assertEqual(
                benchmark.reference_turns(path, 10, 15),
                [{"start": 0, "end": 3, "speaker": "A"}],
            )

    def test_incompatible_references_cannot_be_compared(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for i in range(2):
                path = Path(directory) / f"{i}.json"
                path.write_text(json.dumps({"fixture_manifest_sha256": str(i)}))
                paths.append(path)
            with self.assertRaisesRegex(ValueError, "fixture_manifest_sha256"):
                benchmark.compare(paths)


@unittest.skipUnless(
    importlib.util.find_spec("jiwer"), "run with pixi -e data for metric dependencies"
)
class MetricTests(unittest.TestCase):
    def reference(self):
        return {
            "id": "test",
            "duration_seconds": 2,
            "words": [
                {"word": "hello", "start": 0, "end": 0.5, "speaker": "A"},
                {"word": "world", "start": 1, "end": 1.5, "speaker": "B"},
            ],
            "turns": [
                {"start": 0, "end": 1, "speaker": "A"},
                {"start": 1, "end": 2, "speaker": "B"},
            ],
        }

    def test_perfect_words_and_permuted_speakers_score_zero(self):
        ref = self.reference()
        hyp = {
            "segments": [{"text": "Hello, world!", "words": ref["words"]}],
            "speaker_turns": [
                dict(t, speaker={"A": "Y", "B": "X"}[t["speaker"]])
                for t in ref["turns"]
            ],
        }
        result = benchmark.score(ref, hyp)
        self.assertEqual(result["wer"], 0)
        self.assertEqual(result["der"], 0)
        self.assertEqual(result["word_boundary_mae_seconds"], 0)
        self.assertEqual(result["matched_timed_words"], 2)

    def test_missing_overlap_is_penalized(self):
        ref = self.reference()
        ref["turns"] = [{"start": 0, "end": 2, "speaker": s} for s in "AB"]
        result = benchmark.score(
            ref,
            {
                "segments": [{"text": "hello world"}],
                "speaker_turns": [{"start": 0, "end": 2, "speaker": "X"}],
            },
        )
        self.assertAlmostEqual(result["der"], 0.5)
        self.assertEqual(result["speaker_count_absolute_error"], 1)
        self.assertIsNone(result["word_boundary_mae_seconds"])

    def test_deletions_and_silent_hypothesis_are_not_ignored(self):
        result = benchmark.score(
            self.reference(), {"segments": [], "speaker_turns": []}
        )
        self.assertEqual(result["wer"], 1)
        self.assertEqual(result["der"], 1)
        self.assertEqual(result["word_counts"]["deletions"], 2)


if __name__ == "__main__":
    unittest.main()
