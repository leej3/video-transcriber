"""Behavioral orchestration checks; no model downloads required."""

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

SCRIPT = (
    Path(__file__).parents[1] / ".apm/skills/video-transcriber/scripts/transcribe.py"
)
spec = importlib.util.spec_from_file_location("pipeline", SCRIPT)
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class PipelineTests(unittest.TestCase):
    def test_speaker_count_estimated_and_one_allowed(self):
        self.assertIsNone(pipeline.parse_args(["video.mp4"]).speakers)
        self.assertEqual(
            pipeline.parse_args(["video.mp4", "--speakers", "1"]).speakers, 1
        )
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            pipeline.parse_args(
                ["video.mp4", "--min-speakers", "3", "--max-speakers", "2"]
            )

    def test_auto_device_requires_both_engines(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
        ct = SimpleNamespace(get_cuda_device_count=lambda: 0)
        with patch.dict(sys.modules, torch=torch, ctranslate2=ct):
            self.assertEqual(pipeline.resolve_device("auto"), "cpu")
            with self.assertRaises(RuntimeError):
                pipeline.resolve_device("cuda")
            ct.get_cuda_device_count = lambda: 1
            self.assertEqual(pipeline.resolve_device("auto"), "cuda")

    def test_turns_follow_words_and_leave_unassigned_unknown(self):
        segments = [
            {
                "start": 0,
                "end": 3,
                "text": "Hi there yes",
                "speaker": "A",
                "words": [
                    {"start": 0, "end": 1, "word": "Hi", "speaker": "A"},
                    {"start": 1, "end": 2, "word": "there", "speaker": "B"},
                    {"start": 2, "end": 3, "word": "yes"},
                ],
            }
        ]
        turns = pipeline.speaker_turns(segments)
        self.assertEqual([s["speaker"] for s in turns], ["A", "B", "UNKNOWN"])
        self.assertEqual([s["text"] for s in turns], ["Hi", "there", "yes"])

    def test_missing_input_and_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out.txt"
            output.write_text("keep")
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(
                    pipeline.main(["missing.mp4", "--output", str(output)]), 2
                )
            self.assertEqual(output.read_text(), "keep")
            with self.assertRaises(FileNotFoundError):
                with pipeline.media_path(str(Path(directory) / "absent.mp4")):
                    self.fail("unreachable")

    def fake_runtime(
        self, *, empty=False, alignment_error=False, diarization_error=False
    ):
        wx = ModuleType("whisperx")
        segment = {"start": 0.0, "end": 1.0, "text": "Hello"}
        result = {"segments": [] if empty else [segment], "language": "en"}
        wx.load_audio = MagicMock(return_value=[0] * 16000)
        wx.load_model = MagicMock(
            return_value=SimpleNamespace(transcribe=MagicMock(return_value=result))
        )
        wx.load_align_model = MagicMock(return_value=(object(), {}))
        wx.align = MagicMock(return_value={"segments": [segment]})
        if alignment_error:
            wx.align.side_effect = RuntimeError("alignment unavailable")
        diar = ModuleType("whisperx.diarize")
        diarizer = MagicMock()
        diarizer.return_value.__getitem__.return_value.to_dict.return_value = [
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}
        ]
        if diarization_error:
            diarizer.side_effect = RuntimeError("diarization failed")
        diar.DiarizationPipeline = MagicMock(return_value=diarizer)
        diar.assign_word_speakers = lambda turns, result: result
        torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False), device=lambda s: s
        )
        ct = SimpleNamespace(get_cuda_device_count=lambda: 0)
        hf = SimpleNamespace(get_token=lambda: "secret-not-for-reports")
        return {
            "whisperx": wx,
            "whisperx.asr": wx,
            "whisperx.diarize": diar,
            "torch": torch,
            "ctranslate2": ct,
            "huggingface_hub": hf,
        }

    def execute(self, directory, **runtime_options):
        source = Path(directory) / "input.wav"
        source.touch()
        output = Path(directory) / "out.txt"
        modules = self.fake_runtime(**runtime_options)
        with (
            patch.dict(sys.modules, modules),
            patch.object(pipeline.shutil, "which", return_value="/ffmpeg"),
        ):
            with contextlib.redirect_stderr(io.StringIO()):
                status = pipeline.main([str(source), "--output", str(output)])
        report = json.loads(Path(str(output) + ".run.json").read_text())
        self.assertNotIn("secret-not-for-reports", json.dumps(report))
        self.assertNotIn(str(source), json.dumps(report))
        return status, output, report, modules

    def test_complete_pipeline_preserves_structured_turns(self):
        with tempfile.TemporaryDirectory() as directory:
            status, output, report, _ = self.execute(directory)
            self.assertEqual(status, 0)
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["diarization"], "complete")
            data = json.loads(Path(str(output) + ".json").read_text())
            self.assertEqual(data["speaker_turns"][0]["speaker"], "SPEAKER_00")
            self.assertIn("Hello", output.read_text())

    def test_alignment_failure_preserves_transcript(self):
        with tempfile.TemporaryDirectory() as directory:
            status, output, report, _ = self.execute(directory, alignment_error=True)
            self.assertEqual(status, 1)
            self.assertEqual(report["status"], "partial")
            self.assertEqual(report["stage"], "alignment")
            self.assertIn("Hello", output.read_text())
            self.assertEqual(
                json.loads(Path(str(output) + ".json").read_text())["language"], "en"
            )

    def test_diarization_failure_preserves_transcript(self):
        with tempfile.TemporaryDirectory() as directory:
            status, output, report, _ = self.execute(directory, diarization_error=True)
            self.assertEqual(status, 1)
            self.assertEqual(report["stage"], "diarization")
            self.assertEqual(report["diarization"], "pending")
            self.assertIn("Hello", output.read_text())

    def test_empty_asr_completes_without_alignment_or_clustering(self):
        with tempfile.TemporaryDirectory() as directory:
            status, output, report, modules = self.execute(directory, empty=True)
            self.assertEqual(status, 0)
            self.assertEqual(report["diarization"], "no_speech")
            modules["whisperx"].align.assert_not_called()
            self.assertEqual(output.read_text().strip(), "")

    def test_diarization_access_failure_happens_before_asr(self):
        with tempfile.TemporaryDirectory() as directory:
            modules = self.fake_runtime()
            modules["whisperx.diarize"].DiarizationPipeline.side_effect = RuntimeError(
                "access denied"
            )
            output = Path(directory) / "out.txt"
            with (
                patch.dict(sys.modules, modules),
                patch.object(pipeline.shutil, "which", return_value="/ffmpeg"),
            ):
                with contextlib.redirect_stderr(io.StringIO()):
                    source = Path(directory) / "input.wav"
                    source.touch()
                    code = pipeline.main([str(source), "--output", str(output)])
            self.assertEqual(code, 1)
            modules["whisperx"].load_model.assert_not_called()
            self.assertFalse(output.exists())
            self.assertEqual(
                json.loads(Path(str(output) + ".run.json").read_text())["status"],
                "failed",
            )

    def test_playlist_is_rejected_before_media_download(self):
        downloader = MagicMock()
        downloader.extract_info.return_value = {"_type": "playlist", "entries": []}

        def factory(options):
            return contextlib.nullcontext(downloader)

        with patch.dict(sys.modules, yt_dlp=SimpleNamespace(YoutubeDL=factory)):
            with self.assertRaises(RuntimeError):
                with pipeline.media_path("https://example.org/playlist"):
                    self.fail("playlist should not be accepted")
        downloader.process_ie_result.assert_not_called()
        downloader.extract_info.assert_called_once_with(
            "https://example.org/playlist", download=False
        )

    def test_url_download_uses_extractor_and_cleans_up(self):
        downloader = MagicMock()
        paths = []

        def factory(options):
            path = Path(options["outtmpl"].replace("%(ext)s", "mp4"))
            path.write_bytes(b"media")
            paths.append(path)
            downloader.extract_info.return_value = {"id": "video"}
            downloader.prepare_filename.return_value = str(path)
            return contextlib.nullcontext(downloader)

        with patch.dict(sys.modules, yt_dlp=SimpleNamespace(YoutubeDL=factory)):
            with pipeline.media_path("https://example.org/watch?v=123") as path:
                self.assertEqual(path.read_bytes(), b"media")
        self.assertFalse(paths[0].exists())


if __name__ == "__main__":
    unittest.main()
