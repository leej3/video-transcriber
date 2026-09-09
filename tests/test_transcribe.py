import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))

from transcribe import (  # noqa: E402
    collect_segments,
    default_output,
    json_output,
    parse_args,
    seconds_to_timestamp,
    srt_output,
    text_output,
)


class TranscribeFormattingTests(unittest.TestCase):
    def test_seconds_to_timestamp(self):
        self.assertEqual(seconds_to_timestamp(3661.234), "01:01:01.234")
        self.assertEqual(seconds_to_timestamp(3661.234, decimal=","), "01:01:01,234")

    def test_empty_text_segments_are_omitted(self):
        segments = collect_segments(
            [
                SimpleNamespace(start=0, end=1, text=" hello "),
                SimpleNamespace(start=1, end=2, text="  "),
            ]
        )
        self.assertEqual(segments, [{"start": 0.0, "end": 1.0, "text": "hello"}])

    def test_text_output_contains_timestamps(self):
        segments = [{"start": 0.0, "end": 1.5, "text": "Hello."}]
        self.assertEqual(text_output(segments), "[00:00:00.000] Hello.\n")

    def test_text_output_includes_speaker_and_end_time_when_available(self):
        segments = [
            {
                "start": 0.0,
                "end": 1.5,
                "speaker": "SPEAKER_01",
                "text": "Hello.",
            }
        ]
        self.assertEqual(
            text_output(segments),
            "[00:00:00.000–00:00:01.500] SPEAKER_01\nHello.\n",
        )

    def test_srt_output(self):
        segments = [{"start": 0.0, "end": 1.5, "text": "Hello."}]
        self.assertEqual(
            srt_output(segments),
            "1\n00:00:00,000 --> 00:00:01,500\nHello.\n",
        )

    def test_default_output_for_url(self):
        self.assertEqual(
            default_output("https://example.org/meeting.mp4", "text"),
            Path("meeting.txt"),
        )

    def test_diarization_is_on_by_default_and_can_be_disabled(self):
        self.assertFalse(parse_args(["video.mp4"]).no_diarize)
        self.assertTrue(parse_args(["video.mp4", "--no-diarize"]).no_diarize)

    def test_json_output_lists_generic_speakers(self):
        rendered = json_output(
            [
                {
                    "start": 0.0,
                    "end": 1.5,
                    "speaker": "SPEAKER_01",
                    "text": "Hello.",
                }
            ],
            model="small",
            language="en",
        )
        self.assertIn('"speakers": [\n    "SPEAKER_01"\n  ]', rendered)


if __name__ == "__main__":
    unittest.main()
