import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))

from transcribe import (  # noqa: E402
    collect_segments,
    default_output,
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


if __name__ == "__main__":
    unittest.main()
