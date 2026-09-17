import contextlib
import importlib.util
import io
import json
import sys
import types
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

SOURCE = (
    Path(__file__).parents[1] / ".apm/skills/video-transcriber/scripts/transcribe.py"
)
spec = importlib.util.spec_from_file_location("setup_runner", SOURCE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class SetupTests(unittest.TestCase):
    def check_setup(self, token, network_error=None, import_error=None):
        output = io.StringIO()
        hub = types.SimpleNamespace(get_token=lambda: token)
        with (
            patch.dict(sys.modules, {"huggingface_hub": hub}),
            patch.object(runner.shutil, "which", return_value="/ffmpeg"),
            patch.object(runner.importlib, "import_module", side_effect=import_error),
            patch.object(runner, "resolve_device", return_value="cpu"),
            patch.object(runner.urllib.request, "urlopen", side_effect=network_error),
            contextlib.redirect_stdout(output),
        ):
            status = runner.setup()
        return status, json.loads(output.getvalue())

    def test_success_does_not_reveal_token(self):
        status, report = self.check_setup("test-secret")
        self.assertEqual(status, 0)
        self.assertEqual(report["checks"]["model_access"], "ready")
        self.assertNotIn("test-secret", json.dumps(report))

    def test_missing_credentials_and_broken_runtime_report_all_actions(self):
        error = urllib.error.HTTPError("url", 401, "secret detail", {}, None)
        status, report = self.check_setup(None, error, ImportError("secret detail"))
        self.assertEqual(status, 1)
        self.assertEqual(report["checks"]["credentials"], "missing")
        self.assertEqual(report["checks"]["model_access"], "http_401")
        self.assertEqual(len(report["actions"]), 3)
        self.assertNotIn("secret detail", json.dumps(report))

    def test_network_error_is_not_misreported_as_gating(self):
        status, report = self.check_setup(
            "secret", urllib.error.URLError("private proxy")
        )
        self.assertEqual(status, 1)
        self.assertEqual(report["checks"]["model_access"], "network_error")
        self.assertNotIn("private proxy", json.dumps(report))


if __name__ == "__main__":
    unittest.main()
