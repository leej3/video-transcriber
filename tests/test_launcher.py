import importlib.util
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SKILL = Path(__file__).parents[1] / ".apm/skills/video-transcriber"
spec = importlib.util.spec_from_file_location("launcher", SKILL / "scripts/launch.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class LauncherTests(unittest.TestCase):
    def test_cache_contains_only_runtime_and_changes_with_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "skill"
            for name in launcher.RUNTIME_FILES:
                target = source / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(name)
            (source / ".pixi").mkdir()
            (source / ".pixi/never-copy").write_text("local environment")
            first = launcher.prepare_runtime(source, root / "cache")
            self.assertFalse((first / ".pixi").exists())
            self.assertEqual(first, launcher.prepare_runtime(source, root / "cache"))
            (source / "pixi.lock").write_text("new dependencies")
            self.assertNotEqual(first, launcher.prepare_runtime(source, root / "cache"))

    def test_launcher_preserves_relative_input_and_output(self):
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(launcher.shutil, "which", return_value="/pixi"),
                patch.object(launcher, "cache_root", return_value=Path(directory)),
                patch.object(launcher.subprocess, "call", return_value=0) as call,
            ):
                self.assertEqual(
                    launcher.main(
                        [
                            "movie.mp4",
                            "--output",
                            "result.txt",
                            "--token-file",
                            "private-token",
                        ]
                    ),
                    0,
                )
            command = call.call_args.args[0]
            self.assertIn(str(Path("movie.mp4").resolve()), command)
            self.assertIn(str(Path("result.txt").resolve()), command)
            self.assertIn(str(Path("private-token").resolve()), command)
            self.assertIn("--locked", command)

    def test_copied_skill_runs_help_without_parent_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "standalone"
            shutil.copytree(
                SKILL, target, ignore=shutil.ignore_patterns(".pixi", "__pycache__")
            )
            result = subprocess.run(
                [sys.executable, str(target / "scripts/launch.py"), "--help"],
                cwd=directory,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--quality", result.stdout)


if __name__ == "__main__":
    unittest.main()
