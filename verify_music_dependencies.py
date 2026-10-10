"""离线回归：只有基础依赖时启动器必须补装音乐依赖。"""
import contextlib
import importlib.abc
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import venv
from unittest.mock import patch

import music_worker

ROOT = Path(__file__).resolve().parent


class MusicDependenciesTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "win32", "Windows 启动器")
    def test_bootstrap_repairs_base_only_environment(self):
        (ROOT / ".cache").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache", prefix="music_deps_") as tmp:
            project = Path(tmp)
            venv.create(project / ".venv", with_pip=False)
            script = (ROOT / "bootstrap_windows.ps1").read_text(encoding="utf-8-sig")
            start = script.index("function Invoke-Python(")
            end = script.index("\ntry {", start)
            # Only replace mutations and GUI launch; dependency probing runs in
            # a real isolated Python environment without downloads or user packages.
            script = script[:start] + '''function Invoke-Python([string]$Executable, [string[]]$CommandArgs) {
                Write-Host ('TEST_COMMAND ' + ($CommandArgs -join '|'))
            }
            ''' + script[end:]
            (project / "bootstrap_windows.ps1").write_text(script, encoding="utf-8-sig")
            modules = project / "stub_modules"
            modules.mkdir()
            for name in ("numpy", "scipy", "matplotlib", "sounddevice", "soundfile", "onnxruntime"):
                (modules / f"{name}.py").write_text("")
            env = {**os.environ, "PYTHONPATH": str(modules)}

            def launch():
                result = subprocess.run([
                    "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
                    "-File", str(project / "bootstrap_windows.ps1"), "-Mode", "run",
                ], cwd=project, env=env, capture_output=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return result.stdout.decode("utf-8", errors="replace")

            missing = launch()
            self.assertIn("pip|install|torch>=2.7.1", missing)
            self.assertIn("requirements-music.txt", missing)
            self.assertIn("Music dependencies OK", missing)
            for name in ("torch", "av"):
                (modules / f"{name}.py").write_text("")
            (modules / "pymss.py").write_text("MSSeparator = download_model = resolve_model = None\n")
            ready = launch()
            self.assertNotIn("pip|install", ready)
            self.assertIn("app.py", ready)
            (modules / "pymss.py").write_text("raise ImportError('broken music package')\n")
            self.assertIn("pip|install", launch())

    def test_missing_torch_reports_actionable_error(self):
        class BlockTorch(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "torch":
                    raise ModuleNotFoundError("No module named 'torch'", name="torch")

        with tempfile.TemporaryDirectory() as tmp:
            request = Path(tmp) / "request.json"
            request.write_text("{}")
            output = io.StringIO()
            with patch.dict(sys.modules):
                sys.modules.pop("torch", None)
                with patch.object(sys, "meta_path", [BlockTorch(), *sys.meta_path]), \
                        patch.object(sys, "argv", ["music_worker.py", str(request)]), \
                        contextlib.redirect_stdout(output):
                    self.assertEqual(music_worker.main(), 1)
            event = json.loads(output.getvalue().removeprefix("DSP_EVENT "))
            self.assertEqual(event["type"], "error")
            self.assertIn("setup_music_windows.bat", event["message"])
            self.assertIn(sys.executable, event["message"])
            self.assertIn("No module named 'torch'", event["message"])
            self.assertIn("ModuleNotFoundError", event["detail"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
