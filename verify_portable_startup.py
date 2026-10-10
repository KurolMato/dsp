"""Windows 启动迁移验证：真实复现失效 venv 并在中文空格目录重建，跳过联网安装。"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "results/verification_portable_20261010"
REPORT.mkdir(parents=True, exist_ok=True)
(ROOT / ".cache").mkdir(parents=True, exist_ok=True)


def ps(root, mode="run", flag="-CheckOnly", env=None):
    return subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
        str(root / "bootstrap_windows.ps1"), "-Mode", mode, flag], cwd=root, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


class PortableTests(unittest.TestCase):
    def test_01_existing_primary_and_speaker_envs_work(self):
        for mode in ("run", "speakers"):
            cfg = ROOT / (".venv-speakers" if mode == "speakers" else ".venv") / "pyvenv.cfg"
            before = hashlib.sha256(cfg.read_bytes()).hexdigest()
            result = ps(ROOT, mode)
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertEqual(hashlib.sha256(cfg.read_bytes()).hexdigest(), before)

    def test_02_reproduce_screenshot_then_rebuild_in_moved_project(self):
        # 不修改工作环境。临时目录包含中文、空格和 &，模拟另一台电脑/位置。
        parent = Path(tempfile.mkdtemp(prefix="portable_", dir=ROOT / ".cache"))
        project = parent / "复制到其他电脑 空格 & 项目"
        project.mkdir()
        shutil.copyfile(ROOT / "bootstrap_windows.ps1", project / "bootstrap_windows.ps1")
        env = project / ".venv"
        (env / "Scripts").mkdir(parents=True)
        shutil.copyfile(ROOT / ".venv/Scripts/python.exe", env / "Scripts/python.exe")
        (env / "pyvenv.cfg").write_text("home = C:\\Missing_Original_PC\\Python312\ninclude-system-site-packages = false\nversion = 3.12.2\n", encoding="utf-8")
        failing = subprocess.run([str(env / "Scripts/python.exe"), "-c", "print('ok')"], capture_output=True, timeout=15,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.assertNotEqual(failing.returncode, 0)
        self.assertIn(b"No Python", failing.stderr + failing.stdout)
        self.assertEqual(ps(project).returncode, 1)
        repaired = ps(project, flag="-EnsureOnly")
        self.assertEqual(repaired.returncode, 0, repaired.stdout)
        self.assertEqual(len(list(project.glob(".venv.broken_*"))), 1)
        checked = ps(project)
        self.assertEqual(checked.returncode, 0, checked.stdout)
        run = subprocess.run([str(env / "Scripts/python.exe"), "-c", "import sys; print(sys.prefix)"], capture_output=True, timeout=15)
        self.assertEqual(run.returncode, 0)
        (REPORT / "rebuild_output.txt").write_bytes(repaired.stdout + checked.stdout)
        # 保留临时环境用于检查；不做递归删除或触碰真实 .venv。

    def test_03_fresh_speaker_env_is_separate_and_idempotent(self):
        project = Path(tempfile.mkdtemp(prefix="speaker_portable_", dir=ROOT / ".cache"))
        shutil.copyfile(ROOT / "bootstrap_windows.ps1", project / "bootstrap_windows.ps1")
        first = ps(project, mode="speakers", flag="-EnsureOnly")
        self.assertEqual(first.returncode, 0, first.stdout)
        cfg = project / ".venv-speakers/pyvenv.cfg"
        content = cfg.read_bytes()
        self.assertFalse((project / ".venv").exists())
        second = ps(project, mode="speakers", flag="-EnsureOnly")
        self.assertEqual(second.returncode, 0, second.stdout)
        self.assertEqual(cfg.read_bytes(), content)
        self.assertFalse(list(project.glob(".venv-speakers.broken_*")))

    def test_04_missing_base_python_keeps_old_env(self):
        project = Path(tempfile.mkdtemp(prefix="no_python_", dir=ROOT / ".cache"))
        script = (ROOT / "bootstrap_windows.ps1").read_text()
        # 模拟检测不到本机 Python；用于验证保护顺序，不改变系统注册表/PATH。
        start = script.index("function Find-LocalPython {")
        end = script.index("function Assert-WorkspaceChild", start)
        script = script[:start] + "function Find-LocalPython { throw 'No local Python in simulated target computer.' }\n\n" + script[end:]
        (project / "bootstrap_windows.ps1").write_text(script)
        old = project / ".venv"
        old.mkdir()
        (old / "keep.txt").write_text("keep")
        result = ps(project, flag="-EnsureOnly")
        self.assertEqual(result.returncode, 1)
        self.assertTrue((old / "keep.txt").is_file())
        self.assertFalse(list(project.glob(".venv.broken_*")))

    def test_05_launchers_have_no_personal_python_path(self):
        for name in ("run_windows.bat", "setup_music_windows.bat", "setup_speakers_windows.bat", "bootstrap_windows.ps1"):
            text = (ROOT / name).read_text()
            self.assertNotIn("26824", text)
            self.assertNotIn("C:\\Users", text)
            self.assertIn("bootstrap_windows.ps1", text) if name.endswith(".bat") else None

    def launcher_project(self, broken_first=False):
        project = Path(tempfile.mkdtemp(prefix="multi_launcher_", dir=ROOT / ".cache"))
        shutil.copyfile(ROOT / "bootstrap_windows.ps1", project / "bootstrap_windows.ps1")
        lookup = subprocess.run(["powershell.exe", "-NoProfile", "-Command",
            "Get-Command py.exe -CommandType Application -All | ForEach-Object { $_.Source }"],
            capture_output=True, timeout=15)
        self.assertEqual(lookup.returncode, 0, lookup.stderr)
        launcher = next(Path(p.strip()) for p in lookup.stdout.decode().splitlines()
                        if p.strip() and "WindowsApps" not in p)
        paths = [project / "first launcher", project / "second launcher", project / "WindowsApps"]
        for path in paths:
            path.mkdir()
            shutil.copyfile(launcher, path / "py.exe")
        if broken_first:
            # Real venv redirector with a missing base, rather than a mocked failure.
            shutil.copyfile(ROOT / ".venv/Scripts/python.exe", paths[0] / "py.exe")
            (paths[0] / "pyvenv.cfg").write_text("home = C:\\Missing_Launcher_Base\n", encoding="utf-8")
        env = os.environ.copy()
        # Put the WindowsApps alias first to prove it is skipped even when it wins PATH lookup.
        env["PATH"] = os.pathsep.join(map(str, [paths[2], paths[0], paths[1]])) + os.pathsep + env["PATH"]
        count = subprocess.run(["powershell.exe", "-NoProfile", "-Command",
            "@(Get-Command py.exe -CommandType Application -All).Count"],
            env=env, capture_output=True, timeout=15)
        self.assertGreaterEqual(int(count.stdout.strip()), 3)
        return project, paths, env

    def test_06_multiple_launchers_are_not_joined(self):
        project, paths, env = self.launcher_project()
        # Execute the previous script in the same real multi-launcher PATH.
        previous = ROOT / "backups/before_multi_launcher_fix_20261010/bootstrap_windows.ps1"
        if previous.exists():
            shutil.copyfile(previous, project / "bootstrap_windows.ps1")
            failed = ps(project, flag="-EnsureOnly", env=env)
            self.assertNotEqual(failed.returncode, 0, failed.stdout)
            self.assertFalse((project / ".venv").exists())
            (REPORT / "multiple_launchers_before.txt").write_bytes(failed.stdout)
            shutil.copyfile(ROOT / "bootstrap_windows.ps1", project / "bootstrap_windows.ps1")
        repaired = ps(project, flag="-EnsureOnly", env=env)
        self.assertEqual(repaired.returncode, 0, repaired.stdout)
        self.assertIn(str(paths[0] / "py.exe").encode("utf-8"), repaired.stdout)
        self.assertEqual(ps(project).returncode, 0)
        (REPORT / "multiple_launchers_after.txt").write_bytes(repaired.stdout)

    def test_07_broken_launcher_falls_back_to_next(self):
        project, paths, env = self.launcher_project(broken_first=True)
        repaired = ps(project, flag="-EnsureOnly", env=env)
        self.assertEqual(repaired.returncode, 0, repaired.stdout)
        self.assertIn(str(paths[1] / "py.exe").encode("utf-8"), repaired.stdout)
        self.assertEqual(ps(project).returncode, 0)
        (REPORT / "broken_launcher_fallback.txt").write_bytes(repaired.stdout)


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PortableTests))
    (REPORT / "summary.json").write_text(json.dumps({"tests": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "ok": result.wasSuccessful()}, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
