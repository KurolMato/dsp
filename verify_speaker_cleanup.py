"""逐轨后处理回归：噪声门、真实 GTCRN、原音保护、取消、对比与导出。"""
import hashlib
import json
from pathlib import Path
import queue
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
import soundfile as sf

from core.speakers import ROOT, REFERENCE_DIR, SpeakerPostJob
from core.speaker_cleanup import DEFAULTS, cleanup_track, soft_noise_gate, validate_config
from core.gtcrn import GTCRNDenoiser

REPORT = ROOT / "results/verification_postprocess_20261010"
REPORT.mkdir(parents=True, exist_ok=True)


def reference_result():
    result = json.loads((REFERENCE_DIR / "separation_result.json").read_text(encoding="utf-8"))
    result["files"] = {name: str(REFERENCE_DIR / path) for name, path in result["files"].items()}
    return result


class CoreTests(unittest.TestCase):
    def test_01_disabled_and_zero_strength_exact_copy(self):
        y = np.random.default_rng(2).normal(0, .1, 14000).astype(np.float32)
        for config in ({"denoise_enabled": False}, {"denoise_strength": 0}):
            np.testing.assert_array_equal(cleanup_track(y, 16000, config), y)
        np.testing.assert_array_equal(soft_noise_gate(y, 16000, reduction_db=0), y)

    def test_02_denoise_blending_not_peak_normalization(self):
        class HalfModel:
            is_ready = True
            def enhance(self, audio, fs):
                return audio * .5
        y = np.ones(16000, np.float32) * .02
        np.testing.assert_allclose(cleanup_track(y, 16000, {"denoise_strength": .4}, HalfModel()), y * .8)

    def test_03_gate_weak_residual_and_strong_speech(self):
        fs = 16000
        t = np.arange(4 * fs) / fs
        y = (.001 * np.sin(2 * np.pi * 500 * t)).astype(np.float32)
        y[fs:2 * fs] += (.08 * np.sin(2 * np.pi * 220 * t[fs:2 * fs])).astype(np.float32)
        out = soft_noise_gate(y, fs)
        before = np.mean(y[3 * fs:] ** 2)
        after = np.mean(out[3 * fs:] ** 2)
        drop = float(10 * np.log10(before / after))
        retention = float(np.mean(out[fs + 1000:2 * fs - 1000] ** 2) / np.mean(y[fs + 1000:2 * fs - 1000] ** 2))
        self.assertGreater(drop, 17)
        self.assertGreater(retention, .98)
        (REPORT / "gate_measurement.json").write_text(json.dumps({"synthetic_quiet_drop_db": drop,
            "strong_segment_energy_ratio": retention, "threshold_dbfs": -45, "reduction_db": 18}, indent=2), encoding="utf-8")

    def test_04_hold_release_and_no_hard_chatter(self):
        fs = 16000
        y = np.full(2 * fs, .001, np.float32)
        y[fs // 2:fs] = .05
        out = soft_noise_gate(y, fs)
        gains = out / y
        self.assertGreater(float(gains[fs + round(.08 * fs)]), .95)
        self.assertLess(float(gains[-1]), .15)
        self.assertLess(float(np.max(np.abs(np.diff(gains)))), .005)

    def test_05_silence_rates_and_raw_protection(self):
        denoiser = GTCRNDenoiser()
        self.assertTrue(denoiser.is_ready)
        for fs in (8000, 16000, 44100, 48000):
            y = np.random.default_rng(1).normal(0, .015, fs // 4 + 29).astype(np.float32)
            original = y.copy()
            out = cleanup_track(y, fs, {"gate_enabled": True}, denoiser)
            self.assertEqual(out.shape, y.shape)
            self.assertTrue(np.isfinite(out).all())
            np.testing.assert_array_equal(y, original)
        np.testing.assert_array_equal(soft_noise_gate(np.zeros(3, np.float32), 16000), np.zeros(3))

    def test_06_validation_and_missing_model(self):
        for cfg in ({"denoise_strength": 2}, {"gate_threshold_db": float("nan")}, {"gate_reduction_db": -1}, {"gate_enabled": "False"}):
            with self.assertRaises(ValueError):
                validate_config(cfg)
        with self.assertRaises(RuntimeError):
            cleanup_track(np.ones(16000), 16000, DEFAULTS)
        with self.assertRaises(ValueError):
            cleanup_track([], 16000, {"denoise_enabled": False})

    def test_07_actual_worker_raw_preserved_and_reapply_uses_raw(self):
        source = reference_result()
        configs = {name: DEFAULTS for name in ("speaker_1", "speaker_2")}
        result = None
        for run in range(2):
            out = REPORT / ("worker_first" if run == 0 else "worker_reapply")
            out.mkdir(exist_ok=True)
            request = {"task_id": f"post_validation_{run}", "source_result": source if run == 0 else result,
                       "configs": configs, "output_dir": str(out)}
            path = out / "request.json"
            path.write_text(json.dumps(request), encoding="utf-8")
            with (out / "worker.log").open("w", encoding="utf-8") as log:
                subprocess.run([__import__("sys").executable, "-X", "utf8", str(ROOT / "speaker_postprocess_worker.py"), str(path)],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=30,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            result = json.loads((out / "result.json").read_text(encoding="utf-8"))
            for name in ("speaker_1", "speaker_2"):
                raw, fs = sf.read(source["files"][name], dtype="float32")
                copied, _ = sf.read(result["raw_files"][name], dtype="float32")
                np.testing.assert_array_equal(copied, raw)
                if run:
                    first, _ = sf.read(REPORT / "worker_first" / (name + "_enhanced.wav"))
                    second, _ = sf.read(result["files"][name])
                    np.testing.assert_array_equal(first, second)

    def test_08_actual_postprocess_cancel(self):
        source = reference_result()
        events = queue.Queue()
        job = SpeakerPostJob(events.put)
        job.start(source, {name: DEFAULTS for name in ("speaker_1", "speaker_2")})
        event = events.get(timeout=10)
        self.assertEqual(event["type"], "stage")
        job.cancel()
        self.assertTrue(job.finished.wait(10))
        self.assertIsNotNone(job.process.poll())
        self.assertFalse((job.output / "result.json").exists())

    def test_09_music_hashes_unchanged(self):
        hashes = json.loads((ROOT / "backups/before_speaker_postprocess_20261010/music_hashes.json").read_text())
        for path, sha in hashes.items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), sha)


class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tkinter as tk
        from app import DSPApp
        cls.root = tk.Tk()
        cls.root.withdraw()
        cls.app = DSPApp(cls.root)
        cls.panel = cls.app.speaker_panel
        cls.dialogs = []
        cls.patches = [patch("tkinter.messagebox." + name, side_effect=lambda title, message, **kw: cls.dialogs.append((title, message))) for name in ("showerror", "showinfo", "showwarning")]
        for p in cls.patches:
            p.start()

    @classmethod
    def tearDownClass(cls):
        cls.app.on_close()
        for p in cls.patches:
            p.stop()

    def pump(self, condition, timeout=20):
        deadline = time.monotonic() + timeout
        while not condition() and time.monotonic() < deadline:
            self.root.update()
            time.sleep(.03)
        self.assertTrue(condition(), self.panel.status.get())

    def test_10_per_track_settings_real_job(self):
        panel = self.panel
        panel.open_manifest(str(REFERENCE_DIR / "separation_result.json"))
        self.pump(lambda: not panel.busy)
        original = {k: v.copy() for k, v in panel.raw_tracks.items()}
        panel.post_vars["speaker_1"]["denoise_strength"].set(35)
        panel.post_vars["speaker_2"]["denoise_enabled"].set(False)
        panel.apply_postprocess()
        self.assertTrue(panel.busy)
        self.pump(lambda: not panel.busy)
        self.assertIsNotNone(panel.enhanced_tracks)
        self.assertEqual(panel.version.get(), "增强结果")
        np.testing.assert_array_equal(panel.raw_tracks["speaker_1"], original["speaker_1"])
        np.testing.assert_array_equal(panel.enhanced_tracks["speaker_2"], original["speaker_2"])
        self.assertFalse(np.array_equal(panel.enhanced_tracks["speaker_1"], original["speaker_1"]))
        self.assertEqual(panel.result["postprocess"]["tracks"]["speaker_1"]["denoise_strength"], .35)

    def test_11_compare_retains_cursor_and_volume(self):
        panel = self.panel
        panel.player.seek(2)
        panel.audition("speaker_1")
        gains = panel.player.gains.copy()
        panel.version.set("原始分离")
        self.assertEqual(panel.player.position, 32000)
        self.assertEqual(panel.player.gains, gains)
        np.testing.assert_array_equal(panel.player.tracks["speaker_1"], panel.raw_tracks["speaker_1"])
        panel.version.set("增强结果")
        self.assertEqual(panel.player.position, 32000)
        np.testing.assert_array_equal(panel.player.tracks["speaker_1"], panel.enhanced_tracks["speaker_1"])
        with patch.object(panel, "play") as start_play:
            panel.compare_track("speaker_1", True)
            start_play.assert_called_once()

    def test_12_export_versions_and_reload(self):
        panel = self.panel
        for version in ("原始分离", "增强结果"):
            panel.version.set(version)
            path = str(REPORT / ("raw_mix.wav" if version == "原始分离" else "clean_mix.wav"))
            with patch("tkinter.filedialog.asksaveasfilename", return_value=path):
                panel.export_mix()
            meta = json.loads(Path(path + ".json").read_text(encoding="utf-8"))
            self.assertEqual(meta["audition_version"], "raw" if version == "原始分离" else "enhanced")
            saved, _ = sf.read(path, dtype="float32", always_2d=True)
            np.testing.assert_allclose(saved, panel.player.mix() * meta["export_gain"], atol=1e-7)
        with patch("tkinter.filedialog.askdirectory", return_value=str(REPORT)):
            panel.export_tracks()
        out = sorted(REPORT.glob("stems_*/result.json"))[-1]
        result = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(set(result["raw_files"]), {"original", "speaker_1", "speaker_2"})
        self.assertEqual(set(result["enhanced_files"]), set(result["raw_files"]))
        panel.open_manifest(str(out))
        self.pump(lambda: not panel.busy)
        self.assertIsNotNone(panel.enhanced_tracks)
        self.assertFalse(panel.post_vars["speaker_2"]["denoise_enabled"].get())

    def test_13_invalid_and_stale_results_keep_audio(self):
        panel = self.panel
        original = panel.result
        count = len(self.dialogs)
        panel.post_vars["speaker_1"]["denoise_strength"].set(150)
        panel.apply_postprocess()
        self.assertFalse(panel.busy)
        self.assertIs(panel.result, original)
        self.assertGreater(len(self.dialogs), count)
        panel.post_vars["speaker_1"]["denoise_strength"].set(35)
        panel.handle_event({"type": "loaded", "task_id": "old_post_job"})
        self.assertIs(panel.result, original)

    def test_14_small_window_post_controls_scrollable(self):
        panel = self.panel
        self.app.work_tabs.select(panel)
        notebook = panel.post_canvas.master.master
        notebook.select(panel.post_canvas.master)
        self.root.geometry("980x560")
        self.root.deiconify()
        self.root.update()
        panel.post_canvas.yview_moveto(1)
        self.root.update()
        self.assertTrue(panel.post_canvas.cget("scrollregion"))
        self.assertGreater(panel.post_apply_btn.winfo_width(), 100)
        self.assertGreater(panel.post_apply_btn.winfo_rooty(), panel.post_canvas.winfo_rooty())
        self.assertLess(panel.post_apply_btn.winfo_rooty() + panel.post_apply_btn.winfo_height(),
                        panel.post_canvas.winfo_rooty() + panel.post_canvas.winfo_height())
        self.root.withdraw()


if __name__ == "__main__":
    suite = unittest.TestSuite((unittest.defaultTestLoader.loadTestsFromTestCase(CoreTests), unittest.defaultTestLoader.loadTestsFromTestCase(GuiTests)))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    (REPORT / "summary.json").write_text(json.dumps({"tests": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "ok": result.wasSuccessful()}, indent=2), encoding="utf-8")
    raise SystemExit(0 if result.wasSuccessful() else 1)
