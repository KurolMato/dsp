"""自动陷波合并回归：真实滤波、模式一致性、后台结果、导出与来源对比。"""
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
from scipy import signal
from core import dsp, audio_io

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "results/verification_auto_notch_20261010"
REPORT.mkdir(parents=True, exist_ok=True)
SOURCE = Path("D:/Tim/data/file/dsp_py/core/dsp.py")


def amplitude(y, fs, frequency, start=.5, end=3.5):
    a, b = round(start * fs), min(len(y), round(end * fs))
    t = np.arange(a, b) / fs
    return 2 * abs(np.mean(y[a:b] * np.exp(-2j * np.pi * frequency * t)))


class AlgorithmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clean, cls.fs = audio_io.load_audio(str(ROOT / "samples/demo_speech.wav"))
        cls.clean = audio_io.to_mono(cls.clean)
        cls.t = np.arange(len(cls.clean)) / cls.fs

    def tone(self, frequency):
        return (self.clean + .15 * np.sin(2 * np.pi * frequency * self.t)).astype(np.float32)

    def test_01_skip_clean_white_and_clicks(self):
        rng = np.random.default_rng(7)
        clicks = self.clean.copy()
        clicks[[15000, 25000, 44000]] += .2
        for y in (self.clean, self.clean + rng.normal(0, .04, len(self.clean)), clicks):
            with self.subTest(kind=float(np.std(y))):
                out, info = dsp.adaptive_notch(y, self.fs)
                self.assertFalse(info["applied"], info)
                np.testing.assert_array_equal(out, np.asarray(y, np.float32))

    def test_02_tone_detection_and_real_attenuation(self):
        measurements = []
        for frequency in (50, 800, 1750):
            y = self.tone(frequency)
            out, info = dsp.adaptive_notch(y, self.fs)
            self.assertTrue(info["applied"])
            self.assertEqual(len(info["notched"]), 1)
            self.assertAlmostEqual(info["notched"][0], frequency, delta=1)
            drop = float(20 * np.log10(amplitude(y, self.fs, frequency) / amplitude(out, self.fs, frequency)))
            self.assertGreater(drop, 20)
            measurements.append({"frequency": frequency, "drop_db": drop, "info": info})
        (REPORT / "tone_attenuation.json").write_text(json.dumps(measurements, ensure_ascii=False, indent=2), encoding="utf-8")

    def test_03_real_harmonics_and_max_count(self):
        y = self.clean + sum(.1 / k * np.sin(2 * np.pi * 50 * k * self.t) for k in range(1, 7))
        out, info = dsp.adaptive_notch(y, self.fs)
        np.testing.assert_allclose(info["notched"], np.arange(1, 7) * 50, atol=1)
        self.assertTrue(np.isfinite(out).all())
        self.assertEqual(len(dsp.adaptive_notch(y, self.fs, max_harmonics=2)[1]["notched"]), 2)

    def test_04_rates_exact_length_finite(self):
        y = self.tone(1750)
        for fs in (8000, 16000, 44100, 48000):
            with self.subTest(fs=fs):
                work = signal.resample_poly(y, fs, self.fs).astype(np.float32)
                out, info = dsp.classic_denoise_with_notch(work, fs)
                self.assertEqual(out.shape, work.shape)
                self.assertEqual(out.dtype, np.float32)
                self.assertTrue(np.isfinite(out).all())
                self.assertAlmostEqual(info["notched"][0], 1750, delta=1)

    def test_05_no_second_fixed_bandstop(self):
        y = self.tone(800)
        filtered, _ = dsp.adaptive_notch(y, self.fs)
        expected = dsp.classic_denoise(filtered, self.fs, bandstop=False)
        out, info = dsp.classic_denoise_with_notch(y, self.fs)
        np.testing.assert_allclose(out, expected, atol=1e-7)
        self.assertEqual(info["notched"], [800.0])

    def test_06_skip_means_spectral_subtraction_only(self):
        out, info = dsp.classic_denoise_with_notch(self.clean, self.fs)
        self.assertFalse(info["applied"])
        np.testing.assert_allclose(out, dsp.classic_denoise(self.clean, self.fs, bandstop=False), atol=1e-7)

    def test_07_manual_parameters_low_and_high_frequency(self):
        for fs, frequencies in ((16000, (20, 100, 35, 65)), (48000, (9000, 11000, 9750, 10250))):
            y = signal.resample_poly(self.tone(50), fs, self.fs).astype(np.float32)
            out, info = dsp.classic_denoise_with_notch(y, fs, auto=False, notch=frequencies)
            designed = dsp.design_bandstop(fs, *frequencies)
            first = signal.sosfiltfilt(designed["sos"], y.astype(np.float64)).astype(np.float32)
            np.testing.assert_allclose(out, dsp.classic_denoise(first, fs, bandstop=False), atol=1e-7)
            self.assertEqual(info["frequencies"], list(frequencies))
            self.assertTrue(np.isfinite(out).all())

    def test_08_invalid_manual_is_not_silently_ignored(self):
        for values in ((1500, 1000, 1750, 1850), (0, 2100, 1750, 1850), (1500, 9000, 1750, 1850)):
            with self.assertRaises(ValueError):
                dsp.classic_denoise_with_notch(self.clean, self.fs, auto=False, notch=values)
        with self.assertRaises(ValueError):
            dsp.classic_denoise(self.clean, self.fs, notch=(0, 20, 5, 10))

    def test_09_short_silence_and_bad_input(self):
        for y in (np.zeros(64000, np.float32), self.clean[:10]):
            out, info = dsp.adaptive_notch(y, self.fs)
            self.assertFalse(info["applied"])
            np.testing.assert_array_equal(out, y)
        for y, fs in (([], 16000), ([float("nan")], 16000), (self.clean, 0)):
            with self.assertRaises(ValueError):
                dsp.adaptive_notch(y, fs)

    def test_10_first_second_silence_does_not_hide_tone(self):
        y = self.tone(800)
        y[:16000] = 0
        _, info = dsp.adaptive_notch(y, self.fs)
        self.assertTrue(info["applied"], info)
        self.assertAlmostEqual(info["notched"][0], 800, delta=1)

    def test_11_source_comparison_evidence(self):
        if not SOURCE.is_file():
            self.skipTest("来版源码不在本机；仅跳过来源对比，其他验证正常执行")
        spec = importlib.util.spec_from_file_location("incoming_dsp_reference", SOURCE)
        incoming = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(incoming)
        rows = []
        cases = {"clean": self.clean, "pure_50Hz": self.tone(50), "tone_1750Hz": self.tone(1750),
                 "hum_50_to_300Hz": self.clean + sum(.1 / k * np.sin(2 * np.pi * 50 * k * self.t) for k in range(1, 7))}
        for name, y in cases.items():
            _, old = incoming.adaptive_notch(y, self.fs)
            _, new = dsp.adaptive_notch(y, self.fs)
            rows.append({"case": name, "incoming_applied": old["applied"], "incoming_notched": old.get("notched", []),
                         "merged_applied": new["applied"], "merged_notched": new.get("notched", [])})
        (REPORT / "incoming_vs_merged.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        self.assertEqual(rows[1]["merged_notched"], [50.0])


class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tkinter as tk
        from app import DSPApp
        cls.root = tk.Tk()
        cls.root.withdraw()
        cls.app = DSPApp(cls.root)
        cls.dialogs = []
        cls.patches = [patch("tkinter.messagebox." + name, side_effect=lambda title, message, **kw: cls.dialogs.append((title, message))) for name in ("showerror", "showinfo", "showwarning")]
        for p in cls.patches:
            p.start()

    @classmethod
    def tearDownClass(cls):
        cls.app.on_close()
        for p in cls.patches:
            p.stop()

    def setUp(self):
        self.app._load_path(str(ROOT / "samples/demo_speech.wav"), silent=True)
        self.app.var_auto_notch.set(True)
        self.app._on_auto_notch_toggle()

    def pump(self, condition):
        deadline = time.monotonic() + 20
        while not condition() and time.monotonic() < deadline:
            self.root.update()
            time.sleep(.03)
        self.assertTrue(condition())

    def load_tone(self):
        y = self.app.y
        t = np.arange(len(y)) / self.app.fs
        path = REPORT / "tone_input.wav"
        sf.write(path, y + .15 * np.sin(2 * np.pi * 800 * t), self.app.fs, subtype="FLOAT")
        self.app._load_path(str(path), silent=True)

    def test_12_detection_preview_uses_same_algorithm_and_keeps_audio(self):
        self.load_tone()
        app = self.app
        original = app.y.copy()
        old_dialogs = len(self.dialogs)
        app.btn_detect_notch.invoke()
        self.pump(lambda: len(self.dialogs) > old_dialogs)
        center = (float(app.var_fs1.get()) + float(app.var_fs2.get())) / 2
        self.assertAlmostEqual(center, 800, delta=1)
        np.testing.assert_array_equal(app.y, original)
        self.assertIsNone(app.yx)

    def test_13_auto_decision_and_actual_response(self):
        self.load_tone()
        app = self.app
        app.on_denoise_classic()
        self.pump(lambda: app.yx is not None)
        self.assertTrue(app._notch_info["applied"])
        self.assertEqual(app._notch_info["notched"], [800.0])
        self.assertIn("实际带阻", app.ax3.get_title())
        out, _ = dsp.classic_denoise_with_notch(app.y, app.fs)
        np.testing.assert_array_equal(app.yx, out)
        app.fig.savefig(REPORT / "auto_notch_gui.png")

    def test_14_manual_snapshot_and_export_metadata(self):
        app = self.app
        app.var_auto_notch.set(False)
        frequencies = (600, 1000, 750, 850)
        for v, value in zip((app.var_fp1, app.var_fp2, app.var_fs1, app.var_fs2), frequencies):
            v.set(str(value))
        app.on_denoise_classic(notify=False)
        app.var_auto_notch.set(True)
        app.var_fs1.set("1750")
        self.pump(lambda: app.yx is not None)
        self.assertEqual(app._notch_info["mode"], "manual")
        self.assertEqual(app._notch_info["frequencies"], list(frequencies))
        with tempfile.TemporaryDirectory(dir=REPORT) as directory:
            path = str(Path(directory) / "notch.wav")
            with patch("tkinter.filedialog.asksaveasfilename", return_value=path):
                app.on_export()
            metadata = json.loads(Path(path + ".json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["notch"]["mode"], "manual")
            self.assertEqual(metadata["notch"]["frequencies"], list(frequencies))
            self.assertIsNone(metadata["model"])

    def test_15_comparison_matches_single_classic_run(self):
        self.load_tone()
        app = self.app
        captured = {}
        def choose(report, results, metrics):
            captured.update(results)
            return "经典谱减法"
        with patch.object(app, "_ask_method_choice", side_effect=choose):
            app.on_compare_denoisers()
            self.pump(lambda: app.yx is not None)
        expected, info = dsp.classic_denoise_with_notch(app.y, app.fs)
        np.testing.assert_array_equal(app.yx, expected)
        self.assertEqual(app._notch_info["notched"], info["notched"])

    def test_16_invalid_manual_keeps_result(self):
        app = self.app
        app.yx = app.y.copy()
        before = app.yx
        app.var_auto_notch.set(False)
        app.var_fp1.set("3000")
        app.var_fp2.set("2000")
        count = len(self.dialogs)
        app.on_denoise_classic()
        self.assertGreater(len(self.dialogs), count)
        self.assertFalse(app.tasks.busy)
        self.assertIs(app.yx, before)

    def test_17_skip_response_and_metadata_reset(self):
        app = self.app
        app.on_denoise_classic(notify=False)
        self.pump(lambda: app.yx is not None)
        self.assertFalse(app._notch_info["applied"])
        self.assertIn("只做谱减", app.ax3.get_title())
        np.testing.assert_array_equal(app.ax3.lines[0].get_ydata(), np.zeros(16384))
        before = app.yx
        app.on_denoise_ml()
        self.pump(lambda: app.yx is not before)
        self.assertIsNone(app._notch_info)

    def test_18_stale_detection_and_processing_are_dropped(self):
        app = self.app
        app.tasks.current_id = "old_notch"
        event = {"task_id": "old_notch", "revision": app.tasks.revision, "name": "陷波检测", "result": {"enabled": True, "center": 700, "halfwidth": 25}}
        app._load_path(str(ROOT / "samples/manba.wav"), silent=True)
        values = tuple(v.get() for v in (app.var_fp1, app.var_fp2, app.var_fs1, app.var_fs2))
        app._handle_signal_result(event)
        self.assertEqual(values, tuple(v.get() for v in (app.var_fp1, app.var_fp2, app.var_fs1, app.var_fs2)))
        event.update(name="经典谱减法", result={"output": app.y.copy(), "notch": {"applied": True}})
        app._handle_signal_result(event)
        self.assertIsNone(app.yx)


if __name__ == "__main__":
    suite = unittest.TestSuite((unittest.defaultTestLoader.loadTestsFromTestCase(AlgorithmTests), unittest.defaultTestLoader.loadTestsFromTestCase(GuiTests)))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    (REPORT / "summary.json").write_text(json.dumps({"tests": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "ok": result.wasSuccessful()}, indent=2), encoding="utf-8")
    raise SystemExit(0 if result.wasSuccessful() else 1)
