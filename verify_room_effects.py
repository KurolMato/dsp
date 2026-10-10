"""混响/去混响回归：实际算法、拖尾、后台来源、导出；不播放扬声器。"""
from pathlib import Path
import json
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from core import audio_io, room_effects

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "results/verification_20261010"
REPORT.mkdir(parents=True, exist_ok=True)


class AlgorithmTests(unittest.TestCase):
    def test_01_identity_silence_short(self):
        y = np.random.default_rng(1).normal(0, .1, 6400).astype(np.float32)
        np.testing.assert_array_equal(room_effects.add_reverb(y, 16000, wet=0), y)
        np.testing.assert_array_equal(room_effects.dereverb(y, 16000, strength=0), y)
        np.testing.assert_array_equal(room_effects.dereverb(y[:333], 16000), y[:333])
        for algo in (room_effects.add_reverb, room_effects.dereverb):
            np.testing.assert_array_equal(algo(np.zeros((9000, 2)), 16000), np.zeros((9000, 2)))

    def test_02_rates_channels_length_finite(self):
        for fs in (8000, 16000, 44100, 48000):
            for channels in (1, 2):
                with self.subTest(fs=fs, channels=channels):
                    y = np.random.default_rng(5).normal(0, .03, (fs // 2 + 73, channels)).astype(np.float32)
                    if channels == 1:
                        y = y[:, 0]
                    reverberated = room_effects.add_reverb(y, fs, wet=.4)
                    out = room_effects.dereverb(reverberated, fs)
                    for a in (reverberated, out):
                        self.assertEqual(a.shape, y.shape)
                        self.assertEqual(a.dtype, np.float32)
                        self.assertTrue(np.isfinite(a).all())

    def test_03_impulse_pre_delay_and_tail(self):
        fs = 16000
        y = np.zeros(fs * 2, np.float32)
        y[1000] = .5
        r = room_effects.add_reverb(y, fs, wet=.5, pre_delay_ms=30)
        self.assertAlmostEqual(float(r[1000]), .25, places=6)
        np.testing.assert_allclose(r[1001:1480], 0, atol=1e-12)  # FFT 卷积有浮点舍入噪声。
        self.assertGreater(np.linalg.norm(r[1480:]), .01)
        np.testing.assert_array_equal(r, room_effects.add_reverb(y, fs, wet=.5, pre_delay_ms=30))

    def test_04_known_speech_tail_reduction(self):
        y, fs = audio_io.load_audio(str(ROOT / "samples/demo_speech.wav"))
        y = audio_io.to_mono(y)
        y[round(fs * 1.9):] = 0
        r = room_effects.add_reverb(y, fs, rt60=.9, wet=.5)
        out = room_effects.dereverb(r, fs, strength=.9)
        start, end = round(fs * 1.95), round(fs * 2.5)
        tail_drop = float(10 * np.log10(np.sum(r[start:end] ** 2) / np.sum(out[start:end] ** 2)))
        retention = float(np.sum(out[:start] ** 2) / np.sum(r[:start] ** 2))
        self.assertGreater(tail_drop, 1)
        self.assertGreater(retention, .5)  # 不能以整段静音冒充去混响。
        for name, a in (("dry", y), ("reverb", r), ("dereverb", out)):
            sf.write(REPORT / (name + ".wav"), a, fs, subtype="FLOAT")
        (REPORT / "tail_measurement.json").write_text(json.dumps({"fs": fs, "rt60": .9, "wet": .5, "strength": .9, "tail_window_seconds": [1.95, 2.5], "tail_drop_db": tail_drop, "speech_energy_ratio": retention}, indent=2), encoding="utf-8")

    def test_05_multiblock_no_cut_or_spike(self):
        fs = 16000
        t = np.arange(fs * 9 + 137) / fs
        y = (np.sin(t * 2 * np.pi * 220) * .1).astype(np.float32)
        out = room_effects.dereverb(y, fs)
        self.assertEqual(out.shape, y.shape)
        self.assertTrue(np.isfinite(out).all())
        self.assertLess(np.max(np.abs(np.diff(out[fs * 7 - 20:fs * 8 + 20]))), .05)

    def test_06_invalid_parameters(self):
        y = np.ones(16000, np.float32)
        for params in ({"wet": -1}, {"rt60": 0}, {"pre_delay_ms": 300}, {"damping": float("nan")}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                room_effects.add_reverb(y, 16000, **params)
        for params in ({"strength": 2}, {"taps": 3.5}, {"iterations": 0}, {"delay": 0}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                room_effects.dereverb(y, 16000, **params)
        for audio, fs in (([], 16000), ([float("nan")], 16000), (y, 0), (np.ones((300, 3)), 16000)):
            with self.assertRaises(ValueError):
                room_effects.dereverb(audio, fs)

    def test_07_peak_protection_keeps_channel_ratio(self):
        y = np.zeros((16000, 2), np.float32)
        y[500] = (5, 2.5)
        out = room_effects.add_reverb(y, 16000, wet=.1)
        self.assertLessEqual(float(np.max(np.abs(out))), .990001)
        self.assertAlmostEqual(float(out[500, 0] / out[500, 1]), 2, places=5)


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
        self.app.var_reverb_rt60.set("1.0")
        self.app.var_reverb_wet.set("30")
        self.app.var_dereverb_strength.set("70")

    def pump(self, done, timeout=20):
        until = time.monotonic() + timeout
        while not done() and time.monotonic() < until:
            self.root.update()
            time.sleep(.03)
        self.assertTrue(done())

    def test_08_add_then_dereverb_and_export_metadata(self):
        app = self.app
        app.btn_reverb.invoke()
        self.pump(lambda: len(app._processing_history) == 1)
        mixed = app.yx.copy()
        self.assertEqual(app.var_effect_source.get(), "当前处理结果")
        app.btn_dereverb.invoke()
        self.pump(lambda: len(app._processing_history) == 2)
        np.testing.assert_allclose(app.yx, room_effects.dereverb(mixed, app.fs), atol=1e-6)
        with tempfile.TemporaryDirectory(dir=REPORT) as directory:
            path = str(Path(directory) / "effects.wav")
            with patch("tkinter.filedialog.asksaveasfilename", return_value=path):
                app.on_export()
            audio, fs = sf.read(path)
            meta = json.loads(Path(path + ".json").read_text(encoding="utf-8"))
            self.assertEqual(len(audio), len(app.y))
            self.assertEqual(fs, app.fs)
            self.assertIsNone(meta["model"])
            self.assertEqual([p["algorithm"] for p in meta["processing"]], ["添加混响", "WPE 去混响"])
            self.assertEqual(meta["processing"][0]["parameters"]["rt60"], 1)
        app.fig.savefig(REPORT / "effects_gui_plot.png")

    def test_09_original_source_and_parameter_snapshot(self):
        app = self.app
        app.on_reverb()
        app.var_reverb_rt60.set("4")
        self.pump(lambda: app.yx is not None)
        self.assertEqual(app._processing_history[0]["parameters"]["rt60"], 1)
        app.var_effect_source.set("原信号")
        previous = app.yx
        app.on_dereverb()
        self.pump(lambda: app.yx is not previous)
        np.testing.assert_allclose(app.yx, room_effects.dereverb(app.y, app.fs), atol=1e-6)
        self.assertEqual(len(app._processing_history), 1)

    def test_10_bad_parameter_keeps_previous_result(self):
        app = self.app
        app.on_reverb()
        self.pump(lambda: app.yx is not None)
        previous = app.yx
        count = len(self.dialogs)
        app.var_reverb_wet.set("200")
        app.on_reverb()
        self.pump(lambda: len(self.dialogs) > count)
        self.assertIs(app.yx, previous)
        self.assertFalse(app.tasks.busy)
        app.var_reverb_wet.set("bad")
        app.on_reverb()
        self.assertIs(app.yx, previous)

    def test_11_import_drops_stale_room_result(self):
        app = self.app
        revision = app.tasks.revision
        app.tasks.current_id = "stale_room"
        event = {"name": "添加混响", "revision": revision, "task_id": "stale_room", "audio": app.y.copy(), "result": {"output": app.y.copy(), "history": [{"algorithm": "添加混响"}]}}
        app._load_path(str(ROOT / "samples/manba.wav"), silent=True)
        app._handle_signal_result(event)
        self.assertIsNone(app.yx)
        self.assertEqual(app._processing_history, [])

    def test_12_missing_processed_source_is_guarded(self):
        app = self.app
        app.var_effect_source.set("当前处理结果")
        count = len(self.dialogs)
        app.on_dereverb()
        self.assertGreater(len(self.dialogs), count)
        self.assertFalse(app.tasks.busy)

    def test_13_denoise_clears_effect_history(self):
        app = self.app
        app.on_reverb()
        self.pump(lambda: app.yx is not None)
        previous = app.yx
        app.on_denoise_ml()
        self.pump(lambda: app.yx is not previous)
        self.assertEqual(app._processing_history, [])


if __name__ == "__main__":
    suite = unittest.TestSuite((unittest.defaultTestLoader.loadTestsFromTestCase(AlgorithmTests), unittest.defaultTestLoader.loadTestsFromTestCase(GuiTests)))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    (REPORT / "room_effects_summary.json").write_text(json.dumps({"tests": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "ok": result.wasSuccessful()}, indent=2), encoding="utf-8")
    raise SystemExit(0 if result.wasSuccessful() else 1)
