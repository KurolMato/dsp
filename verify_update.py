r"""更新回归验证：真实降噪/音乐模型、任务隔离、混音和 Tk 界面。

运行：.venv\Scripts\python -X utf8 verify_update.py
不触发录音或扬声器播放；会实际运行本地已缓存的 VR 分离模型。
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import queue
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from core import audio_io, corpus, data, dsp, metrics, model
from core.enhance import Denoiser, _resample_to
from core.gtcrn import GTCRNDenoiser, verify_against_reference
from core.music import SeparationJob, StemPlayer, mix_tracks
from core.tasks import SignalTaskRunner

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "results" / "verification_20261005"
REPORT.mkdir(parents=True, exist_ok=True)


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.den = Denoiser()
        cls.gt = GTCRNDenoiser()
        assert cls.den.is_ready, cls.den.load_error
        assert cls.gt.is_ready, cls.gt.load_error

    def test_01_sampling_rate_and_exact_length(self):
        for fs in (8000, 16000, 44100, 48000):
            with self.subTest(fs=fs):
                t = np.arange(fs // 4 + 37) / fs
                y = (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
                for den in (self.den, self.gt):
                    out = den.enhance(y, fs=fs)
                    self.assertEqual(out.shape, y.shape)
                    self.assertTrue(np.isfinite(out).all())
                native = _resample_to(y, 16000, fs)
                expected = _resample_to(self.den.enhance(native, fs=16000), fs, 16000, length=len(y))
                np.testing.assert_allclose(self.den.enhance(y, fs=fs), expected, atol=1e-7)

    def test_02_stft_tail_no_silent_truncation(self):
        y = np.random.default_rng(1).normal(0, .1, 1000).astype(np.float32)
        out = dsp.istft(dsp.stft(y), length=len(y))
        # 保留原帧起点以兼容已有模型；Hann 起始几个采样点由原阈值置零。
        # 此回归检查不完整的末帧是否仍被保留和重建。
        np.testing.assert_allclose(out[6:], y[6:], atol=2e-5)
        gt_out = self.gt.enhance(y, 16000)
        self.assertGreater(float(np.linalg.norm(gt_out[-232:])), 1e-4)

    def test_03_real_noise_label_handles_phase_cancellation(self):
        s = np.array([1, 2], np.float32)
        n = np.array([.9, 3], np.float32)
        np.testing.assert_allclose(data._irm(s, n), s / (s + n), atol=1e-6)
        np.testing.assert_allclose(model.ideal_ratio_mask(s, n), s / (s + n), atol=1e-6)
        clean = np.sin(np.arange(16000) * .1).astype(np.float32) * .2
        info = data.random_mixture(clean, seed=2)
        np.testing.assert_allclose(info["noise"], info["noisy"] - clean, atol=1e-7)

    def test_04_content_groups_and_temporal_split(self):
        with tempfile.TemporaryDirectory(dir=REPORT) as directory:
            p = Path(directory)
            rng = np.random.default_rng(2)
            y = rng.normal(0, .1, 16000 * 6).astype(np.float32)
            for name, values in (("a", y), ("duplicate", y), ("b", -y[::-1])):
                sf.write(p / (name + ".wav"), values, 16000, subtype="FLOAT")
            tr, va, manifest = corpus.split_clean_corpus([str(f) for f in p.glob("*.wav")])
            self.assertTrue(tr and va)
            self.assertFalse({r["group"] for r in manifest["train"]} & {r["group"] for r in manifest["validation"]})
            self.assertEqual(len({r["group"] for r in manifest["train"] + manifest["validation"]}), 2)
            _, _, temporal = corpus.split_clean_corpus([str(p / "a.wav")])
            self.assertEqual(temporal["mode"], "single_source_temporal")
            self.assertLessEqual(max(r["end"] for r in temporal["train"]), min(r["start"] for r in temporal["validation"]))
            manifest_path = p / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            loaded, _ = corpus.validation_pool(str(manifest_path))
            self.assertEqual(sum(map(len, loaded)), sum(map(len, va)))
            val_source = manifest["validation"][0]["source"]
            sf.write(val_source, np.zeros_like(y), 16000)
            with self.assertRaises(ValueError):
                corpus.validation_pool(str(manifest_path))

    def test_05_features_match_training_configuration(self):
        spec = dsp.stft(np.ones(2000, np.float32) * .1)
        expected = self.den.features(np.abs(spec))
        np.testing.assert_array_equal(expected, self.den.features(np.abs(spec), np.full(spec.shape, -12)))
        with patch.dict(self.den.meta, {"noise_prefix": "gtcrn", "use_noise_floor": 1}):
            with self.assertRaises(ValueError):
                self.den.features(np.abs(spec))
            with self.assertRaises(ValueError):
                self.den.features(np.abs(spec), np.ones((1, 1)))

    def test_06_cascade_is_sequential(self):
        y = np.random.default_rng(5).normal(0, .1, 4000).astype(np.float32)
        expected = self.den.enhance(self.gt.enhance(y, 16000), fs=16000)
        np.testing.assert_allclose(self.den.enhance_cascaded(y, self.gt, fs=16000), expected, atol=1e-7)

    def test_07_shared_gtcrn_concurrency(self):
        y = np.random.default_rng(5).normal(0, .1, 3200).astype(np.float32)
        expected = [self.gt.enhance(y, 16000), self.gt.enhance(-y, 16000)]
        with ThreadPoolExecutor(2) as pool:
            actual = list(pool.map(lambda x: self.gt.enhance(x, 16000), (y, -y)))
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)

    def test_08_stale_task_snapshot_and_busy_guard(self):
        events = queue.Queue()
        runner = SignalTaskRunner(events.put)
        release = threading.Event()
        audio = np.ones(100, np.float32)
        runner.submit("test", audio, 44100, lambda y, fs: (release.wait(5), y.copy())[1])
        audio.fill(0)
        with self.assertRaises(RuntimeError):
            runner.submit("duplicate", audio, 44100, lambda y, fs: y)
        runner.invalidate()
        release.set()
        event = events.get(timeout=5)
        self.assertFalse(runner.accepts(event))
        self.assertEqual(event["fs"], 44100)
        np.testing.assert_array_equal(event["result"], np.ones(100))
        runner.close()

    def test_09_metrics_unreliable_and_strict_length(self):
        rep = metrics.assess(np.zeros(1000), np.zeros(1000), 16000)
        self.assertIsNone(rep["snr_est_improve_db"])
        self.assertIn("无法可靠估计", metrics.format_report("静音", rep))
        json.dumps(rep, allow_nan=False)
        with self.assertRaises(ValueError):
            metrics.assess(np.ones(1000), np.ones(333), 48000)

    def test_10_official_gtcrn_reference(self):
        reference = verify_against_reference()
        (REPORT / "gtcrn_reference.json").write_text(json.dumps(reference, indent=2), encoding="utf-8")
        self.assertTrue(reference["ok"], reference)
        self.assertGreater(reference["corr"], .999)

    def test_11_synchronized_mix_cursor_seek(self):
        tracks = {"original": np.full((100, 2), .5, np.float32), "vocals": np.full((100, 2), .2, np.float32), "instrumental": np.full((100, 2), .3, np.float32)}
        player = StemPlayer()
        player.load(tracks, 100)
        np.testing.assert_allclose(player.render(10), .5)
        player.set_gains({"vocals": .5})
        np.testing.assert_allclose(player.render(10), .1)
        self.assertEqual(player.position, 20)
        player.seek(.5)
        self.assertEqual(player.position, 50)
        player.stop()
        self.assertEqual(player.position, 0)
        with self.assertRaises(ValueError):
            mix_tracks(tracks, {"vocals": 3})

    def test_12_actual_worker_cancel(self):
        events = queue.Queue()
        job = SeparationJob(events.put)
        job.start(str(ROOT / "samples/music_validation_mix.wav"), "1_HP-UVR", "cpu")
        deadline = time.monotonic() + 35
        saw_separate = False
        while time.monotonic() < deadline:
            event = events.get(timeout=max(.1, deadline - time.monotonic()))
            if event.get("stage") == "separate":
                saw_separate = True
                job.cancel()
                break
            if event["type"] in ("done", "error"):
                self.fail(str(event))
        self.assertTrue(saw_separate)
        self.assertTrue(job.finished.wait(10))
        self.assertIsNotNone(job.process.poll())
        self.assertFalse((job.output / "result.json").exists())
        self.assertEqual(json.loads((job.output / "status.json").read_text())["state"], "cancelled")

    def test_19_actual_multi_stem_manifest(self):
        result = json.loads((REPORT / "four_stem/result.json").read_text(encoding="utf-8"))
        tracks = {}
        for name, path in result["files"].items():
            audio, fs = sf.read(path, dtype="float32", always_2d=True)
            self.assertEqual(audio.shape, (192000, 2))
            self.assertEqual(fs, 48000)
            self.assertTrue(np.isfinite(audio).all())
            tracks[name] = audio
        self.assertTrue({"drums", "bass", "other", "vocals", "instrumental"} <= tracks.keys())
        np.testing.assert_allclose(tracks["instrumental"], tracks["drums"] + tracks["bass"] + tracks["other"], atol=1e-7)
        player = StemPlayer()
        player.load(tracks, fs)
        expected = tracks["drums"] + tracks["bass"] + tracks["other"] + tracks["vocals"]
        np.testing.assert_allclose(player.mix(), expected, atol=1e-7)
        self.assertEqual(player.gains["instrumental"], 0)  # 派生伴奏不能叠加到分轨混音。

    def test_20_stereo_m4a_decode(self):
        import subprocess
        output = REPORT / "codec_test.m4a"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(ROOT / "samples/music_validation_mix.wav"), "-c:a", "aac", str(output)], check=True)
        audio, fs = audio_io.load_audio(str(output))
        self.assertEqual(fs, 48000)
        self.assertEqual(audio.shape[1], 2)
        self.assertLess(abs(len(audio) - 192000), 2048)
        self.assertTrue(np.isfinite(audio).all())

    def test_21_active_model_and_bad_weights(self):
        self.assertEqual(self.den.meta["label_version"], "irm_actual_noise_v2")
        manifest = json.loads(Path(self.den.model_path).with_name("split_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(self.den.meta["split_id"], manifest["split_id"])
        with tempfile.TemporaryDirectory(dir=REPORT) as directory:
            bad = Path(directory) / "bad.npz"
            params, norm, meta = model.load_model(self.den.model_path)
            params["W1"] = np.zeros((1, 1))
            model.save_model(str(bad), params, norm, meta)
            with self.assertRaises(ValueError):
                model.set_active_model_path(str(bad))
            self.assertEqual(Path(model.default_model_path()).resolve(), Path(self.den.model_path).resolve())

    def test_23_ab_listening_respects_input_rate(self):
        from ab_listen import build_variants
        y = (np.sin(np.arange(6000) * .07) * .1).astype(np.float32)
        variants = build_variants(y, 44100, "white", 5, self.gt, self.den, normalize=False)
        self.assertEqual(len(variants), 6)
        for _, audio in variants:
            self.assertEqual(audio.shape, y.shape)
            self.assertTrue(np.isfinite(audio).all())

    def test_24_training_rejects_invalid_options(self):
        from train_model import build_parser, train
        for option, value in (("--epochs", "0"), ("--context", "-1"), ("--file-noise-ratio", "2")):
            with self.subTest(option=option):
                args = build_parser().parse_args([option, value])
                with self.assertRaises(ValueError):
                    train(args)

    def test_25_teacher_model_training_inference_contract(self):
        teacher = Denoiser(str(REPORT / "teacher_run/denoise_mlp.npz"))
        self.assertTrue(teacher.is_ready, teacher.load_error)
        self.assertEqual(teacher.meta["noise_prefix"], "gtcrn")
        y = np.random.default_rng(4).normal(0, .1, 24000).astype(np.float32)
        actual = teacher.enhance(y, fs=48000, gtcrn=self.gt)
        self.assertEqual(actual.shape, y.shape)
        self.assertTrue(np.isfinite(actual).all())
        native = _resample_to(y, 16000, 48000)
        spec = dsp.stft(native)
        prefix = teacher.gtcrn_noise_db(spec, self.gt, native, 16000)
        manual = dsp.istft(spec * teacher.predict_mask(spec, gtcrn_noise_db=prefix), length=len(native))
        np.testing.assert_allclose(actual, _resample_to(manual, 48000, 16000, length=len(y)), atol=1e-7)

    def test_26_roformer_canonical_stems(self):
        result = json.loads((REPORT / "roformer/result.json").read_text(encoding="utf-8"))
        self.assertEqual(set(result["files"]), {"original", "vocals", "instrumental"})
        for path in result["files"].values():
            audio, fs = sf.read(path, dtype="float32", always_2d=True)
            self.assertEqual(audio.shape, (192000, 2))
            self.assertEqual(fs, 48000)

    def test_28_actual_mono_separation(self):
        result = json.loads((REPORT / "mono_separation/result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["channels"], 1)
        self.assertEqual(result["fs"], 16000)
        self.assertEqual(result["samples"], 64000)
        for path in result["files"].values():
            audio, fs = sf.read(path, dtype="float32", always_2d=True)
            self.assertEqual(audio.shape, (64000, 1))
            self.assertEqual(fs, 16000)


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

    def pump(self, condition, seconds=35):
        deadline = time.monotonic() + seconds
        while not condition() and time.monotonic() < deadline:
            self.root.update()
            time.sleep(.03)
        self.assertTrue(condition(), self.app.music_panel.status.get())

    def test_13_import_invalidates_background_and_recording(self):
        app = self.app
        app._load_path(str(ROOT / "samples/demo_speech.wav"), silent=True)
        old = {"task_id": app.tasks.current_id, "revision": app.tasks.revision, "name": "旧结果", "result": app.y.copy()}
        app._load_path(str(ROOT / "samples/manba.wav"), silent=True)
        app._handle_signal_result(old)
        self.assertIsNone(app.yx)
        path = app.path
        app._handle_msg({"type": "recorded", "revision": old["revision"], "path": str(ROOT / "samples/demo_speech.wav")})
        self.assertEqual(app.path, path)

    def test_14_unrelated_error_does_not_cancel_training(self):
        app = self.app
        app._training = True
        try:
            app._handle_msg("__ERROR__播放失败测试")
            self.assertTrue(app._training)
        finally:
            app._training = False

    def test_15_actual_music_panel_job_and_mix_export(self):
        panel = self.app.music_panel
        panel.path.set(str(ROOT / "samples/music_validation_mix.wav"))
        panel.device.set("cpu")
        panel.start()
        self.pump(lambda: not panel.busy)
        self.assertIsNotNone(panel.result)
        self.assertEqual(panel.result["samples"], 192000)
        self.assertEqual(panel.result["channels"], 2)
        self.assertEqual(panel.result["fs"], 48000)
        self.assertEqual(set(panel.player.tracks), {"original", "vocals", "instrumental"})
        self.assertTrue(panel.history.get_children())
        panel.audition("vocals")
        panel.gain_vars["vocals"].set(.5)
        panel.update_gains()
        output = REPORT / "gui_mix.wav"
        with patch("tkinter.filedialog.asksaveasfilename", return_value=str(output)):
            panel.export_mix()
        saved, fs = sf.read(output, dtype="float32", always_2d=True)
        meta = json.loads(Path(str(output) + ".json").read_text(encoding="utf-8"))
        np.testing.assert_allclose(saved, panel.player.tracks["vocals"] * .5 * meta["export_gain"], atol=1e-7)
        self.assertEqual(fs, 48000)
        with patch("tkinter.filedialog.askdirectory", return_value=str(REPORT)):
            panel.export_tracks()
        self.assertTrue(list(REPORT.glob("stems_*/result.json")))
        panel.default_mix()
        panel.mute_vars["vocals"].set(True)
        panel.update_gains()
        self.assertEqual(panel.player.gains["vocals"], 0)
        self.assertEqual(panel.player.gains["instrumental"], 1)
        panel.figure.savefig(REPORT / "music_tracks.png")

    def test_16_history_and_corrupt_result_recovery(self):
        panel = self.app.music_panel
        selection = panel.history.get_children()[0]
        panel.history.selection_set(selection)
        panel.open_history()
        self.pump(lambda: not panel.busy)
        self.assertIsNotNone(panel.result)
        bad = REPORT / "bad_result.json"
        bad.write_text(json.dumps({"fs": 16000, "samples": 1, "channels": 1, "files": {"vocals": str(ROOT / "samples/demo_speech.wav")}}), encoding="utf-8")
        panel.open_manifest(str(bad))
        self.pump(lambda: not panel.busy)
        self.assertIn("一致", panel.status.get())
        self.root.update()  # 错误后定时队列仍能运行。

    def test_17_speech_async_completed(self):
        app = self.app
        app._load_path(str(ROOT / "samples/demo_speech.wav"), silent=True)
        for callback in (app.on_denoise_classic, app.on_denoise_ml, app.on_denoise_gtcrn):
            app.yx = None
            callback()
            self.pump(lambda: app.yx is not None, 20)
            self.assertEqual(len(app.yx), len(app.y))
            self.assertTrue(np.isfinite(app.yx).all())

    def test_18_speech_export_unique_and_metadata(self):
        app = self.app
        output = REPORT / "gui_denoise.wav"
        with patch("tkinter.filedialog.asksaveasfilename", return_value=str(output)):
            app.on_export()
        self.assertTrue(output.is_file())
        self.assertTrue(Path(str(output) + ".json").is_file())
        saved, fs = sf.read(output)
        self.assertEqual(len(saved), len(app.y))
        self.assertEqual(fs, app.fs)

    def test_22_music_small_window_and_all_tracks_accessible(self):
        panel = self.app.music_panel
        result = json.loads((REPORT / "four_stem/result.json").read_text(encoding="utf-8"))
        manifest = REPORT / "four_stem/result.json"
        panel.open_manifest(str(manifest))
        self.pump(lambda: not panel.busy)
        self.app.work_tabs.select(panel)
        self.root.geometry("980x560")
        self.root.deiconify()
        self.root.update()
        self.assertEqual(len(panel.gain_vars), 6)
        self.assertGreater(panel.mixer_canvas.winfo_height(), 25)
        self.assertGreater(panel.start_btn.winfo_width(), 40)
        self.assertGreater(panel.seek.winfo_width(), 300)
        panel.mixer_canvas.yview_moveto(1)
        self.root.update()
        self.assertTrue(panel.mixer_canvas.cget("scrollregion"))
        last = next(w for w in panel.mixer.grid_slaves(row=5, column=3))
        self.assertLessEqual(last.winfo_rooty() + last.winfo_height(), panel.mixer_canvas.winfo_rooty() + panel.mixer_canvas.winfo_height() + 2)
        self.root.withdraw()

    def test_27_actual_training_process_cancel_keeps_active_model(self):
        app = self.app
        active = (ROOT / "models/active_model.json").read_bytes()
        app.on_train_model()
        self.assertTrue(app._training)
        self.pump(lambda: app._train_proc is not None, 10)
        self.assertIsNone(app._train_proc.poll())
        app.on_cancel_train()
        self.pump(lambda: not app._training, 10)
        self.assertIsNotNone(app._train_proc.poll())
        self.assertIn("已取消", app.var_status.get())
        self.assertEqual((ROOT / "models/active_model.json").read_bytes(), active)


if __name__ == "__main__":
    suite = unittest.TestSuite((unittest.defaultTestLoader.loadTestsFromTestCase(CoreTests), unittest.defaultTestLoader.loadTestsFromTestCase(GuiTests)))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    (REPORT / "regression_summary.json").write_text(json.dumps({"tests": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "ok": result.wasSuccessful()}, indent=2), encoding="utf-8")
    raise SystemExit(0 if result.wasSuccessful() else 1)
