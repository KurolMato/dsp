"""新增双人语音分离回归：真实模型、匿名轨道、任务、UI、导出及音乐模块保护。"""
import hashlib
import itertools
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
from scipy.signal import resample_poly

from core.speakers import ROOT, REFERENCE_DIR, MODEL_DIR, SpeakerJob, align_overlap, separate_chunks, speaker_python, model_ready

REPORT = ROOT / "results/verification_speakers_20261010"
REPORT.mkdir(parents=True, exist_ok=True)


def si_sdr(reference, estimate):
    reference, estimate = np.asarray(reference, np.float64), np.asarray(estimate, np.float64)
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    target = reference * np.dot(reference, estimate) / max(np.dot(reference, reference), 1e-12)
    return float(10 * np.log10((np.sum(target ** 2) + 1e-12) / (np.sum((estimate - target) ** 2) + 1e-12)))


class CoreTests(unittest.TestCase):
    def test_01_reference_sum_and_known_sources(self):
        a, fs = sf.read(REFERENCE_DIR / "reference_1.wav", dtype="float32")
        b, _ = sf.read(REFERENCE_DIR / "reference_2.wav", dtype="float32")
        mix, _ = sf.read(REFERENCE_DIR / "overlap_mix.wav", dtype="float32")
        np.testing.assert_allclose(mix, a + b, atol=1e-7)
        self.assertEqual(fs, 16000)
        self.assertEqual(len(mix), 96000)
        self.assertLess(abs(np.corrcoef(a, b)[0, 1]), .5)

    def test_02_overlap_permutation_no_identity(self):
        tracks = np.random.default_rng(4).normal(0, .1, (8000, 2)).astype(np.float32)
        aligned, swapped = align_overlap(tracks, tracks[:, ::-1])
        self.assertTrue(swapped)
        np.testing.assert_array_equal(aligned, tracks)
        _, swapped = align_overlap(np.zeros_like(tracks), np.zeros_like(tracks))
        self.assertFalse(swapped)

    def test_03_chunk_crossfade_matches_unpermuted_tracks(self):
        audio = np.random.default_rng(0).normal(0, .03, 8000 * 11 + 13).astype(np.float32)
        count = [0]
        def infer(piece):
            # 两路须具有不同波形；仅做比例缩放的两列无法用相关性区分。
            result = np.stack((piece, piece ** 2), axis=1)
            count[0] += 1
            return result[:, ::-1] if count[0] % 2 == 0 else result
        out, swaps = separate_chunks(audio, infer)
        self.assertGreater(swaps, 0)
        np.testing.assert_allclose(out[:, 0], audio, atol=1e-7)
        np.testing.assert_allclose(out[:, 1], audio ** 2, atol=1e-7)

    def test_04_bad_model_output_rejected(self):
        with self.assertRaises(ValueError):
            separate_chunks(np.ones(4000), lambda p: np.ones((len(p) - 1, 2)))
        with self.assertRaises(ValueError):
            separate_chunks(np.ones(4000), lambda p: np.full((len(p), 2), np.nan))
        with self.assertRaises(ValueError):
            separate_chunks([], lambda p: p)

    def test_05_real_model_and_reference_quality(self):
        manifest = json.loads((REPORT / "real_separation/result.json").read_text(encoding="utf-8"))
        self.assertEqual(set(manifest["files"]), {"original", "speaker_1", "speaker_2"})
        outputs = [sf.read(manifest["files"][f"speaker_{i}"])[0] for i in (1, 2)]
        references = [sf.read(REFERENCE_DIR / f"reference_{i}.wav")[0] for i in (1, 2)]
        mix = sf.read(REFERENCE_DIR / "overlap_mix.wav")[0]
        permutations = [(sum(si_sdr(r, outputs[j]) for r, j in zip(references, order)), order)
                        for order in itertools.permutations(range(2))]
        _, best = max(permutations)
        scores = [si_sdr(r, outputs[j]) for r, j in zip(references, best)]
        baseline = [si_sdr(r, mix) for r in references]
        improvements = [x - y for x, y in zip(scores, baseline)]
        self.assertGreater(min(improvements), 3)
        for output in outputs:
            self.assertEqual(len(output), 96000)
            self.assertTrue(np.isfinite(output).all())
            self.assertGreater(np.std(output), .005)
            self.assertLess(np.max(np.abs(output)), 1)
        report = {"metric": "SI-SDR at original 16 kHz, best reference permutation", "scores_db": scores,
                  "mixture_scores_db": baseline, "improvements_db": improvements,
                  "reference_to_output": list(best), "seconds": manifest["seconds"],
                  "note": "只对这组已知混合来源的示例评估，不是身份识别或任意录音效果保证。"}
        (REPORT / "reference_quality.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    def test_06_actual_stereo_48k_worker(self):
        mix, fs = sf.read(REFERENCE_DIR / "overlap_mix.wav", dtype="float32")
        # 从重叠说话区间取样；与演示不同的率、声道及非整帧长度。
        y = resample_poly(mix[16000:36000], 3, 1)[:59017]
        stereo = np.stack((y, y * .8), axis=1)
        directory = REPORT / "stereo_validation"
        directory.mkdir(exist_ok=True)
        input_path = directory / "input.wav"
        sf.write(input_path, stereo, 48000, subtype="FLOAT")
        request = {"task_id": "stereo_validation", "input": str(input_path), "output_dir": str(directory),
                   "model_dir": str(MODEL_DIR), "device": "cpu", "source": "hf-mirror"}
        path = directory / "request.json"
        path.write_text(json.dumps(request), encoding="utf-8")
        with (directory / "worker.log").open("w", encoding="utf-8") as log:
            subprocess.run([speaker_python(), "-X", "utf8", str(ROOT / "speaker_worker.py"), str(path)],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=90,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["fs"], 48000)
        self.assertEqual(result["input_channels"], 2)
        self.assertEqual(result["channels"], 1)
        for path in result["files"].values():
            audio, fs = sf.read(path)
            self.assertEqual(fs, 48000)
            self.assertEqual(audio.shape, (59017,))

    def test_07_actual_cancel_terminates_process(self):
        events = queue.Queue()
        job = SpeakerJob(events.put)
        job.start(str(REFERENCE_DIR / "overlap_mix.wav"))
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            event = events.get(timeout=30)
            if event.get("stage") == "load":
                job.cancel()
                break
            if event["type"] == "error":
                self.fail(str(event))
        self.assertTrue(job.finished.wait(10))
        self.assertIsNotNone(job.process.poll())
        self.assertFalse((job.output / "result.json").exists())
        self.assertEqual(json.loads((job.output / "status.json").read_text())["state"], "cancelled")

    def test_08_music_code_and_environment_preserved(self):
        hashes = json.loads((ROOT / "backups/before_speaker_separation_20261010/music_hashes.json").read_text())
        for path, digest in hashes.items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest, path)
        from importlib.metadata import version
        self.assertEqual(version("torch"), "2.14.1+cpu")
        self.assertEqual(version("pymss"), "2.1.7")
        self.assertTrue(model_ready())


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

    def pump(self, condition, timeout=40):
        deadline = time.monotonic() + timeout
        while not condition() and time.monotonic() < deadline:
            self.root.update()
            time.sleep(.03)
        self.assertTrue(condition(), self.panel.status.get())

    def test_09_independent_tab_and_real_job(self):
        self.assertEqual(len(self.app.work_tabs.tabs()), 3)
        self.panel.load_reference()
        self.panel.start()
        self.pump(lambda: not self.panel.busy)
        self.assertIsNotNone(self.panel.result)
        self.assertEqual(self.panel.result["kind"], "speaker_separation")
        self.assertEqual(set(self.panel.player.tracks), {"original", "speaker_1", "speaker_2"})
        self.assertEqual(self.panel.player.fs, 16000)
        self.assertTrue(self.panel.history.get_children())
        self.panel.figure.savefig(REPORT / "speaker_tracks.png")

    def test_10_solo_seek_and_export(self):
        panel = self.panel
        panel.audition("speaker_1")
        self.assertEqual(panel.player.gains["speaker_2"], 0)
        panel.player.seek(2)
        self.assertEqual(panel.player.position, 32000)
        output = REPORT / "gui_speaker_1.wav"
        with patch("tkinter.filedialog.asksaveasfilename", return_value=str(output)):
            panel.export_mix()
        audio, fs = sf.read(output, dtype="float32", always_2d=True)
        meta = json.loads(Path(str(output) + ".json").read_text(encoding="utf-8"))
        np.testing.assert_allclose(audio, panel.player.tracks["speaker_1"] * meta["export_gain"], atol=1e-7)
        self.assertEqual(fs, 16000)
        with patch("tkinter.filedialog.askdirectory", return_value=str(REPORT)):
            panel.export_tracks()
        self.assertTrue(list(REPORT.glob("stems_*/result.json")))

    def test_11_history_wrong_type_and_stale_event(self):
        panel = self.panel
        old = panel.result
        panel.open_manifest(str(ROOT / "results/verification_20261005/real_separation/result.json"))
        self.assertIs(panel.result, old)
        panel.handle_event({"type": "error", "task_id": "old_id", "message": "旧错误"})
        self.assertIs(panel.result, old)
        path = panel.history.get_children()[0]
        panel.open_manifest(path)
        self.pump(lambda: not panel.busy)
        self.assertEqual(panel.result["kind"], "speaker_separation")

    def test_12_missing_dependency_reports_without_hanging(self):
        panel = self.panel
        count = len(self.dialogs)
        with patch("core.speakers.speaker_python", side_effect=RuntimeError("独立环境未安装")):
            panel.start()
        self.assertFalse(panel.busy)
        self.assertGreater(len(self.dialogs), count)

    def test_13_small_window_controls_accessible(self):
        self.app.work_tabs.select(self.panel)
        self.root.geometry("980x560")
        self.root.deiconify()
        self.root.update()
        self.assertGreater(self.panel.start_btn.winfo_width(), 60)
        self.assertGreater(self.panel.seek.winfo_width(), 300)
        last = self.panel.mixer.grid_slaves(row=2, column=3)[0]
        self.assertLess(last.winfo_rooty(), self.panel.winfo_rooty() + self.panel.winfo_height())
        self.root.withdraw()

    def test_14_portable_reference_manifest(self):
        self.panel.open_manifest(str(REFERENCE_DIR / "separation_result.json"))
        self.pump(lambda: not self.panel.busy)
        self.assertEqual(set(self.panel.player.tracks), {"original", "speaker_1", "speaker_2"})
        self.assertEqual(self.panel.result["samples"], 96000)


if __name__ == "__main__":
    suite = unittest.TestSuite((unittest.defaultTestLoader.loadTestsFromTestCase(CoreTests), unittest.defaultTestLoader.loadTestsFromTestCase(GuiTests)))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    (REPORT / "regression_summary.json").write_text(json.dumps({"tests": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "ok": result.wasSuccessful()}, indent=2), encoding="utf-8")
    raise SystemExit(0 if result.wasSuccessful() else 1)
