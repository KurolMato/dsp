"""独立双人语音分离页；复用试听/导出组件，不修改音乐分离模块。"""
import json
from pathlib import Path
import time
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from music_panel import MusicPanel
from core.speakers import SpeakerJob, SpeakerPostJob, MODEL_DIR, OUTPUT_DIR, REFERENCE_DIR, MODEL_ID, LABELS, model_ready
from core import audio_io
from core.speaker_cleanup import DEFAULTS, validate_config


class SpeakerPanel(MusicPanel):
    def __init__(self, *args, **kwargs):
        self.raw_tracks = None
        self.enhanced_tracks = None
        self._switching_version = False
        super().__init__(*args, **kwargs)

    def _build(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)
        ttk.Label(self, text="双人语音分离", font=("", 16, "bold")).grid(row=0, column=0, sticky="w")
        setup = ttk.LabelFrame(self, text="两个人同时说话 → 两条分离音轨", padding=8)
        setup.grid(row=1, column=0, sticky="ew", pady=6)
        setup.columnconfigure(1, weight=1)
        self.path = tk.StringVar()
        ttk.Button(setup, text="导入混合录音…", command=self.choose_input).grid(row=0, column=0, padx=4)
        ttk.Entry(setup, textvariable=self.path, state="readonly").grid(row=0, column=1, columnspan=4, sticky="ew")
        controls = ttk.Frame(setup)
        controls.grid(row=1, column=0, columnspan=5, sticky="ew", pady=6)
        self.device, self.source = tk.StringVar(value="cpu"), tk.StringVar(value="hf-mirror")
        ttk.Label(controls, text="设备").pack(side="left")
        ttk.Combobox(controls, textvariable=self.device, values=("cpu", "auto", "cuda"), state="readonly", width=7).pack(side="left", padx=4)
        ttk.Label(controls, text="下载源").pack(side="left")
        ttk.Combobox(controls, textvariable=self.source, values=("hf-mirror", "huggingface"), state="readonly", width=11).pack(side="left", padx=4)
        self.start_btn = ttk.Button(controls, text="分离两路语音", command=self.start)
        self.start_btn.pack(side="left", padx=4)
        self.cancel_btn = ttk.Button(controls, text="取消任务", command=self.cancel, state="disabled")
        self.cancel_btn.pack(side="left", padx=4)
        row = ttk.Frame(setup)
        row.grid(row=2, column=0, columnspan=5, sticky="ew")
        self.download_btn = ttk.Button(row, text="下载双人模型", command=lambda: self.start(True))
        self.download_btn.pack(side="left")
        ttk.Button(row, text="加载参考混合音频", command=self.load_reference).pack(side="left", padx=4)
        ttk.Button(row, text="参考音频目录", command=lambda: self.open_dir(REFERENCE_DIR)).pack(side="left")
        self.model_info = tk.StringVar()
        ttk.Label(setup, textvariable=self.model_info, wraplength=520).grid(row=3, column=0, columnspan=5, sticky="w", pady=4)
        self.refresh_model_status()
        ttk.Label(setup, text="音轨1/2不代表固定身份。输出为单声道，保留原时长与采样率；模型内部按8 kHz处理。",
                  wraplength=520, foreground="#666").grid(row=4, column=0, columnspan=5, sticky="w")
        state = ttk.Frame(self)
        state.grid(row=2, column=0, sticky="ew")
        state.columnconfigure(0, weight=1)
        self.status = tk.StringVar(value="导入两人混合录音，或使用附带参考音频。")
        ttk.Label(state, textvariable=self.status, wraplength=520).grid(row=0, column=0, sticky="w")
        self.progress = ttk.Progressbar(state)
        self.progress.grid(row=1, column=0, sticky="ew", pady=4)
        workspace = ttk.Notebook(self)
        workspace.grid(row=3, column=0, sticky="nsew")
        page, historypage = ttk.Frame(workspace, padding=6), ttk.Frame(workspace, padding=6)
        postpage = ttk.Frame(workspace, padding=6)
        workspace.add(page, text="同步试听 / 导出")
        workspace.add(postpage, text="逐轨降噪 / 噪声门")
        workspace.add(historypage, text="分离历史")
        self._build_postprocess(postpage)
        page.columnconfigure(0, weight=1)
        page.rowconfigure(3, weight=1)
        transport = ttk.Frame(page)
        transport.grid(row=0, column=0, sticky="ew")
        for text, callback in (("播放", self.play), ("暂停", self.player.pause), ("停止", self.player.stop)):
            ttk.Button(transport, text=text, command=callback).pack(side="left", padx=2)
        self.time_label = tk.StringVar(value="00:00 / 00:00")
        ttk.Label(transport, textvariable=self.time_label).pack(side="right")
        self.seek = ttk.Scale(page, from_=0, to=1, command=self.seek_to)
        self.seek.grid(row=1, column=0, sticky="ew", pady=4)
        self._moving_seek = False
        self.mixer = ttk.Frame(page)
        self.mixer.grid(row=2, column=0, sticky="ew")
        plot = ttk.Frame(page)
        plot.grid(row=3, column=0, sticky="nsew")
        self.figure = Figure(figsize=(7, 2.2), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("分离后显示原音及两条音轨")
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        actions = ttk.Frame(page)
        actions.grid(row=4, column=0, sticky="ew", pady=6)
        for text, name in (("听混合原音", "original"), ("听音轨1", "speaker_1"), ("听音轨2", "speaker_2")):
            ttk.Button(actions, text=text, command=lambda n=name: self.audition(n)).pack(side="left", padx=2)
        ttk.Button(actions, text="导出当前试听…", command=self.export_mix).pack(side="right", padx=2)
        ttk.Button(actions, text="导出两路音轨…", command=self.export_tracks).pack(side="right", padx=2)
        self.limit = tk.BooleanVar(value=True)
        ttk.Checkbutton(page, text="试听导出防削顶（统一降低增益）", variable=self.limit).grid(row=5, column=0, sticky="w")
        version_row = ttk.Frame(page)
        version_row.grid(row=6, column=0, sticky="ew", pady=2)
        ttk.Label(version_row, text="试听版本：").pack(side="left")
        self.version = tk.StringVar(value="原始分离")
        ttk.Combobox(version_row, textvariable=self.version, values=("原始分离", "增强结果"),
                     state="readonly", width=12).pack(side="left")
        self.version.trace_add("write", lambda *_: self.switch_version())
        historypage.columnconfigure(0, weight=1)
        historypage.rowconfigure(0, weight=1)
        self.history = ttk.Treeview(historypage, columns=("model", "length"), show="tree headings")
        for key, label in (("#0", "混合录音 / 时间"), ("model", "分离模型"), ("length", "时长")):
            self.history.heading(key, text=label)
        for key, width in (("#0", 240), ("model", 120), ("length", 70)):
            self.history.column(key, width=width)
        self.history.grid(row=0, column=0, sticky="nsew")
        row = ttk.Frame(historypage)
        row.grid(row=1, column=0, sticky="ew", pady=4)
        ttk.Button(row, text="载入所选结果", command=self.open_history).pack(side="left")
        ttk.Button(row, text="刷新", command=self.refresh_history).pack(side="left", padx=4)
        ttk.Button(row, text="打开结果清单…", command=self.open_manifest).pack(side="left")
        ttk.Button(row, text="结果目录", command=lambda: self.open_dir(OUTPUT_DIR)).pack(side="left", padx=4)
        self.history.bind("<Double-1>", lambda _e: self.open_history())

    def choose_input(self):
        if self.busy:
            return
        path = filedialog.askopenfilename(title="选择双人混合录音", filetypes=[("音频", "*.wav *.mp3 *.flac *.m4a *.ogg *.aiff"), ("全部", "*.*")])
        if path:
            self.path.set(path)

    def load_reference(self):
        if self.busy:
            return
        path = REFERENCE_DIR / "overlap_mix.wav"
        if not path.is_file():
            messagebox.showerror("参考音频不存在", "请运行 make_speaker_reference.py 生成参考音频。")
            return
        self.path.set(str(path))
        self.status.set("参考混合音频已载入，点击「分离两路语音」。原始两路在参考目录中。")

    def refresh_model_status(self):
        ready = model_ready()
        self.model_info.set("SepFormer 双人分离 · " + ("本地模型已就绪" if ready else "首次使用需下载约113 MB模型"))

    def set_busy(self, busy):
        self.busy = busy
        self.start_btn.config(state="disabled" if busy else "normal")
        self.download_btn.config(state="disabled" if busy else "normal")
        self.cancel_btn.config(state="normal" if busy else "disabled")
        if hasattr(self, "post_apply_btn"):
            self.post_apply_btn.config(state="disabled" if busy or self.raw_tracks is None else "normal")
            for widget in self.post_inputs:
                widget.config(state="disabled" if busy else "normal")
        if not busy:
            self.progress.stop()

    def start(self, download_only=False):
        if self.busy:
            return
        if not download_only and not self.path.get():
            messagebox.showinfo("选择录音", "请先导入混合录音或加载参考音频。")
            return
        self.player.stop()
        self.job = SpeakerJob(self.events.put)
        self.current_id = self.job.id
        self.set_busy(True)
        self.status.set("正在启动双人分离任务…")
        try:
            self.job.start(self.path.get(), self.device.get(), self.source.get(), download_only)
        except Exception as exc:
            self.set_busy(False)
            self.status.set(str(exc))
            messagebox.showerror("双人分离启动失败", str(exc))

    def handle_event(self, event):
        if self.closed or event.get("task_id") != self.current_id:
            return
        if event.get("type") in ("error", "load_error"):
            if self.current_id in self._terminal and event["type"] != "load_error":
                return
            self._terminal.add(self.current_id)
            self.set_busy(False)
            self.progress.config(mode="determinate", value=0)
            self.status.set(event["message"])
            messagebox.showerror("双人语音分离失败", event["message"])
            return
        if event.get("type") == "loaded":
            self.raw_tracks = event["raw_tracks"]
            self.enhanced_tracks = event.get("enhanced_tracks")
            for name, config in (event.get("post_configs") or {}).items():
                for key, variable in self.post_vars[name].items():
                    variable.set(config[key] * 100 if key == "denoise_strength" else config[key])
            self._switching_version = True
            self.version.set("增强结果" if self.enhanced_tracks is not None and event["result"].get("audition_version") != "raw" else "原始分离")
            self._switching_version = False
        super().handle_event(event)
        if event.get("type") == "loaded":
            action = "逐轨后处理完成，可切换原始/增强对比" if self.enhanced_tracks is not None else "两路分离完成"
            self.status.set(f"{action} · {self.result['fs']} Hz · {self.result['samples']/self.result['fs']:.1f}秒")

    def build_mixer(self):
        super().build_mixer()
        for row, name in enumerate(self.player.tracks):
            for widget in self.mixer.grid_slaves(row=row, column=0):
                widget.config(text=LABELS.get(name, name), width=10)
        for artist, name in zip(self.ax.lines, self.player.tracks):
            artist.set_label(LABELS.get(name, name))
        self.ax.legend(fontsize=8)
        self.ax.set_title("两路语音 · " + self.version.get())
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def refresh_history(self):
        for item in self.history.get_children():
            self.history.delete(item)
        if OUTPUT_DIR.exists():
            for path in sorted(OUTPUT_DIR.glob("*/result.json"), reverse=True)[:100]:
                try:
                    result = json.loads(path.read_text(encoding="utf-8"))
                    if result.get("kind") != "speaker_separation":
                        continue
                    self.history.insert("", "end", iid=str(path), text=Path(result["input"]).name + " · " + path.parent.name,
                                        values=("双人 + 后处理" if result.get("postprocess") else "SepFormer · 双人", f"{result['samples']/result['fs']:.1f}s"))
                except (OSError, ValueError, KeyError):
                    continue

    def open_manifest(self, path=None):
        if self.busy:
            return
        path = path or filedialog.askopenfilename(title="打开双人分离结果", filetypes=[("结果清单", "*.json")])
        if not path:
            return
        try:
            result = json.loads(Path(path).read_text(encoding="utf-8"))
            if result.get("kind") != "speaker_separation" or set(result["files"]) != set(LABELS):
                raise ValueError("请打开双人语音分离结果清单")
            parent = Path(path).resolve().parent
            for key in ("files", "raw_files", "enhanced_files"):
                if key in result:
                    if set(result[key]) != set(LABELS):
                        raise ValueError("原始/增强音轨清单不完整")
                    result[key] = {name: str(p if p.is_absolute() else parent / p)
                                   for name, value in result[key].items() for p in [Path(value)]}
        except Exception as exc:
            messagebox.showerror("结果载入失败", str(exc))
            return
        self.current_id = "history_" + str(time.time_ns())
        self.set_busy(True)
        self.status.set("正在载入双人分离历史音轨…")
        self.load_result(result, self.current_id)

    def _build_postprocess(self, page):
        container = page
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)
        self.post_canvas = tk.Canvas(container, height=280, width=10, highlightthickness=0)
        self.post_canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.post_canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.post_canvas.config(yscrollcommand=scrollbar.set)
        page = ttk.Frame(self.post_canvas)
        window = self.post_canvas.create_window((0, 0), window=page, anchor="nw")
        page.bind("<Configure>", lambda _e: self.post_canvas.config(scrollregion=self.post_canvas.bbox("all")))
        self.post_canvas.bind("<Configure>", lambda e: self.post_canvas.itemconfigure(window, width=e.width))
        self.post_canvas.bind("<MouseWheel>", lambda e: self.post_canvas.yview_scroll(-int(e.delta / 120), "units"))
        page.columnconfigure(0, weight=1)
        self.post_vars, self.post_inputs = {}, []
        ttk.Label(page, text="每次从原始分离音轨计算，保留原音。普通降噪不能保证清除同时说话的另一个人。",
                  wraplength=500, foreground="#666").grid(row=0, column=0, sticky="w", pady=4)
        for row, name in enumerate(("speaker_1", "speaker_2"), 1):
            box = ttk.LabelFrame(page, text=LABELS[name], padding=6)
            box.grid(row=row, column=0, sticky="ew", pady=4)
            box.columnconfigure(1, weight=1)
            variables = {"denoise_enabled": tk.BooleanVar(value=True), "denoise_strength": tk.DoubleVar(value=35),
                         "gate_enabled": tk.BooleanVar(value=False), "gate_threshold_db": tk.DoubleVar(value=-45),
                         "gate_reduction_db": tk.DoubleVar(value=18)}
            self.post_vars[name] = variables
            for r, (text, key) in enumerate((("启用轻度降噪（GTCRN）", "denoise_enabled"),
                                            ("启用噪声门（停顿处压低弱声）", "gate_enabled"))):
                widget = ttk.Checkbutton(box, text=text, variable=variables[key])
                widget.grid(row=r * 2, column=0, columnspan=3, sticky="w")
                self.post_inputs.append(widget)
            for r, (label, key, low, high) in enumerate((("降噪强度 %", "denoise_strength", 0, 100),
                                                        ("门限 dBFS", "gate_threshold_db", -65, -20),
                                                        ("压低 dB", "gate_reduction_db", 0, 36))):
                grid_row = (1, 3, 4)[r]
                ttk.Label(box, text=label).grid(row=grid_row, column=0, sticky="w")
                slider = ttk.Scale(box, from_=low, to=high, variable=variables[key])
                slider.grid(row=grid_row, column=1, sticky="ew", padx=6)
                entry = ttk.Entry(box, textvariable=variables[key], width=7)
                entry.grid(row=grid_row, column=2)
                self.post_inputs.extend((slider, entry))
        self.post_apply_btn = ttk.Button(page, text="应用后处理（两条音轨）", command=self.apply_postprocess, state="disabled")
        self.post_apply_btn.grid(row=3, column=0, sticky="ew", pady=4)
        row = ttk.Frame(page)
        row.grid(row=4, column=0, sticky="ew")
        for text, name, enhanced in (("听原始1", "speaker_1", False), ("听增强1", "speaker_1", True),
                                     ("听原始2", "speaker_2", False), ("听增强2", "speaker_2", True)):
            ttk.Button(row, text=text, command=lambda n=name, e=enhanced: self.compare_track(n, e)).pack(side="left", padx=2)
        ttk.Button(row, text="暂停", command=self.player.pause).pack(side="left", padx=2)
        ttk.Button(row, text="停止", command=self.player.stop).pack(side="left", padx=2)
        ttk.Label(page, text="建议先用 25～40% 降噪；噪声门默认关闭。门限过高会吞轻声和尾音。\n"
                            "门限按原音电平计算，不跟随试听音量；更改参数后需点“应用”。",
                  wraplength=500).grid(row=5, column=0, sticky="w", pady=4)

    def apply_postprocess(self):
        if self.busy:
            return
        if self.result is None or self.raw_tracks is None:
            messagebox.showinfo("先分离语音", "请先分离或载入双人分离结果。")
            return
        try:
            configs = {}
            for name, variables in self.post_vars.items():
                values = {key: var.get() for key, var in variables.items()}
                values["denoise_strength"] /= 100
                configs[name] = validate_config(values)
            job = SpeakerPostJob(self.events.put)
            self.job, self.current_id = job, job.id
            self.set_busy(True)
            self.status.set("正在逐轨后处理…原始分离结果保留。")
            job.start(self.result, configs)
        except Exception as exc:
            self.set_busy(False)
            self.status.set(str(exc))
            messagebox.showerror("后处理失败", str(exc))

    def load_result(self, result, task_id):
        def work():
            try:
                cache = {}
                def read(files):
                    tracks = {}
                    if set(files) != set(LABELS):
                        raise ValueError("音轨清单不完整")
                    for name, path in files.items():
                        if path not in cache:
                            raw, fs = audio_io.load_audio(path)
                            raw = raw[:, None] if raw.ndim == 1 else raw
                            if fs != result["fs"] or raw.shape != (result["samples"], result["channels"]) or not np.isfinite(raw).all():
                                raise ValueError("音轨与结果清单不匹配")
                            cache[path] = raw
                        tracks[name] = cache[path]
                    return tracks
                raw = read(result.get("raw_files", result["files"]))
                enhanced = read(result.get("enhanced_files", result["files"])) if result.get("postprocess") else None
                configs = {name: validate_config(result["postprocess"]["tracks"][name])
                           for name in ("speaker_1", "speaker_2")} if enhanced is not None else None
                selected = enhanced if enhanced is not None and result.get("audition_version") != "raw" else raw
                self.events.put({"type": "loaded", "task_id": task_id, "result": result, "tracks": selected,
                                 "raw_tracks": raw, "enhanced_tracks": enhanced, "post_configs": configs})
            except Exception as exc:
                self.events.put({"type": "load_error", "task_id": task_id, "message": str(exc)})
        threading.Thread(target=work, daemon=True).start()

    def switch_version(self):
        if self._switching_version or self.raw_tracks is None:
            return
        enhanced = self.version.get() == "增强结果"
        if enhanced and self.enhanced_tracks is None:
            self._switching_version = True
            self.version.set("原始分离")
            self._switching_version = False
            messagebox.showinfo("尚无增强结果", "先在“逐轨降噪 / 噪声门”页应用后处理。")
            return
        position, playing, gains = self.player.position / self.player.fs, self.player.playing, dict(self.player.gains)
        self.player.load(self.enhanced_tracks if enhanced else self.raw_tracks, self.result["fs"])
        self.player.set_gains(gains)
        self.player.seek(position)
        self.ax.clear()
        for name, audio in self.player.tracks.items():
            step = max(1, len(audio) // 3000)
            self.ax.plot(np.arange(0, len(audio), step) / self.player.fs, audio[::step].mean(axis=1),
                         linewidth=.65, label=LABELS[name])
        self.ax.set_title("两路语音 · " + self.version.get())
        self.ax.set_xlabel("时间（秒）"); self.ax.legend(fontsize=8)
        self.figure.tight_layout(); self.canvas.draw_idle()
        self.status.set("试听版本：" + self.version.get() + "；播放位置与音量保持。")
        if playing:
            self.play()

    def compare_track(self, name, enhanced):
        if self.raw_tracks is None:
            return
        if enhanced and self.enhanced_tracks is None:
            messagebox.showinfo("尚无增强结果", "先应用后处理。")
            return
        self.version.set("增强结果" if enhanced else "原始分离")
        self.audition(name)
        if not self.player.playing:
            self.play()

    def export_mix(self):
        if not self.player.tracks:
            return
        path = filedialog.asksaveasfilename(title="导出当前试听版本", defaultextension=".wav",
            initialfile="speaker_mix.wav", filetypes=[("WAV", "*.wav"), ("FLAC", "*.flac")])
        if not path:
            return
        try:
            import soundfile as sf
            audio = self.player.mix()
            peak = float(np.max(np.abs(audio)))
            gain = .99 / peak if self.limit.get() and peak > .99 else 1.
            if not self.limit.get() and peak > 1 and Path(path).suffix.lower() == ".flac":
                raise ValueError("FLAC 超出满幅，请启用防削顶或降低音量")
            sf.write(path, audio * gain, self.player.fs,
                     subtype="PCM_24" if Path(path).suffix.lower() == ".flac" else "FLOAT")
            metadata = {"source_result": self.result, "audition_version": "enhanced" if self.version.get() == "增强结果" else "raw",
                        "gains": dict(self.player.gains), "export_gain": gain, "fs": self.player.fs}
            Path(path + ".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            self.status.set(f"已导出{self.version.get()}：{path}")
        except Exception as exc:
            messagebox.showerror("导出失败", str(exc))

    def export_tracks(self):
        if self.raw_tracks is None:
            return
        directory = filedialog.askdirectory(title="导出原始及增强音轨")
        if not directory:
            return
        try:
            import soundfile as sf
            out = Path(directory) / ("stems_" + time.strftime("%Y%m%d_%H%M%S") + "_" + str(time.time_ns() % 1000000))
            out.mkdir(exist_ok=False)
            raw_files, enhanced_files = {}, {}
            for name, audio in self.raw_tracks.items():
                path = out / ("original.wav" if name == "original" else name + "_raw.wav")
                sf.write(path, audio, self.player.fs, subtype="FLOAT")
                raw_files[name] = str(path)
            if self.enhanced_tracks is not None:
                enhanced_files["original"] = raw_files["original"]
                for name in ("speaker_1", "speaker_2"):
                    path = out / (name + "_enhanced.wav")
                    sf.write(path, self.enhanced_tracks[name], self.player.fs, subtype="FLOAT")
                    enhanced_files[name] = str(path)
            use_enhanced = bool(enhanced_files) and self.version.get() == "增强结果"
            result = {**self.result, "files": enhanced_files if use_enhanced else raw_files, "raw_files": raw_files,
                      "audition_version": "enhanced" if use_enhanced else "raw"}
            if enhanced_files:
                result["enhanced_files"] = enhanced_files
            (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            self.status.set(f"已导出原始和可用的增强音轨：{out}")
        except Exception as exc:
            messagebox.showerror("导出失败", str(exc))
