"""音乐分离工作区：模型、任务、共享播放游标、音轨混音和导出。"""
from __future__ import annotations

import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from core import audio_io
from core.music import MODEL_DIR, OUTPUT_DIR, PRESETS, STEM_LABELS, SeparationJob, StemPlayer


class MusicPanel(ttk.Frame):
    def __init__(self, parent, before_play=None):
        super().__init__(parent, padding=10)
        self.before_play = before_play or audio_io.stop_play
        self.player = StemPlayer()
        self.job = None
        self.current_id = None
        self.busy = False
        self.closed = False
        self.result = None
        self.events = queue.Queue()
        self._terminal = set()
        self.gain_vars, self.mute_vars, self.solo_vars = {}, {}, {}
        self._build()
        self.refresh_history()
        self._tick_id = self.after(100, self._tick)

    def _build(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)
        ttk.Label(self, text="音乐音源分离", font=("", 16, "bold")).grid(row=0, column=0, sticky="w")
        setup = ttk.LabelFrame(self, text="1. 选择歌曲与模型", padding=8)
        setup.grid(row=1, column=0, sticky="ew", pady=6)
        setup.columnconfigure(1, weight=1)
        self.path = tk.StringVar()
        ttk.Button(setup, text="导入歌曲…", command=self.choose_input).grid(row=0, column=0, padx=(0, 6))
        ttk.Entry(setup, textvariable=self.path, state="readonly").grid(row=0, column=1, columnspan=4, sticky="ew")
        self.model = tk.StringVar(value=next(iter(PRESETS)))
        self.model_combo = ttk.Combobox(setup, textvariable=self.model, values=list(PRESETS), state="readonly", width=36)
        self.model_combo.grid(row=1, column=0, columnspan=2, sticky="ew", pady=6)
        self.device = tk.StringVar(value="auto")
        ttk.Combobox(setup, textvariable=self.device, values=("auto", "cpu", "cuda"), state="readonly", width=8).grid(row=1, column=2, padx=4)
        self.source = tk.StringVar(value="modelscope")
        ttk.Combobox(setup, textvariable=self.source, values=("modelscope", "hf-mirror", "huggingface"), state="readonly", width=12).grid(row=1, column=3, padx=4)
        self.start_btn = ttk.Button(setup, text="开始分离", command=self.start)
        self.start_btn.grid(row=1, column=4, padx=4)
        row = ttk.Frame(setup)
        row.grid(row=2, column=0, columnspan=5, sticky="ew")
        self.download_btn = ttk.Button(row, text="下载所选模型", command=lambda: self.start(True))
        self.download_btn.pack(side="left")
        self.cancel_btn = ttk.Button(row, text="取消任务", command=self.cancel, state="disabled")
        self.cancel_btn.pack(side="left", padx=5)
        ttk.Button(row, text="模型目录", command=lambda: self.open_dir(MODEL_DIR)).pack(side="left")
        ttk.Button(row, text="结果目录", command=lambda: self.open_dir(OUTPUT_DIR)).pack(side="left", padx=5)
        self.model_info = tk.StringVar()
        ttk.Label(setup, textvariable=self.model_info, foreground="#555", wraplength=780).grid(row=3, column=0, columnspan=5, sticky="w")
        self.model.trace_add("write", lambda *_: self.refresh_model_status())
        self.refresh_model_status()
        state = ttk.Frame(self)
        state.grid(row=2, column=0, sticky="ew")
        state.columnconfigure(0, weight=1)
        self.status = tk.StringVar(value="选择一首歌曲；首次分离会下载模型，可先下载备用。")
        ttk.Label(state, textvariable=self.status, wraplength=780).grid(row=0, column=0, sticky="w")
        self.progress = ttk.Progressbar(state)
        self.progress.grid(row=1, column=0, sticky="ew", pady=4)

        workspace = ttk.Notebook(self)
        workspace.grid(row=3, column=0, sticky="nsew", pady=6)
        mixpage = ttk.Frame(workspace, padding=6)
        historypage = ttk.Frame(workspace, padding=6)
        workspace.add(mixpage, text="2. 试听 / 混音")
        workspace.add(historypage, text="历史结果")
        mixpage.columnconfigure(0, weight=1)
        mixpage.rowconfigure(2, weight=1)
        transport = ttk.Frame(mixpage)
        transport.grid(row=0, column=0, sticky="ew")
        ttk.Button(transport, text="播放", command=self.play).pack(side="left")
        ttk.Button(transport, text="暂停", command=self.player.pause).pack(side="left", padx=4)
        ttk.Button(transport, text="停止", command=self.player.stop).pack(side="left")
        self.time_label = tk.StringVar(value="00:00 / 00:00")
        ttk.Label(transport, textvariable=self.time_label).pack(side="right")
        self.seek = ttk.Scale(mixpage, from_=0, to=1, command=self.seek_to)
        self.seek.grid(row=1, column=0, sticky="ew", pady=4)
        self._moving_seek = False
        center = ttk.Panedwindow(mixpage, orient="vertical")
        center.grid(row=2, column=0, sticky="nsew")
        mixer_container = ttk.Frame(center)
        mixer_container.columnconfigure(0, weight=1)
        mixer_container.rowconfigure(0, weight=1)
        self.mixer_canvas = tk.Canvas(mixer_container, height=150, highlightthickness=0)
        self.mixer_canvas.grid(row=0, column=0, sticky="nsew")
        mixer_scroll = ttk.Scrollbar(mixer_container, orient="vertical", command=self.mixer_canvas.yview)
        mixer_scroll.grid(row=0, column=1, sticky="ns")
        self.mixer_canvas.config(yscrollcommand=mixer_scroll.set)
        self.mixer = ttk.Frame(self.mixer_canvas)
        mixer_window = self.mixer_canvas.create_window((0, 0), window=self.mixer, anchor="nw")
        self.mixer.bind("<Configure>", lambda _e: self.mixer_canvas.config(scrollregion=self.mixer_canvas.bbox("all")))
        self.mixer_canvas.bind("<Configure>", lambda e: self.mixer_canvas.itemconfigure(mixer_window, width=e.width))
        self.mixer_canvas.bind("<MouseWheel>", lambda e: self.mixer_canvas.yview_scroll(-int(e.delta / 120), "units"))
        plot_frame = ttk.Frame(center)
        center.add(mixer_container, weight=1)
        center.add(plot_frame, weight=2)
        self.figure = Figure(figsize=(7, 2.3), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("载入分离结果后显示音轨波形")
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        bottom = ttk.Frame(mixpage)
        bottom.grid(row=3, column=0, sticky="ew", pady=6)
        ttk.Button(bottom, text="听原曲", command=lambda: self.audition("original")).pack(side="left")
        ttk.Button(bottom, text="听人声", command=lambda: self.audition("vocals")).pack(side="left", padx=4)
        ttk.Button(bottom, text="听伴奏", command=lambda: self.audition("instrumental")).pack(side="left")
        ttk.Button(bottom, text="恢复分轨混音", command=self.default_mix).pack(side="left", padx=4)
        ttk.Button(bottom, text="导出音轨…", command=self.export_tracks).pack(side="right")
        ttk.Button(bottom, text="导出混音…", command=self.export_mix).pack(side="right", padx=4)
        self.limit = tk.BooleanVar(value=True)
        ttk.Checkbutton(mixpage, text="混音导出时防削顶（超出峰值时统一降低增益）", variable=self.limit).grid(row=4, column=0, sticky="w")

        historypage.columnconfigure(0, weight=1)
        historypage.rowconfigure(0, weight=1)
        self.history = ttk.Treeview(historypage, columns=("model", "length"), show="tree headings", height=6)
        self.history.heading("#0", text="歌曲 / 结果时间")
        self.history.heading("model", text="模型")
        self.history.heading("length", text="时长")
        self.history.column("#0", width=240)
        self.history.column("model", width=180)
        self.history.column("length", width=70)
        self.history.grid(row=0, column=0, sticky="nsew")
        buttons = ttk.Frame(historypage)
        buttons.grid(row=1, column=0, sticky="ew", pady=4)
        ttk.Button(buttons, text="载入所选结果", command=self.open_history).pack(side="left")
        ttk.Button(buttons, text="刷新", command=self.refresh_history).pack(side="left", padx=4)
        ttk.Button(buttons, text="打开结果清单…", command=self.open_manifest).pack(side="left")
        self.history.bind("<Double-1>", lambda _e: self.open_history())

    def refresh_model_status(self):
        model = PRESETS.get(self.model.get(), self.model.get())
        files = list(MODEL_DIR.rglob(model + ".*")) if MODEL_DIR.exists() else []
        ready = any(p.suffix in (".pth", ".th", ".ckpt") for p in files) and any(p.suffix == ".yaml" for p in files)
        self.model_info.set(f"模型：{model} · {'本地文件已存在' if ready else '首次使用需要下载'} · CPU 可用，耗时取决于歌曲和模型")

    def choose_input(self):
        if self.busy:
            messagebox.showinfo("任务处理中", "请先取消当前任务再更换歌曲。")
            return
        path = filedialog.askopenfilename(title="选择歌曲", filetypes=[("音频", "*.wav *.mp3 *.flac *.m4a *.ogg *.aiff"), ("全部", "*.*")])
        if path:
            self.path.set(path)

    def start(self, download_only=False):
        if self.busy:
            return
        if not download_only and not self.path.get():
            messagebox.showinfo("选择歌曲", "请先导入歌曲。")
            return
        self.player.stop()
        job = SeparationJob(self.events.put)
        self.job = job
        self.current_id = job.id
        self.set_busy(True)
        self.status.set("正在启动任务…")
        try:
            job.start(self.path.get(), PRESETS[self.model.get()], self.device.get(), self.source.get(), download_only)
        except Exception as exc:
            self.set_busy(False)
            messagebox.showerror("启动失败", str(exc))

    def set_busy(self, busy):
        self.busy = busy
        self.start_btn.config(state="disabled" if busy else "normal")
        self.download_btn.config(state="disabled" if busy else "normal")
        self.cancel_btn.config(state="normal" if busy else "disabled")
        self.model_combo.config(state="disabled" if busy else "readonly")
        if not busy:
            self.progress.stop()

    def cancel(self):
        if self.job:
            self.job.cancel()
            self.status.set("正在取消任务…")

    def handle_event(self, event):
        if self.closed or event.get("task_id") != self.current_id:
            return
        kind = event.get("type")
        if self.current_id in self._terminal and kind not in ("loaded", "load_error"):
            return
        if kind == "stage":
            self.progress.config(mode="indeterminate")
            self.progress.start(40)
            self.status.set(event["message"])
        elif kind == "progress":
            total = event.get("total", 0)
            if total > 0:
                self.progress.stop()
                self.progress.config(mode="determinate", value=min(100, event["done"] / total * 100))
            self.status.set(str(event.get("message", "处理中"))[:180])
        elif kind == "log":
            self.status.set(str(event.get("message", ""))[:180])
        elif kind in ("cancelled", "error", "load_error"):
            self._terminal.add(self.current_id)
            self.set_busy(False)
            self.progress.config(mode="determinate", value=0)
            self.status.set(event["message"])
            if kind in ("error", "load_error"):
                messagebox.showerror("音乐分离失败", event["message"] + "\n日志保存在本次任务目录。")
        elif kind == "done":
            self._terminal.add(self.current_id)
            self.refresh_model_status()
            if event.get("download_only"):
                self.set_busy(False)
                self.progress.config(mode="determinate", value=100)
                self.status.set("模型下载完成，可以开始分离。")
            else:
                self.load_result(event["result"], self.current_id)
        elif kind == "loaded":
            try:
                self.player.load(event["tracks"], event["result"]["fs"])
            except Exception as exc:
                self.handle_event({"type": "load_error", "task_id": self.current_id, "message": str(exc)})
                return
            self.result = event["result"]
            self.build_mixer()
            self.set_busy(False)
            self.progress.config(mode="determinate", value=100)
            self.status.set(f"已载入 {len(event['tracks']) - 1} 条输出音轨 · {self.result['fs']} Hz · {self.result['channels']} 声道")
            self.refresh_history()

    def load_result(self, result, task_id):
        def work():
            try:
                tracks = {}
                for name, path in result["files"].items():
                    raw, fs = audio_io.load_audio(path)
                    if fs != result["fs"]:
                        raise ValueError("音轨采样率与结果清单不一致")
                    tracks[name] = raw[:, None] if raw.ndim == 1 else raw
                    if tracks[name].shape != (result["samples"], result["channels"]):
                        raise ValueError("音轨长度或声道数与结果清单不一致")
                self.events.put({"type": "loaded", "task_id": task_id, "result": result, "tracks": tracks})
            except Exception as exc:
                self.events.put({"type": "load_error", "task_id": task_id, "message": str(exc)})
        threading.Thread(target=work, daemon=True).start()

    def build_mixer(self):
        for widget in self.mixer.winfo_children():
            widget.destroy()
        self.gain_vars, self.mute_vars, self.solo_vars = {}, {}, {}
        for row, name in enumerate(self.player.tracks):
            ttk.Label(self.mixer, text=STEM_LABELS.get(name, name), width=8).grid(row=row, column=0, sticky="w")
            var = tk.DoubleVar(value=self.player.gains[name])
            mute, solo = tk.BooleanVar(), tk.BooleanVar()
            self.gain_vars[name], self.mute_vars[name], self.solo_vars[name] = var, mute, solo
            ttk.Scale(self.mixer, from_=0, to=2, variable=var, command=lambda _v: self.update_gains()).grid(row=row, column=1, sticky="ew", padx=5)
            ttk.Checkbutton(self.mixer, text="静音", variable=mute, command=self.update_gains).grid(row=row, column=2)
            ttk.Checkbutton(self.mixer, text="独奏", variable=solo, command=self.update_gains).grid(row=row, column=3)
        self.mixer.columnconfigure(1, weight=1)
        length = len(next(iter(self.player.tracks.values()))) / self.player.fs
        self.seek.config(to=length)
        self.ax.clear()
        for name, audio in self.player.tracks.items():
            step = max(1, len(audio) // 3000)
            values = audio[::step].mean(axis=1)
            self.ax.plot(np.arange(len(values)) * step / self.player.fs, values, linewidth=0.65,
                         alpha=0.7, label=STEM_LABELS.get(name, name))
        self.ax.set_xlabel("时间（秒）")
        self.ax.set_ylabel("幅值")
        self.ax.legend(fontsize=8, loc="upper right")
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def update_gains(self):
        solos = {name for name, var in self.solo_vars.items() if var.get()}
        self.player.set_gains({name: var.get() if not self.mute_vars[name].get() and (not solos or name in solos) else 0
                               for name, var in self.gain_vars.items()})

    def audition(self, name):
        if name not in self.player.tracks:
            return
        for key in self.player.tracks:
            self.solo_vars[key].set(key == name)
            self.mute_vars[key].set(False)
            if key == name and self.gain_vars[key].get() == 0:
                self.gain_vars[key].set(1)
        self.update_gains()

    def default_mix(self):
        if not self.player.tracks:
            return
        for name in self.player.tracks:
            self.solo_vars[name].set(False)
            self.mute_vars[name].set(False)
            self.gain_vars[name].set(float(name not in ("original", "instrumental")))
        if set(self.player.tracks) == {"original", "vocals", "instrumental"}:
            self.gain_vars["instrumental"].set(1)
        self.update_gains()

    def play(self):
        try:
            self.before_play()
            self.player.play()
        except Exception as exc:
            messagebox.showerror("播放失败", str(exc))

    def seek_to(self, value):
        if not self._moving_seek:
            self.player.seek(value)

    def export_mix(self):
        if not self.player.tracks:
            return
        path = filedialog.asksaveasfilename(title="导出当前混音", defaultextension=".wav", initialfile="mix.wav", filetypes=[("WAV", "*.wav"), ("FLAC", "*.flac")])
        if not path:
            return
        try:
            import soundfile as sf
            audio = self.player.mix()
            peak = float(np.max(np.abs(audio)))
            gain = 0.99 / peak if self.limit.get() and peak > 0.99 else 1.0
            if not self.limit.get() and peak > 1 and Path(path).suffix.lower() == ".flac":
                raise ValueError("FLAC 超出满幅会削顶，请启用防削顶或降低音量")
            sf.write(path, audio * gain, self.player.fs, subtype="PCM_24" if Path(path).suffix.lower() == ".flac" else "FLOAT")
            metadata = {"source_result": self.result, "gains": self.player.gains, "export_gain": gain, "fs": self.player.fs}
            Path(path + ".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            self.status.set(f"混音已导出：{path} · 导出增益 {gain:.3f}")
        except Exception as exc:
            messagebox.showerror("导出失败", str(exc))

    def export_tracks(self):
        if not self.player.tracks:
            return
        directory = filedialog.askdirectory(title="选择音轨导出目录")
        if not directory:
            return
        try:
            import soundfile as sf
            output = Path(directory) / ("stems_" + time.strftime("%Y%m%d_%H%M%S") + "_" + str(time.time_ns() % 1000000))
            output.mkdir(exist_ok=False)
            for name, audio in self.player.tracks.items():
                sf.write(output / (name + ".wav"), audio, self.player.fs, subtype="FLOAT")
            (output / "result.json").write_text(json.dumps({**self.result, "files": {name: str(output / (name + ".wav")) for name in self.player.tracks}}, ensure_ascii=False, indent=2), encoding="utf-8")
            self.status.set(f"音轨已导出：{output}")
        except Exception as exc:
            messagebox.showerror("导出失败", str(exc))

    def refresh_history(self):
        for item in self.history.get_children():
            self.history.delete(item)
        if OUTPUT_DIR.exists():
            for path in sorted(OUTPUT_DIR.glob("*/result.json"), reverse=True)[:100]:
                try:
                    result = json.loads(path.read_text(encoding="utf-8"))
                    duration = result["samples"] / result["fs"]
                    self.history.insert("", "end", iid=str(path), text=Path(result["input"]).name + " · " + path.parent.name,
                                        values=(result["model"], f"{duration:.1f}s"))
                except (OSError, ValueError, KeyError):
                    continue

    def open_history(self):
        selection = self.history.selection()
        if selection:
            self.open_manifest(selection[0])

    def open_manifest(self, path=None):
        if self.busy:
            return
        path = path or filedialog.askopenfilename(title="打开分离结果", filetypes=[("结果清单", "*.json")])
        if not path:
            return
        try:
            result = json.loads(Path(path).read_text(encoding="utf-8"))
            self.current_id = "history_" + str(time.time_ns())
            self.set_busy(True)
            self.status.set("正在载入历史音轨…")
            self.load_result(result, self.current_id)
        except Exception as exc:
            self.set_busy(False)
            messagebox.showerror("载入失败", str(exc))

    @staticmethod
    def open_dir(directory):
        import os
        directory.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(directory)
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(directory)])

    def _tick(self):
        if self.closed:
            return
        for _ in range(100):
            try:
                self.handle_event(self.events.get_nowait())
            except queue.Empty:
                break
        if self.player.tracks:
            duration = len(next(iter(self.player.tracks.values()))) / self.player.fs
            position = self.player.position / self.player.fs
            self.time_label.set(f"{int(position)//60:02d}:{int(position)%60:02d} / {int(duration)//60:02d}:{int(duration)%60:02d}")
            self._moving_seek = True
            self.seek.set(position)
            self._moving_seek = False
        self._tick_id = self.after(100, self._tick)

    def close(self):
        self.closed = True
        self.after_cancel(self._tick_id)
        self.player.stop()
        if self.job:
            self.job.cancel()
