"""语音信号处理与机器学习降噪系统（tkinter + matplotlib 桌面应用）

功能：录音 / 导入 / 播放、端点检测、回声、倒放、时域与频域分析、滤波器设计演示，
以及 4 种降噪方案（GTCRN 预训练 / 自训练 MLP / 经典谱减法 / 串联融合），
可在界面上对同一段音频同台对比并选用。

运行（任意目录都可以，程序内部全部使用绝对路径）：
    python E:\\dsp\\dsp_py\\app.py
"""

from __future__ import annotations

import glob
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback

import numpy as np

# matplotlib 必须先用 TkAgg 后端，才能嵌进 tkinter
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "matplotlib"))
import matplotlib
matplotlib.use("TkAgg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

import tkinter as tk                                                    # noqa: E402
from tkinter import filedialog, messagebox, ttk                         # noqa: E402

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # noqa: E402
from matplotlib.figure import Figure                                    # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import audio_io, data as data_mod, dsp, metrics        # noqa: E402
from core.tasks import SignalTaskRunner
from core.corpus import default_clean_paths
from core import model as mlp
from core import room_effects
from music_panel import MusicPanel
from speaker_panel import SpeakerPanel
from core.enhance import Denoiser                                # noqa: E402
from core.gtcrn import GTCRNDenoiser                             # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
RESULT_DIR = os.path.join(HERE, "results")
RECORD_DIR = os.path.join(HERE, "recordings")          # 每次录音都归档到这里
NOISE_DIR = os.path.join(MODEL_DIR, "noise_bank")      # 合成噪声 + 真实噪声数据集


def _open_in_file_manager(path: str) -> bool:
    """在系统文件管理器里打开目录（跨平台：Windows / macOS / Linux）。"""
    import subprocess as _sp

    try:
        if sys.platform.startswith("win"):
            os.startfile(path)                          # noqa: S606 - Windows 资源管理器
        elif sys.platform == "darwin":
            _sp.Popen(["open", path])
        else:
            _sp.Popen(["xdg-open", path])
        return True
    except Exception:
        return False

# 配色（贴近 MATLAB 默认风格）
C_ORIG = "#0072BD"   # 原信号：蓝
C_DEN = "#D95319"    # 处理后：橙
C_HL = "#ED1C24"     # 高亮/语音段：红
C_FILT = "#77AC30"   # 滤波器响应：绿


class DSPApp:
    """主界面。"""

    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("语音信号处理与机器学习降噪系统 —— Python 版")
        # 默认窗口取足够高度，让左侧面板全部内容一次显示（实测需要约 920 px）；
        # 屏幕不够高时自动降级，此时左侧面板可滚动。
        try:
            sh = root.winfo_screenheight()
        except Exception:
            sh = 1080
        h = min(940, max(620, sh - 120))
        root.geometry(f"1330x{h}")
        root.minsize(980, 560)

        # ---- 信号状态（对应 MATLAB 里的 global y fs ...）----
        self.fs: int = 16000
        self.y: np.ndarray | None = None          # 当前信号（单声道 float32）
        self.y_raw: np.ndarray | None = None      # 导入/录制得到的原始信号
        self.yx: np.ndarray | None = None         # 降噪或房间音效后的信号（对应 yx）
        self.yr: np.ndarray | None = None         # 回声 / 倒放结果
        self.path: str = ""
        self.denoiser = Denoiser()
        self.gtcrn = GTCRNDenoiser()          # 预训练模型（onnxruntime，无需 PyTorch）
        self._training = False
        self._recording = False
        self._log_q = queue.Queue()
        self.tasks = SignalTaskRunner(self._log_q.put)
        self._closing = False
        self._train_proc = None
        self._train_cancel = threading.Event()
        self._train_lock = threading.Lock()
        self._choice_window = None
        self._processing_history = []
        self._notch_info = None
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._detail_windows: list[dict] = []      # 放大查看窗口（便于测试与管理）
        self._pan_state: dict | None = None        # 拖动平移的临时状态

        self._build_ui()
        self._refresh_model_status()
        self._autoload_sample()
        self._update_corpus_label()
        self._drain_after = self.root.after(150, self._drain_log)

    # =====================================================================
    # 界面搭建
    # =====================================================================
    def _build_ui(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("vista")
        except Exception:
            pass

        main = ttk.Frame(self.root, padding=6)
        main.pack(fill="both", expand=True)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        # ---------------- 左侧控制面板（可滚动，保证任何窗口高度下都能访问全部控件）----------------
        # 之前的做法：left 固定宽度直接 grid，内容比窗口高时底部控件会被裁掉（需手动拉伸窗口）。
        # 现在：canvas + 内嵌 frame + 竖向滚动条；内容不超过可视高度时自动隐藏滚动条。
        # 宽度依据实测：左侧内容（2 列按钮各 172px + 边框内边距）需要 368 px，
        # 所以画布给到 372 px，保证默认就能完整显示、不出现横向滚动条。
        left = ttk.Frame(main, width=380)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 8))
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        self.left_canvas = tk.Canvas(left, borderwidth=0, highlightthickness=0, width=372)
        self.left_canvas.grid(row=0, column=0, sticky="nsew")
        self.left_scroll = ttk.Scrollbar(left, orient="vertical", command=self.left_canvas.yview)
        self.left_scroll.grid(row=0, column=1, sticky="ns")
        self.left_hscroll = ttk.Scrollbar(left, orient="horizontal", command=self.left_canvas.xview)
        # 横向滚动条平时隐藏（内容装得下时不需要），窗口被压窄时才出现
        self.left_canvas.configure(yscrollcommand=self._on_left_scroll_set,
                                   xscrollcommand=self._on_left_xscroll_set)

        inner = ttk.Frame(self.left_canvas)
        self.left_inner = inner
        self._left_win = self.left_canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda _e: self.left_canvas.configure(
            scrollregion=self.left_canvas.bbox("all")))
        self.left_canvas.bind("<Configure>", self._on_left_canvas_resize)
        # 滚轮：全局监听，但只在指针位于左侧面板之上时滚动它，避免抢绘图区的滚轮
        self.root.bind_all("<MouseWheel>", self._on_global_wheel, add="+")

        # 以下所有面板按功能分组放进标签页：这样内容不再沿垂直方向堆积，
        # 小窗口也能完整访问（比"一个超长面板 + 滚动条"更好用）。
        tabs = ttk.Notebook(inner)
        self.left_tabs = tabs
        tabs.pack(fill="both", expand=True)
        page_proc = ttk.Frame(tabs, padding=6)      # 处理与展示
        page_files = ttk.Frame(tabs, padding=6)     # 录音 / 音频 / 语料
        page_model = ttk.Frame(tabs, padding=6)     # 机器学习与模型
        page_effects = ttk.Frame(tabs, padding=6)
        tabs.add(page_proc, text="处理")
        tabs.add(page_files, text="数据")
        tabs.add(page_model, text="模型")
        tabs.add(page_effects, text="音效")

        source_box = ttk.LabelFrame(page_effects, text="处理来源", padding=6)
        source_box.pack(fill="x", pady=(0, 6))
        self.var_effect_source = tk.StringVar(value="原信号")
        ttk.Combobox(source_box, textvariable=self.var_effect_source,
                     values=("原信号", "当前处理结果"), state="readonly", width=20).pack(fill="x")
        ttk.Label(source_box, text="使用语音工作区的单声道信号。\n可先加混响，再对处理结果去混响。",
                  wraplength=260).pack(anchor="w", pady=(4, 0))
        reverb_box = ttk.LabelFrame(page_effects, text="房间混响", padding=6)
        reverb_box.pack(fill="x", pady=(0, 6))
        self.var_reverb_rt60 = tk.StringVar(value="1.0")
        self.var_reverb_wet = tk.StringVar(value="30")
        self.var_reverb_delay = tk.StringVar(value="20")
        self.var_reverb_damping = tk.StringVar(value="50")
        for row, (label, var) in enumerate((("衰减时间 RT60 (秒)", self.var_reverb_rt60),
                                            ("湿声比例 (%)", self.var_reverb_wet),
                                            ("预延迟 (毫秒)", self.var_reverb_delay),
                                            ("高频衰减 (%)", self.var_reverb_damping))):
            ttk.Label(reverb_box, text=label).grid(row=row, column=0, sticky="w", pady=2)
            ttk.Entry(reverb_box, textvariable=var, width=8).grid(row=row, column=1, padx=4)
        self.btn_reverb = ttk.Button(reverb_box, text="添加混响", command=self.on_reverb)
        self.btn_reverb.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        dereverb_box = ttk.LabelFrame(page_effects, text="去混响 · WPE", padding=6)
        dereverb_box.pack(fill="x", pady=(0, 6))
        self.var_dereverb_strength = tk.StringVar(value="70")
        ttk.Label(dereverb_box, text="去混响强度 (%)").grid(row=0, column=0, sticky="w")
        ttk.Entry(dereverb_box, textvariable=self.var_dereverb_strength, width=8).grid(row=0, column=1, padx=4)
        self.btn_dereverb = ttk.Button(dereverb_box, text="去混响", command=self.on_dereverb)
        self.btn_dereverb.grid(row=1, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Label(dereverb_box, text="减弱晚期拖尾；强度过高可能损伤人声。\n静音及不足 0.4 秒的素材原样返回。",
                  wraplength=260).grid(row=2, column=0, columnspan=2, sticky="w")
        actions = ttk.Frame(page_effects)
        actions.pack(fill="x")
        ttk.Button(actions, text="试听原信号", command=lambda: self._play(self.y, "原信号") if self.need_signal() else None).pack(fill="x", pady=2)
        ttk.Button(actions, text="试听处理结果", command=lambda: self._play(self.yx, "处理结果") if self.need_denoised() else None).pack(fill="x", pady=2)
        ttk.Button(actions, text="导出处理结果…", command=self.on_export).pack(fill="x", pady=2)

        # 录音参数（对应 MATLAB 里的 Fs/duration/nBits/nChannels）
        box = ttk.LabelFrame(page_files, text="录音参数", padding=6)
        box.pack(fill="x", pady=(0, 6))
        self.var_fs = tk.StringVar(value="16000")
        self.var_dur = tk.StringVar(value="4")
        self.var_ch = tk.StringVar(value="2")   # 与原 MATLAB 的 nChannels = 2 一致
        for r, (lab, var) in enumerate((("采样率 Fs (Hz)", self.var_fs),
                                        ("录音时长 (s)", self.var_dur),
                                        ("声道数", self.var_ch))):
            ttk.Label(box, text=lab).grid(row=r, column=0, sticky="w", pady=1)
            if lab.startswith("声道"):
                ttk.Combobox(box, textvariable=var, values=("1", "2"), width=8,
                             state="readonly").grid(row=r, column=1, sticky="e")
            else:
                ttk.Entry(box, textvariable=var, width=9).grid(row=r, column=1, sticky="e")

        # 按钮区（对应 sjjg.fig 的 pushbutton1~11）—— 「处理」标签页
        btns = ttk.LabelFrame(page_proc, text="功能按钮", padding=6)
        btns.pack(fill="x", pady=(0, 6))
        specs = [
            ("① 录音", self.on_record),
            ("② 导入音频", self.on_import),
            ("③ 播放", self.on_play),
            ("④ 回声", self.on_echo),
            ("⑤ GTCRN 预训练降噪 ★", self.on_denoise_gtcrn),
            ("⑥ 自训练 MLP 降噪", self.on_denoise_ml),
            ("⑦ 经典谱减法（对比）", self.on_denoise_classic),
            ("⑧ 多方案降噪对比", self.on_compare_denoisers),
            ("⑨ 端点检测", self.on_endpoint),
            ("⑩ 倒放", self.on_reverse),
            ("⑪ 原信号频谱", self.on_spectrum_before),
            ("⑫ 处理后频谱", self.on_spectrum_after),
            ("⑬ 原信号时域", self.on_time_before),
            ("⑭ 处理后时域", self.on_time_after),
        ]
        self.buttons: dict[str, ttk.Button] = {}
        for i, (text, cmd) in enumerate(specs):
            b = ttk.Button(btns, text=text, command=cmd, width=22)
            b.grid(row=i // 2, column=i % 2, padx=2, pady=2, sticky="ew")
            self.buttons[text] = b

        # 机器学习面板 —— 「模型」标签页
        ml = ttk.LabelFrame(page_model, text="机器学习降噪", padding=6)
        ml.pack(fill="x", pady=(0, 6))
        self.lbl_gtcrn = ttk.Label(ml, text="预训练模型：检测中…", wraplength=210, justify="left")
        self.lbl_gtcrn.pack(anchor="w")
        self.lbl_model = ttk.Label(ml, text="自训练模型：检测中…", wraplength=210, justify="left")
        self.lbl_model.pack(anchor="w", pady=(4, 0))
        ttk.Separator(ml, orient="horizontal").pack(fill="x", pady=4)
        ttk.Button(ml, text="训练降噪模型（保存新版本）", command=self.on_train_model).pack(fill="x", pady=(0, 2))
        ttk.Button(ml, text="选择模型版本 / 回退…", command=self.on_choose_model).pack(fill="x", pady=(0, 2))
        ttk.Button(ml, text="取消训练", command=self.on_cancel_train).pack(fill="x", pady=(0, 2))
        row = ttk.Frame(ml)
        row.pack(fill="x")
        ttk.Button(row, text="重新加载模型", command=self.on_reload_model).pack(side="left", expand=True, fill="x")
        ttk.Button(row, text="导出降噪 wav", command=self.on_export).pack(side="left", expand=True, fill="x")

        # 训练数据面板 —— 「数据」标签页
        tr = ttk.LabelFrame(page_files, text="训练数据", padding=6)
        tr.pack(fill="x", pady=(0, 6))
        self.lbl_corpus = ttk.Label(tr, text="语料统计中…", wraplength=210, justify="left")
        self.lbl_corpus.pack(anchor="w")
        ttk.Button(tr, text="导入干净语音…", command=self.on_import_speech).pack(fill="x", pady=(4, 2))
        ttk.Button(tr, text="批量导入文件夹…（推荐）",
                   command=self.on_import_folder).pack(fill="x", pady=(0, 2))
        row2 = ttk.Frame(tr)
        row2.pack(fill="x")
        ttk.Button(row2, text="导入真实噪声…", command=self.on_import_noise).pack(side="left", expand=True, fill="x")
        ttk.Button(row2, text="打开数据目录", command=self.on_open_data_dir).pack(side="left", expand=True, fill="x")

        # 手动参数供经典谱减法使用；默认自动检测持续窄带干扰。
        fb = ttk.LabelFrame(page_proc, text="谱减法带阻 / 陷波设置", padding=6)
        fb.pack(fill="x", pady=(0, 6))
        self.var_fp1 = tk.StringVar(value="1500")
        self.var_fp2 = tk.StringVar(value="2100")
        self.var_fs1 = tk.StringVar(value="1750")
        self.var_fs2 = tk.StringVar(value="1850")
        for r, (lab, var) in enumerate((("通带下限 fp1", self.var_fp1), ("通带上限 fp2", self.var_fp2),
                                        ("阻带下限 fs1", self.var_fs1), ("阻带上限 fs2", self.var_fs2))):
            ttk.Label(fb, text=lab).grid(row=r // 2, column=(r % 2) * 2, sticky="w", pady=1)
            ttk.Entry(fb, textvariable=var, width=7).grid(row=r // 2, column=(r % 2) * 2 + 1, sticky="e")
        self.var_auto_notch = tk.BooleanVar(value=True)
        ttk.Checkbutton(fb, text="谱减法自动陷波（推荐）", variable=self.var_auto_notch,
                        command=self._on_auto_notch_toggle).grid(row=2, column=0, columnspan=4, sticky="w", pady=4)
        self.btn_detect_notch = ttk.Button(fb, text="检测并填入带阻频率", command=self._fill_notch_from_signal)
        self.btn_detect_notch.grid(row=3, column=0, columnspan=4, sticky="ew")
        self.lbl_notch_hint = ttk.Label(fb, wraplength=290, justify="left")
        self.lbl_notch_hint.grid(row=4, column=0, columnspan=4, sticky="w", pady=(4, 0))
        self._on_auto_notch_toggle()

        # ---------------- 右侧绘图区 ----------------
        self.work_tabs = ttk.Notebook(main)
        self.work_tabs.grid(row=0, column=1, sticky="nsew")
        right = ttk.Frame(self.work_tabs)
        self.work_tabs.add(right, text="语音处理 / 信号分析")
        self.music_panel = MusicPanel(self.work_tabs, before_play=audio_io.stop_play)
        self.work_tabs.add(self.music_panel, text="音乐分离 / 分轨混音")
        self.speaker_panel = SpeakerPanel(self.work_tabs,
            before_play=lambda: (self.music_panel.player.pause(), audio_io.stop_play()))
        self.work_tabs.add(self.speaker_panel, text="双人语音分离")
        # 协调输出设备；音乐算法、模型选择及导出模块保持原样。
        self.music_panel.before_play = lambda: (self.speaker_panel.player.pause(), audio_io.stop_play())
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        self.fig = Figure(figsize=(8.4, 6.4), dpi=100)
        gs = self.fig.add_gridspec(2, 2, hspace=0.34, wspace=0.24)
        self.ax1 = self.fig.add_subplot(gs[0, 0])   # 对应 axes1
        self.ax2 = self.fig.add_subplot(gs[0, 1])   # 对应 axes2
        self.ax3 = self.fig.add_subplot(gs[1, 0])   # 对应 axes3
        self.ax4 = self.fig.add_subplot(gs[1, 1])   # 新增：谱图 / 对比
        for ax in (self.ax1, self.ax2, self.ax3, self.ax4):
            ax.grid(True, alpha=0.3)
        self._set_titles()
        self.fig.set_layout_engine("constrained")

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        toolbar = NavigationToolbar2Tk(self.canvas, right, pack_toolbar=False)
        toolbar.update()
        toolbar.grid(row=1, column=0, sticky="ew")

        # 点击任意子图 -> 弹出放大窗口；滚轮以光标为中心缩放（用工具栏的平移/缩放时不会误触发）
        self._bind_plot_interactions(self.canvas)
        ttk.Label(right, text="提示：单击子图可弹出放大窗口；滚轮以光标为中心缩放；"
                              "也可用下方工具栏框选/平移/保存图片",
                  foreground="#666666").grid(row=2, column=0, sticky="w", pady=(2, 0))

        # ---------------- 底部状态栏 ----------------
        bar = ttk.Frame(self.root, padding=(8, 3))
        bar.pack(fill="x", side="bottom")
        self.var_status = tk.StringVar(value="就绪：可点击「导入音频」或「录音」开始")
        ttk.Label(bar, textvariable=self.var_status, anchor="w").pack(side="left", fill="x", expand=True)
        self.progress = ttk.Progressbar(bar, mode="determinate", length=140)
        self.progress.pack(side="right")

    def _on_left_scroll_set(self, first: str, last: str) -> None:
        """内容装得下时自动隐藏滚动条（更清爽），装不下才显示。"""
        self.left_scroll.set(first, last)
        try:
            need = float(last) - float(first) < 0.999
        except ValueError:
            need = True
        if need:
            self.left_scroll.grid()
        else:
            self.left_scroll.grid_remove()

    def _on_left_xscroll_set(self, first: str, last: str) -> None:
        """横向滚动条：仅在内容宽度超出可视宽度时出现（窗口被压窄的情况）。"""
        self.left_hscroll.set(first, last)
        try:
            need = float(last) - float(first) < 0.999
        except ValueError:
            need = True
        if need:
            # 跨两列，避免与竖向滚动条在右下角重叠
            self.left_hscroll.grid(row=1, column=0, columnspan=2, sticky="ew")
        else:
            self.left_hscroll.grid_remove()

    def _on_left_canvas_resize(self, event) -> None:
        """内嵌 frame 宽度跟随画布（不小于内容请求宽度），并刷新滚动范围。"""
        need = max(event.width, self.left_inner.winfo_reqwidth())
        self.left_canvas.itemconfigure(self._left_win, width=need)
        self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all"))

    def _bind_left_wheel(self, enable: bool) -> None:
        """（保留兼容）显式启用/停用左侧面板滚轮。现在用 _on_global_wheel 自动判断。"""
        return

    def _on_global_wheel(self, event) -> None:
        """全局滚轮：指针在左侧面板上时滚动面板，否则交给绘图区（matplotlib 自己处理）。

        之前试图用 <Enter>/<Leave> 动态绑定，但测试与真实使用中指针未进入时就不生效；
        改为按"指针坐标是否落在面板内"判断，无需依赖 Enter 事件。
        """
        try:
            x, y = self.root.winfo_pointerxy()
            cx0, cy0 = self.left_canvas.winfo_rootx(), self.left_canvas.winfo_rooty()
            inside = (cx0 <= x <= cx0 + self.left_canvas.winfo_width()
                      and cy0 <= y <= cy0 + self.left_canvas.winfo_height())
            if not inside:
                return
            if self.left_canvas.bbox("all") is None:
                return
            top, bottom = self.left_canvas.yview()
            if top <= 0.0 and bottom >= 1.0:
                # 纵向装得下时，Shift+滚轮改为横向滚动（可选便利功能）
                if event.state & 0x0001:
                    self.left_canvas.xview_scroll(-1 if event.delta > 0 else 1, "units")
                return
            if event.state & 0x0001:             # Shift+滚轮 = 横向
                self.left_canvas.xview_scroll(-1 if event.delta > 0 else 1, "units")
            else:
                self.left_canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
        except Exception:
            pass

    def _set_titles(self) -> None:
        self.ax1.set_title("时域波形（原信号）")
        self.ax2.set_title("时域波形（处理后）")
        self.ax3.set_title("带阻滤波器幅频响应")
        self.ax4.set_title("对数谱图")
        for ax in (self.ax1, self.ax2, self.ax3, self.ax4):
            ax.set_xlabel("频率 (Hz)" if ax is self.ax3 else "采样点")
            ax.set_ylabel("幅值")

    # =====================================================================
    # 图形交互：滚轮缩放 + 点击放大
    # =====================================================================
    def _bind_plot_interactions(self, canvas: FigureCanvasTkAgg, allow_enlarge: bool = True) -> None:
        """给一个 matplotlib 画布挂上交互。

        · **滚轮**：以光标为中心缩放（TkAgg 默认不支持，这里补上）；
        · **按住左键拖动**：平移视图（等价于工具栏的"手型"模式，但无需先切模式）；
        · **左键单击**（位移 <4 px）：弹出放大窗口——仅主界面画布启用（allow_enlarge）。
          放大窗口里**不再**弹新窗口：否则在放大图上再点一下就会"窗口套窗口"，
          越点越多（实测就是这个原因）。

        滚轮步长 1.25×，离散 13 档（单轴约 4.0×~0.26×）。
        """
        canvas.mpl_connect("button_press_event",
                           lambda e: self._on_plot_click(e, allow_enlarge))
        canvas.mpl_connect("scroll_event", self._on_plot_scroll)
        canvas.mpl_connect("motion_notify_event", self._on_plot_drag)
        canvas.mpl_connect("button_release_event", self._on_plot_release)
        canvas.get_tk_widget().bind(
            "<MouseWheel>", lambda e, c=canvas: self._on_wheel_event(c, e))
        canvas.get_tk_widget().configure(cursor="fleur")

    # ---- 拖动平移（按下左键拖动；无需先切换工具栏的"手型"模式）----
    def _on_plot_drag(self, event) -> None:
        """matplotlib 的 motion 事件：按住左键拖动即平移视图。"""
        st = getattr(self, "_pan_state", None)
        if st is None or event.button != 1:
            return
        ax = st.get("ax")
        if ax is None:
            return
        dx = event.x - st["x"]
        dy = event.y - st["y"]                  # matplotlib 显示坐标，与按下时同一坐标系
        if abs(event.x - st["origin"][0]) + abs(event.y - st["origin"][1]) > 3:
            st["moved"] = True
        st["x"], st["y"] = event.x, event.y
        if not st["moved"] or (dx == 0 and dy == 0):
            return
        try:
            inv = ax.transData.inverted()
            x0, y0 = inv.transform((0, 0))
            x1, y1 = inv.transform((dx, dy))
            sx, sy = x1 - x0, y1 - y0
            (xa, xb), (ya, yb) = ax.get_xlim(), ax.get_ylim()
            ax.set_xlim(xa - sx, xb - sx)
            ax.set_ylim(ya - sy, yb - sy)
            event.canvas.draw_idle()
        except Exception:
            pass

    def _on_plot_release(self, event) -> None:
        st = getattr(self, "_pan_state", None)
        if st is not None and st.get("moved"):
            self.status("已平移视图（滚轮缩放；主界面单击可放大）")
        self._pan_state = None

    @staticmethod
    def _axes_at(canvas: FigureCanvasTkAgg, x: int, y_tk: int):
        """根据 Tk 坐标找出所在子图（注意 y 需要翻转）。"""
        y = canvas.get_tk_widget().winfo_height() - y_tk
        for a in canvas.figure.axes:
            bb = a.get_window_extent()
            if bb.x0 <= x <= bb.x1 and bb.y0 <= y <= bb.y1:
                return a
        return None

    def _on_wheel_event(self, canvas: FigureCanvasTkAgg, event) -> None:
        """把 Tk 的滚轮事件转成缩放手势（Windows 下 delta=±120）。

        关键坑：**Tk 的事件坐标原点在左上角，而 matplotlib 的显示坐标原点在左下角**，
        所以必须把 y 翻转为 height - y，否则垂直方向的缩放中心会错位（表现为"滚轮放大
        反而让范围变宽"）。
        """
        try:
            x, y_tk = event.x, event.y
            y = canvas.get_tk_widget().winfo_height() - y_tk
            ax = None
            for a in canvas.figure.axes:
                bbox = a.get_window_extent()
                if bbox.x0 <= x <= bbox.x1 and bbox.y0 <= y <= bbox.y1:
                    ax = a
                    break
            if ax is None:
                return
            self._zoom_axes(ax, 0.8 if event.delta > 0 else 1.25, x, y)
            canvas.draw_idle()
        except Exception:
            pass

    def _on_plot_scroll(self, event) -> None:
        """支持 matplotlib 原生 scroll_event（部分后端会直接派发）。"""
        if event.inaxes is None:
            return
        step = 0.8 if getattr(event, "button", None) == "up" else 1.25
        self._zoom_axes(event.inaxes, step, event.x, event.y)
        event.canvas.draw_idle()

    @staticmethod
    def _zoom_axes(ax, factor: float, x: float, y: float) -> None:
        """以像素点 (x, y) 为中心缩放坐标轴。

        factor 是**可视范围的比例因子**：<1 表示放大（范围变窄、看细节），>1 表示缩小。
        （坐标 (x, y) 用 matplotlib 的显示坐标：原点在左下角！）

        实现用的是**离散缩放档位**（可视范围 = 基准跨度 × 1.25^k，k ∈ [−6, +6]，
        即 4.0×~0.26× 单轴范围、累计约 1000× 放大到 0.01× 缩小）：
        连续乘因子会因限幅边界附近的取整抖动，反复滚动时跨度可能趋近 0
        （坐标轴退化成空图），离散档位既能稳定推进又不会退化。
        """
        if not hasattr(ax, "_zoom_base_x"):
            ax._zoom_base_x = abs(ax.get_xlim()[1] - ax.get_xlim()[0])
            ax._zoom_base_y = abs(ax.get_ylim()[1] - ax.get_ylim()[0])
            ax._zoom_step = 0.0
        # 档位：factor<1（放大）让 k 减小，factor>1（缩小）让 k 增大
        step = 1.0 if factor >= 1 else -1.0
        ax._zoom_step = float(np.clip(ax._zoom_step + step, -6, 6))
        scale = 1.25 ** ax._zoom_step

        inv = ax.transData.inverted()
        try:
            cx, cy = inv.transform((x, y))
        except Exception:
            return
        for get_lim, set_lim, center, base in (
                (ax.get_xlim, ax.set_xlim, cx, ax._zoom_base_x),
                (ax.get_ylim, ax.set_ylim, cy, ax._zoom_base_y)):
            if base <= 0:
                continue
            lo, hi = get_lim()
            span = abs(hi - lo)
            if span <= 0 or abs(hi - lo) > base * 100:
                lo, hi = center - base / 2, center + base / 2   # 复位后再缩放
            # 光标下方的数据点保持不动
            rel = (center - lo) / (hi - lo)
            half = base * scale / 2
            set_lim(center - rel * 2 * half, center + (1 - rel) * 2 * half)

    def _on_plot_click(self, event, allow_enlarge: bool = True) -> None:
        """左键按下：先记录拖动起点；若是"单击"（位移很小）且允许，则弹出放大窗口。

        allow_enlarge=False（放大窗口内部）时**不弹新窗口**——否则在放大图上再点一下
        就会窗口套窗口、越点越多。
        """
        if event.button != 1 or event.inaxes is None:
            return
        self._pan_state = {"ax": event.inaxes, "x": event.x, "y": event.y,
                           "origin": (event.x, event.y), "moved": False}
        if not allow_enlarge:
            return
        mode = getattr(self.canvas, "toolbar", None)
        if mode is not None and getattr(mode, "mode", "") not in ("", None):
            self._pan_state = None
            return                        # 正在用工具栏的平移/缩放，交给工具栏处理
        self._enlarge_axes(event.inaxes)
        self._pan_state = None            # 已开窗，本次按下不再作为拖动起点

    def _new_detail_window(self, title: str, size: str = "1080x700") -> tk.Toplevel:
        """新建一个置顶的详情窗口。

        踩坑记录：Toplevel 默认可能出现在主窗口**后面**（用户以为"没反应"）。
        这里统一做三件事：transient(主窗口) 让它跟随主窗口、lift() 提到最前、
        attributes('-topmost') 短暂置顶后取消（否则会一直压住别的程序）。
        """
        top = tk.Toplevel(self.root)
        top.title(title)
        top.geometry(size)
        top.transient(self.root)
        top.lift()
        try:
            top.attributes("-topmost", True)
            top.after(400, lambda: top.attributes("-topmost", False))
        except Exception:
            pass
        top.focus_force()
        top.protocol("WM_DELETE_WINDOW", lambda: (self._detail_windows.remove(
            next(d for d in self._detail_windows if d["window"] is top)), top.destroy())
            if any(d["window"] is top for d in self._detail_windows) else top.destroy())
        return top

    def _enlarge_axes(self, ax) -> None:
        """把指定子图复制到一个大窗口里，保留标题/标签/图例/对数轴与当前视图范围。"""
        title = ax.get_title() or "图形详情"
        top = self._new_detail_window(f"放大查看 —— {title}")

        big = Figure(figsize=(10.4, 6.4), dpi=100)
        bax = big.add_subplot(111)
        n_copied = 0
        for line in ax.get_lines():
            bax.plot(line.get_xdata(), line.get_ydata(), color=line.get_color(),
                     linewidth=max(line.get_linewidth(), 1.0), label=line.get_label(),
                     linestyle=line.get_linestyle(), marker=line.get_marker(),
                     alpha=line.get_alpha() if line.get_alpha() is not None else 1.0)
            n_copied += 1
        for coll in ax.collections:        # stem 图（茎叶图）走 collection
            try:
                offs = coll.get_offsets()
                bax.stem(np.asarray(offs)[:, 0], np.asarray(offs)[:, 1],
                         markerfmt=" ", basefmt=" ", linefmt=(coll.get_color() or "C0") + "-")
                n_copied += 1
            except Exception:
                pass
        for patch in ax.patches:           # 柱状/填充类
            try:
                bax.add_patch(patch)
                n_copied += 1
            except Exception:
                pass

        bax.set_xlabel(ax.get_xlabel()); bax.set_ylabel(ax.get_ylabel())
        if ax.get_yscale() == "log":
            bax.set_yscale("log")
        if ax.get_xlim() != (0.0, 1.0) or ax.get_ylim() != (0.0, 1.0):
            bax.set_xlim(*ax.get_xlim()); bax.set_ylim(*ax.get_ylim())
        bax.grid(True, alpha=0.3)
        if ax.get_legend_handles_labels()[1]:
            bax.legend(loc="upper right", fontsize=9)
        bax.set_title(f"{title}", fontsize=12)
        big.tight_layout()

        ttk.Label(top, text="操作提示：滚轮 = 以光标为中心缩放；按住左键拖动 = 平移；"
                            "下方工具栏可框选缩放/复位/保存为 PNG",
                  foreground="#666666").pack(fill="x", padx=6)

        canvas = FigureCanvasTkAgg(big, master=top)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        bar = NavigationToolbar2Tk(canvas, top, pack_toolbar=False)
        bar.update()
        bar.pack(fill="x")
        canvas.draw_idle()
        # 说明：放大窗口内**不再**允许继续弹新窗口，避免"窗口套窗口"
        self._bind_plot_interactions(canvas, allow_enlarge=False)
        self._detail_windows.append({"window": top, "figure": big, "title": title,
                                     "n_curves": n_copied})
        self.status(f"已放大显示：{title}（复制了 {n_copied} 条曲线，可用滚轮缩放）")

    # =====================================================================
    # 降噪方案选择对话框
    # =====================================================================
    def _ask_method_choice(self, report: str, results: dict[str, np.ndarray],
                           metrics_by_name: dict[str, dict]) -> str | None:
        """让用户从各降噪方案里**单选**一个作为输出（而不是默认取第一个）。

        返回选中的方案名；取消则返回 None。
        """
        dialog_fs = self.fs
        names = [k for k in results if k != "原信号"]
        if not names:
            return None
        # 注意：这里**不用** grab_set（模态会锁住主窗口的点击，导致无法点图放大）。
        # 改为非模态 + 自动靠右摆放，用户可以在对比的同时操作主界面的图。
        top = tk.Toplevel(self.root)
        self._choice_window = top
        top.title("选择要采用的降噪结果（本窗口不阻挡主界面操作）")
        top.geometry("780x540")
        top.transient(self.root)
        try:
            self.root.update_idletasks()
            rx, ry = self.root.winfo_rootx(), self.root.winfo_rooty()
            rw = self.root.winfo_width()
            sw = top.winfo_screenwidth()
            x = min(rx + rw + 12, max(sw - 800, 0))      # 优先摆在主窗口右侧
            top.geometry(f"780x540+{max(x, 0)}+{ry + 40}")
        except Exception:
            pass
        top.lift()

        ttk.Label(top, text="各方案指标对比（均为无参考指标）", font=("", 11, "bold")).pack(anchor="w", padx=12, pady=(10, 4))

        # 指标表
        table = ttk.Frame(top)
        table.pack(fill="x", padx=12)
        cols = ("方案", "能量对比改善", "低能量帧下降", "高能量帧保留", "频谱改变量")
        for c, head in enumerate(cols):
            ttk.Label(table, text=head, font=("", 9, "bold"), width=14 if c == 0 else 13,
                      anchor="center").grid(row=0, column=c, padx=2, pady=2)
        for r, name in enumerate(names, start=1):
            m = metrics_by_name.get(name, {})
            cells = (name,
                     (f"{m['snr_est_improve_db']:+.1f} dB" if m.get("snr_est_improve_db") is not None else "无法可靠估计"),
                     f"{m.get('noise_floor_drop_db', float('nan')):.1f} dB",
                     f"{m.get('speech_keep_db', float('nan')):+.1f} dB",
                     f"{m.get('lsd_self_db', float('nan')):.1f} dB")
            for c, txt in enumerate(cells):
                ttk.Label(table, text=txt, width=14 if c == 0 else 13,
                          anchor="center").grid(row=r, column=c, padx=2, pady=1)
        ttk.Label(top, text="（能量对比为近似值；低能量帧不一定是静音；"
                           "高能量帧保留不代表语音质量；频谱改变量不能当作人声失真）",
                  foreground="#666666", wraplength=720, justify="left").pack(anchor="w", padx=12, pady=(4, 8))

        # 选择 + 试听
        choice = tk.StringVar(value=names[0])
        pick = ttk.LabelFrame(top, text="选择要采用的方案（单选）", padding=8)
        pick.pack(fill="x", padx=12)
        for name in names:
            row = ttk.Frame(pick)
            row.pack(fill="x", pady=2)
            ttk.Radiobutton(row, text=name, value=name, variable=choice, width=24).pack(side="left")
            ttk.Button(row, text="试听", width=6,
                       command=lambda n=name: self._play(results[n], n, fs=dialog_fs)).pack(side="left", padx=4)
            ttk.Button(row, text="放大看频谱", width=12,
                       command=lambda n=name: self._plot_single_method_spectrum(results, n)).pack(side="left")

        ttk.Button(top, text="停止播放", command=audio_io.stop_play).pack(anchor="w", padx=12, pady=(6, 0))
        ttk.Button(top, text="放大查看频谱（全部方案叠加）", width=28,
                   command=self._show_last_compare_enlarged).pack(anchor="w", padx=12, pady=(4, 0))
        self._last_compare = {"results": results, "metrics": metrics_by_name}

        # 详情文本
        box = tk.Text(top, height=8, wrap="word")
        box.pack(fill="both", expand=True, padx=12, pady=(8, 4))
        box.insert("1.0", report)
        box.config(state="disabled")

        result: dict[str, str | None] = {"name": None}

        def _ok() -> None:
            result["name"] = choice.get()
            top.destroy()

        btns = ttk.Frame(top)
        btns.pack(fill="x", padx=12, pady=(0, 10))
        ttk.Button(btns, text="采用所选方案", command=_ok).pack(side="right", padx=4)
        ttk.Button(btns, text="取消（保持当前结果）", command=top.destroy).pack(side="right", padx=4)

        top.wait_window()
        return result["name"]

    def _plot_single_method_spectrum(self, results: dict[str, np.ndarray],
                                     name: str | None = None) -> None:
        """放大查看频谱及波形，并保留原信号作为对比。

        name 为空则显示全部方案（对比功能里的「放大查看频谱」按钮）；
        指定 name 则只显示该方案（方案列表里的「放大看频谱」按钮）。
        窗口支持滚轮缩放和拖动平移。
        """
        methods = [k for k in results if k != "原信号"]
        if name is not None:
            if name not in methods:
                self.status(f"无法查看频谱：方案 {name} 不在本次对比结果中")
                return
            methods = [name]
        title = f"频谱与时域对比 —— {name}" if name else "频谱与时域对比 —— 全部方案"
        top = self._new_detail_window(title, "1180x780")

        fig = Figure(figsize=(11.4, 7.0), dpi=100)
        ax_s = fig.add_subplot(211)
        ax_t = fig.add_subplot(212)
        palette = ["#0072BD", "#D95319", "#77AC30", "#7E2F8E", "#EDB120", "#4DBEEE"]

        # --- 上：频谱对比（对数纵轴，便于看噪声底） ---
        sp_c = dsp.spectrum(self.y, self.fs)
        ax_s.plot(sp_c["freqs"], sp_c["mag"], color="#9AA0A6", linewidth=0.8, label="原信号")
        for i, m in enumerate(methods):
            sig = results[m]
            sp = dsp.spectrum(sig[:len(self.y)] if len(sig) >= len(self.y) else sig, self.fs)
            highlight = (m == name)
            ax_s.plot(sp["freqs"], sp["mag"],
                      color=palette[i % len(palette)],
                      linewidth=1.3 if highlight else 0.8,
                      alpha=1.0 if highlight else 0.85,
                      label=m + ("（当前查看）" if highlight else ""))
        ax_s.set_yscale("log")
        ax_s.set_xlim(0, self.fs / 4)
        spectrum_title = f"{name} 与原信号的频谱对比" if name is not None else "全部方案频谱对比"
        ax_s.set_title(spectrum_title + "（对数纵轴）", fontsize=12)
        ax_s.set_xlabel("频率 (Hz)"); ax_s.set_ylabel("幅值")
        ax_s.grid(True, alpha=0.3); ax_s.legend(loc="upper right", fontsize=9)

        # --- 下：时域波形（只看前若干秒，避免挤在一起） ---
        seg = min(len(self.y), self.fs * 2)
        ax_t.plot(np.arange(seg) / self.fs, self.y[:seg], color="#9AA0A6",
                  linewidth=0.7, label="原信号")
        for i, m in enumerate(methods):
            sig = results[m][:len(self.y)]
            ax_t.plot(np.arange(seg) / self.fs, sig[:seg],
                      color=palette[i % len(palette)], linewidth=1.2 if m == name else 0.7,
                      alpha=1.0 if m == name else 0.8, label=m)
        ax_t.set_title(f"时域波形对比（前 {seg / self.fs:.1f} 秒）", fontsize=12)
        ax_t.set_xlabel("时间 (秒)"); ax_t.set_ylabel("振幅")
        ax_t.grid(True, alpha=0.3); ax_t.legend(loc="upper right", fontsize=9)

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        bar = NavigationToolbar2Tk(canvas, top, pack_toolbar=False)
        bar.update(); bar.pack(fill="x")
        canvas.draw_idle()
        # 这个窗口内的两个子图支持滚轮缩放与拖动平移（但不再弹新窗口）
        self._bind_plot_interactions(canvas, allow_enlarge=False)
        self._detail_windows.append({"window": top, "figure": fig, "title": title,
                                     "n_curves": len(methods) + 1})
        self.status(f"已打开放大对比窗口（{len(methods)} 个方案 + 原信号）")

    def _show_last_compare_enlarged(self) -> None:
        """放大查看最近一次对比的全部方案（通过对比对话框的按钮触发）。"""
        results = getattr(self, "_last_compare", {}).get("results")
        if results:
            self._plot_single_method_spectrum(results, None)

    # =====================================================================
    # 降噪结果对比（多种方法同台比较）
    # =====================================================================
    def _refresh_compare_plot(self, results: dict[str, np.ndarray]) -> None:
        """在右下角画出各降噪方法的频谱曲线，便于直观比较。"""
        self.ax4.clear()
        self.ax4.grid(True, alpha=0.3)
        colors = {"原信号": "#9AA0A6", "GTCRN 预训练": "#0072BD",
                  "自训练 MLP": "#D95319", "经典谱减法": "#77AC30"}
        for name, sig in results.items():
            sp = dsp.spectrum(sig[:len(self.y)] if len(sig) >= len(self.y) else sig, self.fs)
            self.ax4.plot(sp["freqs"], sp["mag"], linewidth=0.75,
                          color=colors.get(name, None), label=name, alpha=0.9)
        self.ax4.set_title("各方法降噪后的频谱对比")
        self.ax4.set_xlabel("频率 (Hz)")
        self.ax4.set_ylabel("幅值")
        self.ax4.set_xlim(0, self.fs / 4)
        self.ax4.set_yscale("log")
        self.ax4.legend(loc="upper right", fontsize=8)
        self.fig.set_layout_engine("constrained")
        self._redraw()

    # =====================================================================
    # 工具
    # =====================================================================
    def status(self, text: str) -> None:
        self.var_status.set(text)
        self.root.update_idletasks()

    def warn(self, text: str) -> None:
        messagebox.showwarning("提示", text)

    def need_signal(self) -> bool:
        if self.y is None or len(self.y) == 0:
            self.warn("还没有音频数据，请先点「① 录音」或「② 导入音频」。")
            return False
        return True

    def need_denoised(self) -> bool:
        if self.yx is None or len(self.yx) == 0:
            self.warn("还没有处理结果，请先执行降噪，或在「音效」页添加混响 / 去混响。")
            return False
        return True

    def _plot(self, ax, x, y, color=C_ORIG, title=None, xlabel=None, ylabel="幅值", **kw) -> None:
        ax.clear()
        ax.grid(True, alpha=0.3)
        ax.plot(x, y, color=color, linewidth=0.8, **kw)
        if title:
            ax.set_title(title)
        ax.set_xlabel(xlabel or "采样点")
        ax.set_ylabel(ylabel)
        self.fig.set_layout_engine("constrained")

    def _redraw(self) -> None:
        self.canvas.draw_idle()

    def _autoload_sample(self) -> None:
        p = audio_io.default_sample_path()
        if p:
            try:
                self._load_path(p, silent=True)
                self.status(f"已自动载入示例音频 {os.path.basename(p)}（{len(self.y) / self.fs:.2f} s）"
                            "；可直接试「⑤ GTCRN 预训练降噪 ★」")
            except Exception as exc:
                self.status(f"示例音频载入失败：{exc}")

    def _load_path(self, path: str, silent: bool = False) -> None:
        y, fs = audio_io.load_audio(path)
        if not y.size or not np.isfinite(y).all():
            raise ValueError("音频为空或包含 NaN/Inf")
        self.tasks.invalidate()
        self._processing_history = []
        self.var_effect_source.set("原信号")
        self._notch_info = None
        self._last_compare = {}
        if self._choice_window is not None and self._choice_window.winfo_exists():
            self._choice_window.destroy()
        self.y_raw = y
        self.y = audio_io.to_mono(y)
        self.fs = int(fs)
        self.path = path
        self.yx = None
        self.yr = None
        self.var_fs.set(str(fs))
        ch = 1 if y.ndim == 1 else y.shape[1]
        self.ax1.clear(); self.ax2.clear(); self.ax3.clear(); self.ax4.clear()
        for ax in (self.ax1, self.ax2, self.ax3, self.ax4):
            ax.grid(True, alpha=0.3)
        self._set_titles()
        t = np.arange(len(self.y))
        self._plot(self.ax1, t, self.y, C_ORIG, "时域波形（原信号）", "采样点")
        self._plot(self.ax4, np.arange(len(self.y)) / self.fs, self.y, C_ORIG,
                   "时域波形（时间轴）", "时间 (秒)")
        if not silent:
            self.status(f"已导入 {os.path.basename(path)}：{fs} Hz，{ch} 声道，{len(self.y) / fs:.2f} s")
        self._redraw()

    # =====================================================================
    # ① 录音
    # =====================================================================
    def on_record(self) -> None:
        if self._recording:
            self.warn("正在录音，请稍候…")
            return
        try:
            fs = int(float(self.var_fs.get()))
            dur = float(self.var_dur.get())
            ch = int(self.var_ch.get())
        except ValueError:
            self.warn("录音参数必须是数字。")
            return
        if not np.isfinite(dur) or not 0 < dur <= 3600:
            self.warn("录音时长必须在 0~3600 秒之间。")
            return
        if fs < 8000 or fs > 48000:
            self.warn("采样率建议 8000~48000 Hz。")
            return
        self._recording = True
        revision = self.tasks.revision
        self.status(f"开始录音 {dur:.0f} 秒（{fs} Hz / {ch} 声道）…")

        def _work() -> None:
            try:
                data = audio_io.record(duration=dur, fs=fs, channels=ch,
                                       on_progress=lambda e, t: self._log_q.put(
                                           f"__PROGRESS__{e / max(t, 1e-6):.3f}"))
                mono = audio_io.to_mono(data)
                # 1) 归档：每次录音都留一份，方便累积训练语料（不再互相覆盖）
                os.makedirs(RECORD_DIR, exist_ok=True)
                stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1000000:06d}"
                arch = os.path.join(RECORD_DIR, f"rec_{stamp}.wav")
                audio_io.save_audio(arch, mono, fs)
                # 2) 项目内工作副本；不改动原 MATLAB 项目的素材。
                out = os.path.join(HERE, "data.wav")
                audio_io.save_audio(out, mono, fs)
                self._log_q.put({"type": "recorded", "path": arch, "revision": revision})
            except Exception as exc:
                self._log_q.put(f"__ERROR__录音失败：{exc}")
            finally:
                self._recording = False

        threading.Thread(target=_work, daemon=True).start()

    # =====================================================================
    # ② 导入
    # =====================================================================
    def on_import(self) -> None:
        path = filedialog.askopenfilename(
            title="选择语音信号",
            filetypes=[("音频文件", "*.wav *.m4a *.mp3 *.flac *.ogg *.aiff *.aif"), ("所有文件", "*.*")],
            initialdir=os.path.dirname(os.path.dirname(HERE)))
        if not path:
            return
        try:
            self._load_path(path)
        except Exception as exc:
            messagebox.showerror("导入失败", f"{exc}\n\nWAV 以外的格式需要 soundfile 支持。")

    # =====================================================================
    # ③ 播放（下拉选择播放对象）
    # =====================================================================
    def on_play(self) -> None:
        if not self.need_signal():
            return
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="原信号", command=lambda: self._play(self.y, "原信号"))
        if self.yx is not None:
            menu.add_command(label=f"处理后信号（{len(self.yx) / self.fs:.2f} s）",
                             command=lambda: self._play(self.yx, "处理后信号"))
        if self.yr is not None:
            menu.add_command(label="回声 / 倒放结果", command=lambda: self._play(self.yr, "回声/倒放结果"))
        menu.add_separator()
        menu.add_command(label="停止播放", command=lambda: (audio_io.stop_play(), self.status("已停止播放")))
        try:
            menu.tk_popup(self.buttons["③ 播放"].winfo_rootx(),
                          self.buttons["③ 播放"].winfo_rooty() + 26)
        finally:
            menu.grab_release()

    def _play(self, y: np.ndarray, name: str, fs=None) -> None:
        self.status(f"正在播放：{name}")
        snapshot, fs = np.array(y, copy=True), int(self.fs if fs is None else fs)
        self.music_panel.player.pause()
        self.speaker_panel.player.pause()
        def work():
            try:
                audio_io.play(snapshot, fs, blocking=True)
                self._log_q.put("__PLAYED__")
            except Exception as exc:
                self._log_q.put(f"__ERROR__播放失败：{exc}")
        threading.Thread(target=work, daemon=True).start()

    # =====================================================================
    # ④ 回声
    # =====================================================================
    def on_echo(self) -> None:
        if not self.need_signal():
            return
        delay, a = 10000, 0.3          # 与原 MATLAB 完全一致
        self.yr = dsp.echo(self.y, self.fs, delay=delay, a=a)
        t = np.arange(len(self.yr))
        self._plot(self.ax2, t, self.yr, C_DEN,
                   f"回声信号（延迟 {delay / self.fs:.2f} s，幅度 {a}）", "采样点")
        self._redraw()
        self.status(f"回声已生成：延迟 {delay} 采样点（{delay / self.fs:.3f} s），回声幅度 {a}")
        self._play(self.yr, "回声 / 倒放")

    # =====================================================================
    # ⑥ 自训练 MLP 降噪
    # =====================================================================
    def on_denoise_ml(self) -> None:
        if not self.denoiser.is_ready:
            self.warn("自训练模型不可用，请训练或选择已有模型版本。")
            return
        self._run_audio_task("自训练 MLP", lambda y, fs: self.denoiser.enhance(y, fs=fs, gtcrn=self.gtcrn))

    def _on_auto_notch_toggle(self):
        auto = self.var_auto_notch.get()
        self.lbl_notch_hint.config(text=(
            "自动：检出持续窄带才陷波，其余做谱减。\n四个频率框用于预览，手动模式才参与处理。"
            if auto else "手动：使用四个频率做带阻后谱减。\n须满足 fp1 < fs1 < fs2 < fp2 < Nyquist。"))

    def _classic_config(self):
        if self.var_auto_notch.get():
            return {"auto": True, "notch": None}
        try:
            frequencies = tuple(float(v.get()) for v in (self.var_fp1, self.var_fp2, self.var_fs1, self.var_fs2))
        except ValueError as exc:
            raise ValueError("手动带阻频率必须是数字") from exc
        dsp.design_bandstop(self.fs, *frequencies)  # 先校验，错误不能偷偷退回默认频率。
        return {"auto": False, "notch": frequencies}

    @staticmethod
    def _classic_work(y, fs, config):
        out, info = dsp.classic_denoise_with_notch(y, fs, **config)
        return {"output": out, "notch": info, "parameters": config}

    def on_denoise_classic(self, notify=True) -> None:
        if not self.need_signal():
            return
        try:
            config = self._classic_config()
        except ValueError as exc:
            self.warn(str(exc))
            return
        def work(y, fs):
            result = self._classic_work(y, fs, config)
            result["notify"] = bool(notify)
            return result
        self._run_audio_task("经典谱减法", work)

    def _fill_notch_from_signal(self):
        self._run_audio_task("陷波检测", lambda y, fs: dsp.analyze_narrowband(y, fs))

    def _apply_notch_preview(self, info):
        if not info.get("enabled"):
            self.status(info["reason"])
            messagebox.showinfo("未检出持续窄带干扰", info["reason"])
            return
        center, half = info["center"], info["halfwidth"]
        nyq = self.fs / 2
        values = (max(2, center - 2.6 * half), min(nyq - 20, center + 2.6 * half),
                  max(5, center - half), min(nyq - 50, center + half))
        for var, value in zip((self.var_fp1, self.var_fp2, self.var_fs1, self.var_fs2), values):
            var.set(f"{value:.2f}")
        self._draw_filter_response()
        self._redraw()
        self.status(f"已填入 {center:.1f} Hz 附近的频率；仅预览，未修改音频。")
        messagebox.showinfo("已填入带阻频率", info["reason"] +
                            "\n\n这是预览。切换手动模式后，再点⑦才会使用这些频率处理。")

    def _record_classic_result(self, result):
        self._notch_info = result["notch"]
        self._processing_history = [{"algorithm": "经典谱减法", "parameters": result["parameters"],
                                     "notch": self._notch_info}]
        self._draw_applied_notch(self._notch_info)
        self._redraw()

    def _draw_applied_notch(self, info):
        from scipy import signal
        frequencies = np.linspace(0, self.fs / 2, 16384)
        response = np.ones(len(frequencies), dtype=np.complex128)
        for entry in info.get("filters", []):
            _, h = signal.sosfreqz(np.array(entry["sos"]), worN=frequencies, fs=self.fs)
            response *= h
        self.ax3.clear()
        self.ax3.grid(True, alpha=.3)
        self.ax3.plot(frequencies, 20 * np.log10(np.maximum(np.abs(response), 1e-8)), color=C_FILT)
        self.ax3.set_title("本次实际带阻响应" if info["applied"] else "本次跳过带阻（只做谱减）")
        self.ax3.set_xlabel("频率 (Hz)"); self.ax3.set_ylabel("幅度 (dB)")
        self.ax3.set_xlim(0, min(self.fs / 2, 4000)); self.ax3.set_ylim(-60, 3)

    def on_reverb(self):
        try:
            params = {"rt60": float(self.var_reverb_rt60.get()),
                      "wet": float(self.var_reverb_wet.get()) / 100,
                      "pre_delay_ms": float(self.var_reverb_delay.get()),
                      "damping": float(self.var_reverb_damping.get()) / 100}
        except ValueError:
            self.warn("混响参数必须是数字。")
            return
        self._run_room_effect("添加混响", room_effects.add_reverb, params)

    def on_dereverb(self):
        try:
            params = {"strength": float(self.var_dereverb_strength.get()) / 100,
                      "taps": 10, "delay": 3, "iterations": 3}
        except ValueError:
            self.warn("去混响强度必须是数字。")
            return
        self._run_room_effect("WPE 去混响", room_effects.dereverb, params)

    def _run_room_effect(self, name, algorithm, params):
        if not self.need_signal():
            return
        use_processed = self.var_effect_source.get() == "当前处理结果"
        if use_processed and not self.need_denoised():
            return
        source = self.yx if use_processed else self.y
        history = list(self._processing_history) if use_processed else []
        if use_processed and not history:
            history = [{"algorithm": getattr(self, "_denoise_tag", "处理结果"),
                        "model": self.denoiser.model_path}]
        history = history + [{"algorithm": name, "parameters": params}]
        def work(audio, fs):
            output = algorithm(audio, fs, **params)
            return {"output": output, "history": history, "parameters": params}
        self._run_audio_task(name, work, audio=source)

    def _apply_denoised(self, out: np.ndarray, tag: str, reference=None) -> None:
        out = np.asarray(out, dtype=np.float32).reshape(-1)
        if self.y is None or out.shape != self.y.shape or not np.isfinite(out).all():
            raise ValueError(f"{tag} 输出长度或数值异常，已拒绝应用；请检查算法采样率")
        self._denoise_tag = tag
        original = self.y if reference is None else reference
        self._processing_history = []
        self._notch_info = None
        self._denoise_metrics = metrics.assess(original, out, self.fs)
        self.yx = out
        n = min(len(self.y), len(self.yx))
        raw_peak = float(np.max(np.abs(self.y[:n]))) or 1.0
        out_peak = float(np.max(np.abs(self.yx[:n]))) or 1.0
        drop_db = 20 * np.log10(out_peak / raw_peak)
        t = np.arange(len(self.yx))
        self._plot(self.ax2, t, self.yx, C_DEN, f"处理后时域波形（{tag}）", "采样点")
        # 频谱对比
        sp_o = dsp.spectrum(original, self.fs)
        sp_d = dsp.spectrum(self.yx, self.fs)
        self.ax4.clear(); self.ax4.grid(True, alpha=0.3)
        self.ax4.plot(sp_o["freqs"], sp_o["mag"], color=C_ORIG, linewidth=0.7, label="本次处理输入")
        self.ax4.plot(sp_d["freqs"], sp_d["mag"], color=C_DEN, linewidth=0.7, alpha=0.85, label=f"{tag}后")
        self.ax4.set_title("频谱对比（处理前后）")
        self.ax4.set_xlabel("频率 (Hz)"); self.ax4.set_ylabel("幅值")
        self.ax4.set_xlim(0, self.fs / 4)
        span = max(np.percentile(sp_o["mag"], 99.5), np.percentile(sp_d["mag"], 99.5)) * 1.15
        self.ax4.set_ylim(0, max(span, 1e-3))
        self.ax4.legend(loc="upper right", fontsize=8)
        self.fig.set_layout_engine("constrained")
        self._redraw()
        self.status(f"{tag}完成：峰值 {raw_peak:.3f} → {out_peak:.3f}（{drop_db:+.1f} dB）；"
                    "可点「⑫ 处理后时域」播放/查看")

    def _draw_filter_response(self) -> None:
        try:
            fp1, fp2 = float(self.var_fp1.get()), float(self.var_fp2.get())
            fs1, fs2 = float(self.var_fs1.get()), float(self.var_fs2.get())
        except ValueError:
            return
        try:
            info = dsp.design_bandstop(self.fs, fp1, fp2, fs1, fs2)
        except Exception:
            return
        self.ax3.clear(); self.ax3.grid(True, alpha=0.3)
        self.ax3.plot(info["freqs"], info["mag_db"], color=C_FILT, linewidth=1.2)
        self.ax3.set_title(f"带阻滤波器幅频响应（n={info['order']}，{fs1:.0f}~{fs2:.0f} Hz）")
        self.ax3.set_xlabel("频率 (Hz)"); self.ax3.set_ylabel("幅度 (dB)")
        self.ax3.set_xlim(0, self.fs / 4)
        self.ax3.set_ylim(-40, 2)

    # =====================================================================
    # ⑨ 端点检测
    # =====================================================================
    def on_endpoint(self) -> None:
        if not self.need_signal():
            return
        thr = 0.02                     # 与原 MATLAB 相同的阈值（信号已归一化到 [-1,1]）
        det = dsp.endpoint_detect(self.y, self.fs, energy_threshold=thr)
        self.ax4.clear(); self.ax4.grid(True, alpha=0.3)
        t = np.arange(len(self.y)) / self.fs
        self.ax4.plot(t, self.y, color="#9AA0A6", linewidth=0.6, label="全信号")
        if det["detected"]:
            s, e = det["start_sample"] - 1, det["end_sample"]
            self.ax4.plot(t[s:e], self.y[s:e], color=C_HL, linewidth=0.9, label="检测到的语音段")
            self.ax4.legend(loc="upper right", fontsize=8)
            msg = (f"端点检测：起始 {det['start_time']:.3f} s，结束 {det['end_time']:.3f} s，"
                   f"共 {det['end_sample'] - det['start_sample']} 采样点（帧长 {det['frame_size']}，帧移 {det['frame_step']}）")
        else:
            msg = f"未检测到语音活动（阈值 {thr}）；可换更小的阈值再试"
        self.ax4.set_title("语音端点检测结果（短时能量法）")
        self.ax4.set_xlabel("时间 (秒)"); self.ax4.set_ylabel("振幅")
        self.fig.set_layout_engine("constrained"); self._redraw()
        self.status(msg)
        if det["detected"]:
            self._play(det["speech"], "检测到的语音")

    # =====================================================================
    # ⑩ 倒放
    # =====================================================================
    def on_reverse(self) -> None:
        if not self.need_signal():
            return
        self.yr = dsp.reverse(self.y)
        self._plot(self.ax2, np.arange(len(self.yr)), self.yr, C_DEN, "倒放信号", "采样点")
        self._redraw()
        self.status("已生成倒放信号（对应 MATLAB 的 flip(y,1)）并开始播放")
        self._play(self.yr, "回声 / 倒放")

    # =====================================================================
    # ⑪ / ⑫ 频谱
    # =====================================================================
    def on_spectrum_before(self) -> None:
        if not self.need_signal():
            return
        sp = dsp.spectrum(self.y, self.fs)
        self.ax1.clear(); self.ax1.grid(True, alpha=0.3)
        self.ax1.stem(sp["freqs"], sp["mag"], markerfmt=" ", basefmt=" ", linefmt="C0-")
        self.ax1.set_title("原语音信号的频谱")
        self.ax1.set_xlabel("频率 (Hz)"); self.ax1.set_ylabel("幅值")
        self.ax1.set_xlim(0, self.fs / 4)
        self.ax1.set_ylim(0, max(np.percentile(sp["mag"], 99.5) * 1.2, 1e-3))
        self.fig.set_layout_engine("constrained"); self._redraw()
        self.status(f"原信号频谱：N={len(self.y)} 点 FFT，频率分辨率 {sp['df']:.2f} Hz（显示 0~{self.fs / 4000:.0f} kHz）")

    def on_spectrum_after(self) -> None:
        if not self.need_denoised():
            return
        sp = dsp.spectrum(self.yx, self.fs)
        self.ax2.clear(); self.ax2.grid(True, alpha=0.3)
        self.ax2.stem(sp["freqs"], sp["mag"], markerfmt=" ", basefmt=" ", linefmt="C1-")
        self.ax2.set_title("处理后的频谱")
        self.ax2.set_xlabel("频率 (Hz)"); self.ax2.set_ylabel("幅值")
        self.ax2.set_xlim(0, self.fs / 4)
        self.ax2.set_ylim(0, max(np.percentile(sp["mag"], 99.5) * 1.2, 1e-3))
        self.fig.set_layout_engine("constrained"); self._redraw()
        self.status("处理后频谱已绘制（频谱对比图见右下角）")

    # =====================================================================
    # ⑬ / ⑭ 时域
    # =====================================================================
    def on_time_before(self) -> None:
        if not self.need_signal():
            return
        self._plot(self.ax1, np.arange(len(self.y)), self.y, C_ORIG, "原语音信号的时域图", "采样点")
        self._redraw()
        self.status(f"原信号时域图：{len(self.y)} 采样点，{len(self.y) / self.fs:.2f} s")

    def on_time_after(self) -> None:
        if not self.need_denoised():
            return
        self._plot(self.ax2, np.arange(len(self.yx)), self.yx, C_DEN, "处理后信号的时域图", "采样点")
        self._redraw()
        self.status("正在播放处理后信号…")
        self._play(self.yx, "处理后信号")

    # =====================================================================
    # 机器学习：模型管理
    # =====================================================================
    def _refresh_model_status(self) -> None:
        if self.gtcrn.is_ready:
            self.lbl_gtcrn.config(text="预训练模型：已就绪 ✓\n" + self.gtcrn.describe())
        else:
            self.lbl_gtcrn.config(text=f"预训练模型：不可用\n{self.gtcrn.load_error}")
        if self.denoiser.is_ready:
            self.lbl_model.config(text="自训练模型：已加载 ✓\n" + self.denoiser.describe())
        else:
            self.lbl_model.config(text="自训练模型：未训练\n点下方按钮用你的录音+噪声训练")

    def on_reload_model(self) -> None:
        if self.tasks.busy or self._training:
            self.warn("请等待音频处理或训练完成，再重新加载模型。")
            return
        self.denoiser.model_path = mlp.default_model_path()
        ok = self.denoiser.reload()
        self.gtcrn.reload()
        self._refresh_model_status()
        self.status("模型已重新加载" if ok else f"自训练模型加载失败：{self.denoiser.load_error}")

    # =====================================================================
    # ⑤ GTCRN 预训练降噪（ICASSP 2024，DNS3 训练，onnxruntime 推理）
    # =====================================================================
    def on_denoise_gtcrn(self) -> None:
        if not self.gtcrn.is_ready:
            self.warn(f"预训练模型不可用：{self.gtcrn.load_error}")
            return
        self._run_audio_task("GTCRN 预训练", lambda y, fs: self.gtcrn.enhance(y, fs))

    def on_compare_denoisers(self) -> None:
        if not self.need_signal():
            return
        try:
            classic_config = self._classic_config()
        except ValueError as exc:
            self.warn(str(exc))
            return
        methods = []
        if self.gtcrn.is_ready:
            methods.append(("GTCRN 预训练", lambda y, fs: self.gtcrn.enhance(y, fs)))
        if self.denoiser.is_ready:
            methods.append(("自训练 MLP", lambda y, fs: self.denoiser.enhance(y, fs=fs, gtcrn=self.gtcrn)))
        methods.append(("经典谱减法", lambda y, fs: self._classic_work(y, fs, classic_config)))
        if self.gtcrn.is_ready and self.denoiser.is_ready:
            methods.append(("串联 GTCRN→自训练", lambda y, fs: self.denoiser.enhance_cascaded(y, self.gtcrn, fs=fs)))
        if len(methods) < 2:
            self.warn("至少需要两种可用方法才能对比。")
            return
        def work(y, fs):
            results, reports = {"原信号": y}, {}
            classic_result = None
            for name, fn in methods:
                computed = fn(y, fs)
                if name == "经典谱减法":
                    classic_result = computed
                    computed = computed["output"]
                out = np.asarray(computed, dtype=np.float32)
                reports[name] = metrics.assess(y, out, fs)
                results[name] = out
            return {"results": results, "metrics": reports, "classic_result": classic_result}
        self._run_audio_task("多方案对比", work)

    def on_train_model(self) -> None:
        if self.tasks.busy:
            self.warn("请先等待音频处理完成。")
            return
        if self._training:
            self.warn("正在训练中，请看状态栏进度。")
            return
        clean = self._find_training_files()
        if not clean:
            self.warn("没有找到用于训练的干净语音。\n\n"
                      "请先点「① 录音」录几段（会自动归档到 recordings 目录，可累计多次），\n"
                      "或点「导入干净语音…」选择已有的 wav/mp3 文件。")
            return
        noise_dirs = self._find_noise_dirs()
        self._training = True
        self._train_cancel.clear()
        self.progress.config(mode="indeterminate")
        self.progress.start(60)
        n_real = sum(len(glob.glob(os.path.join(d, "**", "*.wav"), recursive=True)) for d in noise_dirs)
        self.status(f"开始训练：{len(clean)} 个语音文件"
                   + (f" + {n_real} 个真实噪声文件" if n_real else " + 程序合成噪声")
                   + "…（详见下方命令行输出）")

        def _work() -> None:
            cmd = [sys.executable, "-u", os.path.join(HERE, "train_model.py"),
                   "--clean", *clean, "--context", "6", "--epochs", "18",
                   "--steps-per-epoch", "40", "--batch-size", "8",
                   "--hidden", "256,128", "--lr", "2.5e-3", "--activate"]
            if noise_dirs:
                cmd += ["--noise-dir", *noise_dirs]
            try:
                with self._train_lock:
                    if self._train_cancel.is_set():
                        self._log_q.put("__TRAINDONE__-1")
                        return
                    proc = self._train_proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, encoding="utf-8", errors="replace", bufsize=1,
                                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                assert proc.stdout is not None
                for line in proc.stdout:
                    self._log_q.put("__TRAIN__" + line.rstrip())
                proc.wait()
                self._log_q.put(f"__TRAINDONE__{proc.returncode}")
            except Exception as exc:
                self._log_q.put(f"__TRAINERROR__训练进程启动失败：{exc}")

        threading.Thread(target=_work, daemon=True).start()

    def on_export(self) -> None:
        if not self.need_denoised():
            return
        from uuid import uuid4
        base = os.path.splitext(os.path.basename(self.path or "denoised"))[0]
        tag = getattr(self, "_denoise_tag", "denoised").replace("→", "_").replace(" ", "_")
        name = f"{base}_{tag}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}.wav"
        out = filedialog.asksaveasfilename(title="导出处理结果", initialdir=RESULT_DIR,
                initialfile=name, defaultextension=".wav", filetypes=[("WAV", "*.wav")])
        if not out:
            return
        try:
            audio_io.save_audio(out, self.yx, self.fs)
            metadata = {"source": self.path, "algorithm": getattr(self, "_denoise_tag", ""),
                        "fs": self.fs, "samples": len(self.yx),
                        "model": self.denoiser.model_path if not self._processing_history else None,
                        "processing": self._processing_history,
                        "notch": self._notch_info,
                        "metrics": getattr(self, "_denoise_metrics", {})}
            with open(out + ".json", "w", encoding="utf-8") as stream:
                json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
            self.status(f"已导出音频及参数清单：{out}")
        except Exception as exc:
            messagebox.showerror("导出失败", str(exc))

    def _find_training_files(self) -> list[str]:
        files = default_clean_paths()
        for ext in ("*.flac", "*.mp3", "*.m4a", "*.ogg"):
            files.extend(glob.glob(os.path.join(RECORD_DIR, ext)))
        return list(dict.fromkeys(files))

    def _find_noise_dirs(self) -> list[str]:
        """真实噪声数据集目录（由 download_datasets.py 下载或「导入真实噪声…」导入）。"""
        real = os.path.join(NOISE_DIR, "real")
        if os.path.isdir(real) and glob.glob(os.path.join(real, "**", "*.wav"), recursive=True):
            return [real]
        return []

    def _update_corpus_label(self) -> None:
        files = self._find_training_files()
        if not files:
            self.lbl_corpus.config(text="语料：暂无\n请先「① 录音」或「导入干净语音…」")
            return

        def _work() -> None:
            total = 0.0
            for p in files:
                try:
                    y, sr = audio_io.load_audio(p)
                    total += (y.shape[0] if y.ndim == 1 else y.shape[0]) / float(sr)
                except Exception:
                    pass
            noise_files = sum(len(glob.glob(os.path.join(d, "**", "*.wav"), recursive=True))
                              for d in self._find_noise_dirs())
            noise_txt = (f"真实噪声 {noise_files} 个文件" if noise_files else "程序化合成 8 类")
            tip = "\n（语音偏少，建议多录几段）" if total < 60 else ""
            txt = f"语料：{len(files)} 个语音文件，共 {total:.1f} 秒\n噪声：{noise_txt}{tip}"
            self._log_q.put("__CORPUS__" + txt)

        threading.Thread(target=_work, daemon=True).start()

    def on_import_folder(self) -> None:
        """批量导入整个文件夹（递归子目录），自动重采样/去重/质检。

        用的是与命令行 corpus_tool.py 完全相同的核心函数 import_corpus()，
        所以界面与命令行行为一致，不存在两套实现。
        """
        d = filedialog.askdirectory(title="选择包含音频的文件夹（会递归处理子目录）")
        if not d:
            return
        d = os.path.normpath(d)
        if not messagebox.askyesno(
                "批量导入训练语音",
                f"将递归导入：\n{d}\n\n"
                "自动处理：统一转 16 kHz 单声道、峰值归一化、跳过重复文件、\n"
                "跳过 0.3 秒以下或 30 秒以上的片段，并在导入后做底噪质检。\n\n"
                "原件不会被修改（复制进 recordings/）。是否继续？"):
            return

        top = self._new_detail_window("批量导入训练语音 —— 进行中", "560x260")
        ttk.Label(top, text=f"来源：{d}", wraplength=520, justify="left").pack(anchor="w", padx=12, pady=(10, 4))
        info = tk.StringVar(value="正在收集文件…")
        ttk.Label(top, textvariable=info, wraplength=520, justify="left").pack(anchor="w", padx=12)
        bar = ttk.Progressbar(top, mode="determinate", length=500)
        bar.pack(padx=12, pady=8)
        self._import_ui = {"window": top, "info": info, "bar": bar, "done": False}

        def _work() -> None:
            try:
                from corpus_tool import import_corpus

                def _prog(done: int, total: int, name: str, stats: dict) -> None:
                    self._log_q.put(f"__IMPORT_P__{done}|{total}|{name[:36]}|"
                                    f"{stats['n_ok']}|{stats['n_dup']}|{stats['n_skip']}")

                res = import_corpus(source=d, recursive=True, progress=_prog)
                self._log_q.put("__IMPORT_DONE__" + json.dumps(res, ensure_ascii=False))
            except Exception as exc:
                self._log_q.put(f"__ERROR__批量导入失败：{exc}\n{traceback.format_exc()}")

        threading.Thread(target=_work, daemon=True).start()

    def _finish_folder_import(self, res: dict) -> None:
        """导入结束后：更新语料统计、刷新进度窗口、汇报结果。"""
        ui = getattr(self, "_import_ui", None)
        if ui and ui["window"].winfo_exists():
            if res.get("error"):
                ui["info"].set(f"失败：{res['error']}")
            else:
                ui["info"].set(
                    f"完成：成功 {res['n_ok']} 个（{res['seconds']:.1f} 秒）\n"
                    f"重复跳过 {res['n_dup']}｜时长不符 {res['n_skip']}｜读取失败 {res['n_fail']}")
                ui["bar"]["value"] = 100
                ui["done"] = True
            ui["window"].after(1600, lambda: ui["window"].winfo_exists() and ui["window"].destroy())

        if res.get("error"):
            messagebox.showerror("批量导入失败", res["error"])
            return
        self._update_corpus_label()
        msg = (f"批量导入完成\n\n成功 {res['n_ok']} 个（{res['seconds']:.1f} 秒）\n"
               f"重复跳过 {res['n_dup']} 个｜时长不符 {res['n_skip']} 个｜读取失败 {res['n_fail']} 个")
        if res.get("quality_warnings"):
            msg += "\n\n" + "\n".join(res["quality_warnings"][:6])
        msg += "\n\n下一步：点「训练降噪模型」即可把这些素材纳入训练。"
        messagebox.showinfo("批量导入结果", msg)
        self.status(f"批量导入完成：新增 {res['n_ok']} 个片段"
                   + (f"，{res['n_dup']} 个重复已跳过" if res["n_dup"] else ""))

    def on_import_speech(self) -> None:
        paths = filedialog.askopenfilenames(
            title="选择用于训练的干净语音（可多选）",
            filetypes=[("音频文件", "*.wav *.flac *.mp3 *.m4a *.ogg"), ("所有文件", "*.*")])
        if not paths:
            return
        os.makedirs(RECORD_DIR, exist_ok=True)
        n = 0
        for p in paths:
            try:
                dst = os.path.join(RECORD_DIR, f"user_{os.path.basename(p)}")
                if os.path.abspath(p) != os.path.abspath(dst):
                    shutil.copy2(p, dst)
                n += 1
            except Exception as exc:
                self.status(f"复制失败：{os.path.basename(p)}（{exc}）")
        self.status(f"已导入 {n} 个语音文件到 recordings/，下次训练会自动纳入")
        self._update_corpus_label()

    def on_import_noise(self) -> None:
        paths = filedialog.askopenfilenames(
            title="选择真实噪声音频（可多选，如 DEMAND / MUSAN / ESC-50 里的片段）",
            filetypes=[("音频文件", "*.wav *.flac *.mp3 *.m4a *.ogg"), ("所有文件", "*.*")])
        if not paths:
            return
        real = os.path.join(NOISE_DIR, "real")
        os.makedirs(real, exist_ok=True)
        n = 0
        for p in paths:
            try:
                shutil.copy2(p, os.path.join(real, os.path.basename(p)))
                n += 1
            except Exception as exc:
                self.status(f"复制失败：{os.path.basename(p)}（{exc}）")
        self.status(f"已导入 {n} 个真实噪声文件，训练时会与合成噪声混合使用")
        self._update_corpus_label()

    def on_open_data_dir(self) -> None:
        os.makedirs(RECORD_DIR, exist_ok=True)
        os.makedirs(NOISE_DIR, exist_ok=True)
        if not _open_in_file_manager(RECORD_DIR):
            messagebox.showinfo("数据目录", f"语音语料：{RECORD_DIR}\n噪声库：{NOISE_DIR}")

    # =====================================================================
    # 后台线程 -> 主线程消息
    # =====================================================================
    def _drain_log(self) -> None:
        if self._closing:
            return
        try:
            while True:
                msg = self._log_q.get_nowait()
                try:
                    self._handle_msg(msg)
                except Exception as exc:
                    self.status(f"结果处理失败：{exc}")
        except queue.Empty:
            pass
        self._drain_after = self.root.after(150, self._drain_log)

    def _handle_msg(self, msg) -> None:
        if isinstance(msg, dict):
            if msg.get("type") == "recorded":
                self.progress["value"] = 0
                if msg["revision"] == self.tasks.revision:
                    self._load_path(msg["path"])
                    self.status("录音完成，已载入并归档到 recordings/。")
                else:
                    self.status("录音已归档；保留您刚导入的素材。")
                self._update_corpus_label()
                return
            self._handle_signal_result(msg)
            return
        if msg.startswith("__PROGRESS__"):
            self.progress.config(mode="determinate")
            self.progress["value"] = float(msg.split("__PROGRESS__")[1]) * 100
            return
        if msg.startswith("__RECORDED__"):
            path = msg.split("__RECORDED__", 1)[1]
            self.progress["value"] = 0
            self._load_path(path)
            self.status("录音完成：已载入，并归档到 recordings/（多次录音会累积，训练时全部使用）")
            self._update_corpus_label()
            return
        if msg.startswith("__IMPORT_P__"):
            parts = msg.split("__IMPORT_P__", 1)[1].split("|")
            done, total = int(parts[0]), max(int(parts[1]), 1)
            ui = getattr(self, "_import_ui", None)
            if ui and ui["window"].winfo_exists():
                ui["bar"]["value"] = done / total * 100
                ui["info"].set(f"已处理 {done}/{total}（成功 {parts[3]}，重复 {parts[4]}，"
                               f"时长不符 {parts[5]}）\n当前：{parts[2]}")
            self.status(f"批量导入中：{done}/{total}（成功 {parts[3]}，重复跳过 {parts[4]}）")
            return
        if msg.startswith("__IMPORT_DONE__"):
            self._finish_folder_import(json.loads(msg.split("__IMPORT_DONE__", 1)[1]))
            return
        if msg.startswith("__TRAIN__"):
            line = msg.split("__TRAIN__", 1)[1]
            if line.strip():
                self.status("训练中：" + line.strip()[:110])
            return
        if msg.startswith("__TRAINDONE__"):
            code = int(msg.split("__TRAINDONE__", 1)[1])
            self._training = False
            self.progress.stop()
            self.progress.config(mode="determinate")
            self.progress["value"] = 100 if code == 0 else 0
            if code == 0:
                self.denoiser.model_path = mlp.default_model_path()
                self.denoiser.reload()
                self._refresh_model_status()
                self.status("训练完成 ✓ 自训练模型已加载，可点「⑥ 自训练 MLP 降噪」试用")
                messagebox.showinfo("训练完成", f"降噪模型训练完成并已加载。\n\n{self.denoiser.describe()}\n\n"
                                                "训练日志与评估报告见 models/runs/ 对应版本目录。")
            elif self._train_cancel.is_set():
                self.status("训练已取消，当前模型保持原版本；中途保存的版本可手动选择。")
            else:
                self.status(f"训练失败（退出码 {code}），请检查命令行输出")
            return
        if msg.startswith("__TRAINERROR__"):
            self._training = False
            self.progress.stop()
            messagebox.showerror("训练失败", msg.split("__TRAINERROR__", 1)[1])
            return
        if msg.startswith("__ERROR__"):
            messagebox.showerror("出错了", msg.split("__ERROR__", 1)[1])
            return
        if msg.startswith("__CORPUS__"):
            self.lbl_corpus.config(text=msg.split("__CORPUS__", 1)[1])
            return
        if msg == "__PLAYED__":
            self.status("播放结束")

    def _run_audio_task(self, name, work, audio=None):
        if not self.need_signal():
            return
        if self._training:
            self.warn("训练期间请等待模型保存完成后再处理音频。")
            return
        try:
            self.tasks.submit(name, self.y if audio is None else audio, self.fs, work)
            self.status(f"{name}处理中…")
        except RuntimeError as exc:
            self.warn(str(exc))

    def _handle_signal_result(self, event):
        if not self.tasks.accepts(event):
            return
        if event.get("error"):
            self.status(f"{event.get('name', '音频处理')}失败：{event['error']}")
            messagebox.showerror("处理失败", event["error"])
            return
        name, result = event["name"], event["result"]
        if name == "陷波检测":
            self._apply_notch_preview(result)
        elif name == "经典谱减法" and isinstance(result, dict):
            self._apply_denoised(result["output"], name)
            self._record_classic_result(result)
            self.status("经典谱减法完成：" + result["notch"]["reason"])
            if result.get("notify"):
                title = "谱减法完成（已陷波）" if result["notch"]["applied"] else "谱减法完成（跳过带阻）"
                messagebox.showinfo(title, result["notch"]["reason"] + "\n\n本次带阻响应已显示，可试听或导出处理结果。")
        elif name in ("添加混响", "WPE 去混响"):
            self._apply_denoised(result["output"], name, reference=event["audio"])
            self._processing_history = result["history"]
            self.var_effect_source.set("当前处理结果")
            seconds = len(event["audio"]) / event["fs"]
            note = "；素材过短，已原样保留" if name == "WPE 去混响" and seconds < .4 else ""
            self.status(f"{name}完成{note}，可在「音效」页试听或导出；处理后 {seconds:.2f} 秒。")
        elif name == "多方案对比":
            results, reports = result["results"], result["metrics"]
            self._refresh_compare_plot(results)
            report = "\n".join(metrics.format_report(key, value) for key, value in reports.items())
            chosen = self._ask_method_choice(report, results, reports)
            if self.tasks.accepts(event) and chosen in results:
                self._apply_denoised(results[chosen], chosen)
                if chosen == "经典谱减法" and result.get("classic_result"):
                    self._record_classic_result(result["classic_result"])
                self.status(f"已采用 {chosen}")
        else:
            self._apply_denoised(result, name)
            if name == "经典谱减法":
                self._draw_filter_response()
                self._redraw()
            self.status(metrics.format_report(name, self._denoise_metrics))

    def on_choose_model(self):
        if self.tasks.busy or self._training:
            self.warn("请先等待当前任务完成。")
            return
        path = filedialog.askopenfilename(title="选择模型版本（原版或 models/runs 下的新版本）",
                initialdir=MODEL_DIR, filetypes=[("MLP 模型", "*.npz")])
        if path:
            try:
                mlp.set_active_model_path(path)
                self.on_reload_model()
            except Exception as exc:
                messagebox.showerror("模型加载失败", str(exc))

    def on_cancel_train(self):
        self._train_cancel.set()
        with self._train_lock:
            if self._train_proc is not None and self._train_proc.poll() is None:
                self._train_proc.terminate()
        self.status("训练取消已请求，已保存的版本保留。")

    def on_close(self):
        self._closing = True
        self.tasks.close()
        self.on_cancel_train()
        self.music_panel.close()
        self.speaker_panel.close()
        audio_io.stop_play()
        if hasattr(self, "_drain_after"):
            self.root.after_cancel(self._drain_after)
        self.root.destroy()


def main() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    root = tk.Tk()
    DSPApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
