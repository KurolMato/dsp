"""下载权威噪声数据集，统一整理成训练可用的噪声库。

支持的库（均为学术/工业界常用的权威来源，公开发布、许可清晰）：

    DEMAND  —— 18 种真实环境多通道录音（Zenodo，无需注册）
               https://zenodo.org/records/1227121       许可 CC BY 4.0
    MUSAN   —— 109 小时语音/音乐/噪声语料，只用其中 noise 部分（OpenSLR）
               https://www.openslr.org/17/              许可 CC BY 4.0
    ESC-50  —— 2000 条环境声、50 个类别（键盘/雨声/警笛/狗叫…）（GitHub）
               https://github.com/karolpiczak/ESC-50    许可 CC BY-NC 3.0（非商用）

整理结果都放在 models/noise_bank/real/<数据集或场景>/ 下，每个片段单独一个 wav，
训练时用 --noise-dir models\\noise_bank\\real 即可全部用上。

用法：
    python download_datasets.py --list                          # 看 DEMAND 可选场景
    python download_datasets.py --dataset demand                # 默认 4 个场景（约 400 MB）
    python download_datasets.py --dataset demand --scenes TCAR TMETRO STRAFFIC
    python download_datasets.py --dataset demand --all-scenes   # 全部 17 个场景（约 1.6 GB）
    python download_datasets.py --dataset musan                 # MUSAN 噪声部分（流式，取够即停）
    python download_datasets.py --dataset esc50                 # ESC-50（约 600 MB）
    python download_datasets.py --dataset all                   # 依次下载上面三个
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import urllib.parse
import urllib.request
import zipfile
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np

from core import audio_io

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(HERE, "models", "noise_bank", "real")
PROVENANCE = os.path.join(OUT_ROOT, "SOURCES.json")

ZENODO_API = "https://zenodo.org/api/records/1227121"
MUSAN_URL = "https://www.openslr.org/resources/17/musan.tar.gz"
# 同一份数据给出多个托管点，脚本会先测速再选最快的（国内网络差异极大）
ESC50_URLS = [
    ("codeload", "https://codeload.github.com/karolpiczak/ESC-50/zip/refs/heads/master"),
    ("github-archive", "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"),
    ("ghfast 代理", "https://ghfast.top/https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"),
]
SNSD_URLS = [
    ("codeload", "https://codeload.github.com/microsoft/MS-SNSD/zip/refs/heads/master"),
    ("github-archive", "https://github.com/microsoft/MS-SNSD/archive/refs/heads/master.zip"),
    ("ghfast 代理", "https://ghfast.top/https://github.com/microsoft/MS-SNSD/archive/refs/heads/master.zip"),
]

# DEMAND 场景（名称均已核对存在于数据集中）
SCENES: dict[str, tuple[str, str]] = {
    "DWASHING": ("洗碗机", "DWASHING_16k.zip"),
    "OOFFICE": ("办公室", "OOFFICE_16k.zip"),
    "OMEETING": ("会议室", "OMEETING_16k.zip"),
    "DKITCHEN": ("厨房", "DKITCHEN_16k.zip"),
    "PCAFETER": ("公共餐厅/自助餐厅", "PCAFETER_16k.zip"),
    "PRESTO": ("餐厅", "PRESTO_16k.zip"),
    "OHALLWAY": ("走廊", "OHALLWAY_16k.zip"),
    "DLIVING": ("客厅", "DLIVING_16k.zip"),
    "TBUS": ("公交车", "TBUS_16k.zip"),
    "TCAR": ("车内", "TCAR_16k.zip"),
    "TMETRO": ("地铁", "TMETRO_16k.zip"),
    "STRAFFIC": ("街道交通", "STRAFFIC_16k.zip"),
    "NRIVER": ("河边", "NRIVER_16k.zip"),
    "NPARK": ("公园", "NPARK_16k.zip"),
    "SPSQUARE": ("公共广场", "SPSQUARE_16k.zip"),
    "PSTATION": ("车站/机场", "PSTATION_16k.zip"),
    "NFIELD": ("旷野", "NFIELD_16k.zip"),
}
DEFAULT_SCENES = ["OOFFICE", "DWASHING", "DKITCHEN", "PCAFETER"]

NOISE_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")


# ----------------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------------
def _bar(done: int, total: int, note: str = "") -> None:
    if total > 0:
        pct = min(done / total, 1.0) * 100
        print(f"\r      [{'#' * int(pct // 3):<33}] {pct:5.1f}%  "
              f"{done / 1048576:7.1f}/{total / 1048576:.1f} MB {note}", end="", flush=True)
    else:
        print(f"\r      已下载 {done / 1048576:7.1f} MB {note}", end="", flush=True)


def _note_source(name: str, license_: str, url: str, files: int, seconds: float) -> None:
    """记录数据来源与许可（写进 SOURCES.json，写报告时可直接引用）。"""
    data: dict = {}
    if os.path.isfile(PROVENANCE):
        try:
            with open(PROVENANCE, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    data[name] = {"license": license_, "url": url, "files": files, "seconds": round(seconds, 1)}
    os.makedirs(os.path.dirname(PROVENANCE), exist_ok=True)
    with open(PROVENANCE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _write_clip(dst_dir: str, stem: str, y: np.ndarray, fs: int) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    audio_io.save_audio(os.path.join(dst_dir, f"{stem}.wav"), np.asarray(y, dtype=np.float32), fs)


def _count_wavs(dst_dir: str) -> int:
    if not os.path.isdir(dst_dir):
        return 0
    return len([f for f in os.listdir(dst_dir) if f.lower().endswith(".wav")])


def _resample(y: np.ndarray, sr: int, fs: int) -> np.ndarray:
    if sr == fs:
        return y.astype(np.float32)
    from scipy import signal as sps
    return sps.resample(y, int(round(len(y) * fs / sr))).astype(np.float32)


def _read_audio_bytes(raw: bytes) -> tuple[np.ndarray, int]:
    """从内存字节读音频（soundfile 优先，标准库 wave 兜底）。"""
    import tempfile
    import wave

    try:
        import soundfile as sf
        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        return np.asarray(data, dtype=np.float32), int(sr)
    except Exception:
        pass
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        tf.write(raw)
        path = tf.name
    try:
        with wave.open(path, "rb") as w:
            fs_, n_ch, width, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
            frames = w.readframes(n)
        if width == 2:
            y = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        elif width == 1:
            y = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128) / 128.0
        else:
            y = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
        if n_ch > 1:
            y = y.reshape(-1, n_ch)
        return y, fs_
    finally:
        os.remove(path)


def _urlopen_retry(url: str, tries: int = 4, timeout: int = 120, headers: dict | None = None):
    """带重试的 urlopen。

    两条实战经验：
      1. 国内网络对境外站点时常瞬断，重试能显著提高成功率（指数退避）；
      2. **UA 要按站点区分**——Zenodo 的 API 会拒绝伪装浏览器的 `Mozilla/5.0`
         （返回 403），而 GitHub/OpenSLR 又需要它，所以这里按域名给不同的 UA。
    """
    import time as _t

    host = urllib.parse.urlparse(url).netloc.lower()
    if "zenodo.org" in host:
        ua = "dsp_py-noise-downloader/1.0 (research; python-urllib)"
    else:
        ua = "Mozilla/5.0 (compatible; dsp_py-noise-downloader/1.0)"
    hdrs = {"User-Agent": ua} | (headers or {})

    last: Exception | None = None
    for i in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=hdrs), timeout=timeout)
        except Exception as exc:
            last = exc
            if i < tries - 1:
                wait = 3 * (i + 1)
                print(f"\n      第 {i + 1} 次失败（{str(exc)[:60]}），{wait} 秒后重试…")
                _t.sleep(wait)
    raise last if last else RuntimeError("urlopen 失败")


def _download_resumable(url: str, note: str = "", retries: int = 8,
                        expect_zip: bool = True) -> io.BytesIO:
    """断点续传下载：连接被中途掐断时，用 HTTP Range 从断点继续，而不是整包重下。

    必要性来自实测：Zenodo 会在随机位置掐断传输（111 MB 的文件曾只给到 11 MB、
    13 MB、24 MB），单纯重试要么反复失败、要么白下几百 MB。有了续传，
    每次中断只损失最后一段，几轮之内就能把整包收齐。
    """
    import time as _t

    buf = io.BytesIO()
    total = 0
    attempts = 0
    while attempts < retries:
        attempts += 1
        got = buf.tell()
        if total and got >= total:
            break
        headers = {"Range": f"bytes={got}-"} if got else None
        try:
            with _urlopen_retry(url, tries=2, timeout=90, headers=headers) as resp:
                if not total:
                    total = int(resp.headers.get("Content-Length", 0)) + got
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    buf.write(chunk)
                    _bar(buf.tell(), total, note + (f"（第 {attempts} 段）" if attempts > 1 else ""))
        except Exception as exc:
            print(f"\n      第 {attempts} 段中断（{str(exc)[:50]}），准备续传…")
            _t.sleep(2)
            continue
        print()
        if total and buf.tell() < total:
            print(f"      已收 {buf.tell() / 1048576:.1f}/{total / 1048576:.1f} MB，"
                  f"连接提前结束，{2} 秒后从断点续传…")
            _t.sleep(2)
    size = buf.tell()
    if total and size < total:
        raise RuntimeError(f"续传 {attempts} 轮仍不完整：{size}/{total} 字节")
    buf.seek(0)
    if expect_zip:
        head = buf.read(4)
        buf.seek(0)
        if head[:2] != b"PK":
            raise RuntimeError(f"内容不是 zip（首 4 字节 {head!r}）")
    buf.size_hint = lambda: size          # 便于日志
    return buf


def _download_to_bytes(url: str, note: str = "", tries: int = 3, expect_zip: bool = False) -> io.BytesIO:
    """下载到内存（带完整性校验）。DEMAND 等大文件建议用 _download_resumable。"""
    last_err: Exception | None = None
    for attempt in range(tries):
        with _urlopen_retry(url, tries=2) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            buf = io.BytesIO()
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                buf.write(chunk)
                _bar(buf.tell(), total, note)
        got = buf.tell()
        print()
        buf.seek(0)
        if total > 0 and got != total:
            last_err = RuntimeError(f"下载不完整：收到 {got} 字节，应为 {total} 字节（连接被提前中断）")
            print(f"      第 {attempt + 1} 次不完整（{got}/{total} 字节），重试…")
            continue
        if expect_zip:
            head = buf.read(4)
            buf.seek(0)
            if head[:2] != b"PK":
                last_err = RuntimeError(f"返回内容不是 zip（首 4 字节 {head!r}），可能是服务端临时故障")
                print(f"      第 {attempt + 1} 次内容异常（{head!r}），重试…")
                continue
        return buf
    raise last_err if last_err else RuntimeError("下载失败")


def _download_first_working(candidates: list[tuple[str, str]], note: str = "",
                            probe_bytes: int = 1 << 20) -> io.BytesIO:
    """依次尝试多个下载地址，**先测速再下**，返回第一个可用且较快的。

    为什么要这样：国内网络下同一份数据的不同托管点差异极大。实测
    `github.com/.../archive/...` 直接超时，而 `codeload.github.com/.../zip/...`
    能跑到 1.6 MB/s；公共加速代理则可能只有 0.02 MB/s。先各取 1 MB 试一下，
    比盲目下几百 MB 再失败要省事得多。
    """
    import time as _t

    scored: list[tuple[float, str, str]] = []
    for name, url in candidates:
        try:
            t0 = _t.time()
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (compatible; dsp_py-noise-downloader/1.0)",
                "Range": f"bytes=0-{probe_bytes - 1}"})
            with urllib.request.urlopen(req, timeout=25) as r:
                got = len(r.read(probe_bytes))
            dt = max(_t.time() - t0, 1e-6)
            speed = got / dt
            scored.append((speed, name, url))
            print(f"      测速 {name:<28} {speed / 1048576:6.2f} MB/s")
        except Exception as exc:
            print(f"      测速 {name:<28} 不可用（{str(exc)[:40]}）")
    if not scored:
        raise RuntimeError("所有下载地址都不可用：" + ", ".join(n for n, _ in candidates))
    scored.sort(reverse=True)
    best_speed, best_name, best_url = scored[0]
    if best_speed < 50 * 1024:                     # < 50 KB/s 基本等于不可用
        raise RuntimeError(f"最快的源 {best_name} 也只有 {best_speed / 1024:.0f} KB/s，"
                           f"建议换个时间再试，或手动下载后放到 {OUT_ROOT}")
    print(f"      选用 {best_name}（{best_speed / 1048576:.2f} MB/s）开始下载")
    return _download_to_bytes(best_url, note=note)


# ----------------------------------------------------------------------------
# 1) DEMAND：每个场景一个 zip，直接解出 wav
# ----------------------------------------------------------------------------
def fetch_zenodo_links() -> dict[str, str]:
    with _urlopen_retry(ZENODO_API, timeout=60) as resp:
        data = json.load(resp)
    return {f["key"]: f["links"]["self"] for f in data.get("files", [])}


def download_demand(scenes: list[str], fs: int = 16000, out_root: str = OUT_ROOT) -> tuple[int, float]:
    print("DEMAND：获取文件列表（Zenodo API）…")
    links = fetch_zenodo_links()
    total_files, total_sec = 0, 0.0
    for scene in scenes:
        desc, fname = SCENES[scene]
        dst = os.path.join(out_root, "DEMAND_" + scene)
        n_have = _count_wavs(dst)
        if n_have:
            print(f"  DEMAND_{scene:<9} {desc:<12} 已存在 {n_have} 个文件，跳过")
            total_files += n_have
            continue
        url = links.get(fname)
        if not url:
            print(f"  {scene:<9} 数据集里没有 {fname}，跳过")
            continue
        print(f"  DEMAND_{scene:<9} {desc:<12} 下载中（断点续传）…")
        try:
            buf = _download_resumable(url, note=desc, retries=10, expect_zip=True)
        except Exception as exc:
            print(f"      失败：{str(exc)[:90]}（可重跑本命令续下）")
            continue
        n = 0
        try:
            zf = zipfile.ZipFile(buf)
        except Exception as exc:
            print(f"      压缩包损坏：{str(exc)[:70]}（可重跑本命令重下）")
            continue
        with zf:
            for name in zf.namelist():
                if not name.lower().endswith(".wav"):
                    continue
                y, sr = _read_audio_bytes(zf.read(name))
                y = _resample(audio_io.to_mono(y), sr, fs)
                _write_clip(dst, os.path.splitext(os.path.basename(name))[0], y, fs)
                total_sec += len(y) / fs
                n += 1
        print(f"      -> {n} 个 wav（{desc}）")
        total_files += n
    _note_source("DEMAND", "CC BY 4.0", "https://zenodo.org/records/1227121", total_files, total_sec)
    return total_files, total_sec


# ----------------------------------------------------------------------------
# 2) MUSAN：流式解压 tar.gz，只保留 noise/ 下的音频，取够即停
# ----------------------------------------------------------------------------
class _CountingReader(io.RawIOBase):
    """给底层流加字节计数，方便显示进度（MUSAN 流式下载用）。"""

    def __init__(self, src):
        self.src = src
        self.n = 0

    def readable(self) -> bool:
        return True

    def readinto(self, b):
        data = self.src.read(len(b))
        b[:len(data)] = data
        self.n += len(data)
        return len(data)


class _GzipStream(io.RawIOBase):
    """把 HTTP 响应 + zlib 解压包装成可 read() 的对象，供 tar 顺序解析。"""

    def __init__(self, src, dec):
        self.src = src
        self.dec = dec
        self.buf = b""

    def readable(self) -> bool:
        return True

    def read(self, size=-1):
        if size is None or size < 0:
            out = [self.buf]
            self.buf = b""
            while True:
                chunk = self.src.read(1 << 20)
                if not chunk:
                    break
                out.append(self.dec.decompress(chunk))
            return b"".join(out)
        while len(self.buf) < size:
            chunk = self.src.read(1 << 20)
            if not chunk:
                break
            self.buf += self.dec.decompress(chunk)
        out, self.buf = self.buf[:size], self.buf[size:]
        return out


def _iter_tar(stream, max_bytes: int | None = None):
    """顺序解析 tar 流，逐个 yield (文件名, 内容)。"""
    block = stream.read(512)
    consumed = 512
    while len(block) == 512 and any(block):
        name = block[0:100].split(b"\0")[0].decode("utf-8", "replace")
        try:
            size = int(block[124:136].split(b"\0")[0].strip() or b"0", 8)
        except ValueError:
            size = 0
        typeflag = block[156:157]
        padded = (size + 511) // 512 * 512
        if typeflag in (b"0", b"\0", b"") and size > 0:
            remaining = size
            chunks = []
            while remaining > 0:
                part = stream.read(min(1 << 20, remaining))
                if not part:
                    break
                chunks.append(part)
                remaining -= len(part)
                consumed += len(part)
            rest = padded - size
            if rest > 0:
                stream.read(rest)
                consumed += rest
            yield name, b"".join(chunks)
        else:
            stream.read(padded)
            consumed += padded
        if max_bytes and consumed > max_bytes:
            return
        block = stream.read(512)
        consumed += 512


def download_musan(fs: int = 16000, out_root: str = OUT_ROOT, max_clips: int = 1500,
                   clip_sec: float = 10.0, max_download_gb: float = 12.0) -> tuple[int, float]:
    """MUSAN 的 noise 部分：流式解压，累计到 max_clips 个片段就停止下载。"""
    dst = os.path.join(out_root, "MUSAN_noise")
    n_have = _count_wavs(dst)
    if n_have:
        print(f"  MUSAN_noise 已存在 {n_have} 个文件，跳过（删掉目录可重新下载）")
        return n_have, 0.0

    print(f"  MUSAN 流式获取中（目标 {max_clips} 个噪声片段，取够即停止下载）…")
    n, total_sec = 0, 0.0
    counted = None
    try:
        req = urllib.request.Request(MUSAN_URL, headers={"Range": "bytes=0-"})
        with urllib.request.urlopen(req, timeout=180) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            counted = _CountingReader(resp)
            stream = io.BufferedReader(counted, buffer_size=1 << 20)
            gz = _GzipStream(stream, zlib.decompressobj(16 + zlib.MAX_WBITS))

            for name, data in _iter_tar(gz, max_bytes=int(max_download_gb * 1024 ** 3)):
                low = name.lower()
                if not low.startswith("noise/") or not low.endswith(NOISE_EXTS):
                    continue
                try:
                    y, sr = _read_audio_bytes(data)
                except Exception:
                    continue
                y = _resample(audio_io.to_mono(np.asarray(y, dtype=np.float32)), sr, fs)
                step = int(clip_sec * fs)
                stem = os.path.splitext(os.path.basename(name))[0]
                for k in range(max(1, len(y) // step)):
                    seg = y[k * step:(k + 1) * step]
                    if len(seg) < fs:                    # 丢掉不足 1 秒的尾巴
                        continue
                    _write_clip(dst, f"{stem}_{k:03d}", seg, fs)
                    total_sec += len(seg) / fs
                    n += 1
                if n and n % 200 < 8:
                    print(f"\r      归档已读取 {counted.n / 1048576:7.1f} MB，"
                          f"已提取 {n} 个片段…", end="", flush=True)
                if n >= max_clips:
                    print(f"\n      已取够 {n} 个片段"
                          f"（归档数据约 {counted.n / 1048576:.0f} MB，提前停止下载）")
                    break
    except Exception as exc:
        print(f"\n      MUSAN 下载中断：{exc}")
    print(f"      -> {n} 个噪声片段（{total_sec / 60:.1f} 分钟）")
    _note_source("MUSAN(noise)", "CC BY 4.0", "https://www.openslr.org/17/", n, total_sec)
    return n, total_sec


# ----------------------------------------------------------------------------
# 3) ESC-50：zip 直接解出 wav
# ----------------------------------------------------------------------------
def download_esc50(fs: int = 16000, out_root: str = OUT_ROOT, max_clips: int = 2000) -> tuple[int, float]:
    dst = os.path.join(out_root, "ESC50")
    n_have = _count_wavs(dst)
    if n_have:
        print(f"  ESC50 已存在 {n_have} 个文件，跳过")
        return n_have, 0.0
    print("  ESC-50 开始（约 600 MB，先测速再选最快源）…")
    n, total_sec = 0, 0.0
    buf = _download_first_working(ESC50_URLS, note="ESC-50")
    with zipfile.ZipFile(buf) as z:
        for name in z.namelist():
            if not name.lower().endswith(".wav"):
                continue
            y, sr = _read_audio_bytes(z.read(name))
            y = _resample(audio_io.to_mono(np.asarray(y, dtype=np.float32)), sr, fs)
            _write_clip(dst, os.path.splitext(os.path.basename(name))[0], y, fs)
            total_sec += len(y) / fs
            n += 1
            if n >= max_clips:
                break
    print(f"      -> {n} 个环境声片段（{total_sec / 60:.1f} 分钟）")
    _note_source("ESC-50", "CC BY-NC 3.0（非商用）", "https://github.com/karolpiczak/ESC-50", n, total_sec)
    return n, total_sec


def download_ms_snsd(fs: int = 16000, out_root: str = OUT_ROOT, max_clips: int = 2000,
                     clean_dir: str | None = None) -> tuple[int, float]:
    """MS-SNSD（微软）：噪声库进噪声池，干净语音直接补进训练语料。

    这个数据集的价值在于它**同时提供**干净语音与噪声，正好补上本项目
    "干净语音只有 25 秒"的短板。
    """
    dst = os.path.join(out_root, "MS_SNSD_noise")
    n_have = _count_wavs(dst)
    if n_have:
        print(f"  MS_SNSD 已存在 {n_have} 个噪声文件，跳过")
        return n_have, 0.0
    print("  MS-SNSD 开始（数百 MB，先测速再选最快源）…")
    buf = _download_first_working(SNSD_URLS, note="MS-SNSD")
    n_noise, n_clean, total_sec = 0, 0, 0.0
    clean_dir = clean_dir or os.path.join(HERE, "recordings")
    with zipfile.ZipFile(buf) as z:
        names = z.namelist()
        for name in names:
            low = name.lower()
            if not low.endswith(".wav"):
                continue
            # 按路径关键字分流：噪声 -> 噪声库；干净语音 -> 训练语料
            parts = low.split("/")
            folder = "/".join(parts[1:-1]) if len(parts) > 2 else ""
            is_noise = ("noise" in folder) and ("clean" not in folder)
            is_clean = ("clean" in folder) and ("noise" not in folder.replace("clean", ""))
            try:
                y, sr = _read_audio_bytes(z.read(name))
            except Exception:
                continue
            y = _resample(audio_io.to_mono(np.asarray(y, dtype=np.float32)), sr, fs)
            stem = os.path.splitext(os.path.basename(name))[0]
            if is_noise and n_noise < max_clips:
                _write_clip(dst, f"{folder.replace('/', '_')}_{stem}".strip("_"), y, fs)
                n_noise += 1
            elif is_clean and n_clean < max_clips:
                _write_clip(clean_dir, f"ms_snsd_{stem}", y, fs)
                n_clean += 1
            total_sec += len(y) / fs
    print(f"      -> 噪声 {n_noise} 个（进噪声库），干净语音 {n_clean} 个（进 recordings/，训练时会用上）")
    _note_source("MS-SNSD", "MIT（微软开放数据集）", "https://github.com/microsoft/MS-SNSD", n_noise, total_sec)
    return n_noise, total_sec


# ----------------------------------------------------------------------------
# 统计与建议
# ----------------------------------------------------------------------------
def summarize(out_root: str = OUT_ROOT) -> None:
    if not os.path.isdir(out_root):
        print("\n噪声库还是空的。")
        return
    groups: dict[str, int] = {}
    total_bytes = 0
    for root, _dirs, files in os.walk(out_root):
        wavs = [f for f in files if f.lower().endswith(".wav")]
        if wavs:
            groups[os.path.relpath(root, out_root)] = len(wavs)
            total_bytes += sum(os.path.getsize(os.path.join(root, f)) for f in wavs)
    if not groups:
        print("\n噪声库还是空的。")
        return
    print("\n=== 当前噪声库 ===")
    for k in sorted(groups):
        print(f"  {k:<24} {groups[k]:>5} 个片段")
    print(f"  合计 {sum(groups.values())} 个片段，{total_bytes / 1048576:.0f} MB，覆盖 {len(groups)} 组")
    if len(groups) >= 3:
        print("\n覆盖已达标（≥3 组），建议训练命令：")
        print("  python train_model.py --clean ..\\data.wav --noise-dir models\\noise_bank\\real "
              "--file-noise-ratio 0.6")
    else:
        print("\n覆盖偏少（<3 组）：真实噪声太窄反而可能让模型变差，建议再下几个场景。")
    if os.path.isfile(PROVENANCE):
        with open(PROVENANCE, encoding="utf-8") as f:
            print("\n数据来源与许可：")
            for k, v in json.load(f).items():
                print(f"  {k:<16} {v['license']:<24} {v['url']}")


def main() -> int:
    ap = argparse.ArgumentParser(description="下载权威噪声数据集（DEMAND / MUSAN / ESC-50）")
    ap.add_argument("--dataset", choices=["demand", "musan", "esc50", "ms_snsd", "all"], default="demand")
    ap.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES, help=f"DEMAND 场景，默认 {DEFAULT_SCENES}")
    ap.add_argument("--all-scenes", action="store_true", help="下载 DEMAND 全部 17 个场景")
    ap.add_argument("--list", action="store_true", help="列出 DEMAND 全部场景后退出")
    ap.add_argument("--summary", action="store_true", help="只看当前噪声库统计，不下载")
    ap.add_argument("--out", default=OUT_ROOT, help="输出目录")
    ap.add_argument("--max-clips", type=int, default=1500, help="MUSAN 最多提取的噪声片段数")
    ap.add_argument("--fs", type=int, default=16000, help="目标采样率")
    args = ap.parse_args()

    if args.list:
        print(f"{'场景名':<10}{'说明':<18}16k 文件")
        for k, (desc, f) in SCENES.items():
            print(f"{k:<10}{desc:<18}{f}")
        print(f"\n默认：{' '.join(DEFAULT_SCENES)}")
        return 0

    out = os.path.abspath(args.out)
    if args.summary:
        summarize(out)
        return 0
    scenes = list(SCENES) if args.all_scenes else args.scenes
    bad = [s for s in scenes if s not in SCENES]
    if bad:
        print(f"未知场景：{bad}（--list 查看可用场景）", file=sys.stderr)
        return 2

    which = ["demand", "musan", "esc50", "ms_snsd"] if args.dataset == "all" else [args.dataset]
    for name in which:
        print(f"\n########## {name.upper()} ##########")
        try:
            if name == "demand":
                download_demand(scenes, fs=args.fs, out_root=out)
            elif name == "musan":
                download_musan(fs=args.fs, out_root=out, max_clips=args.max_clips)
            elif name == "esc50":
                download_esc50(fs=args.fs, out_root=out)
            else:
                download_ms_snsd(fs=args.fs, out_root=out)
        except Exception as exc:
            print(f"  {name} 失败：{exc}")

    summarize(out)
    print("\n下一步：")
    print("  python check_corpus.py                       # 确认噪声被识别")
    print("  python train_model.py --noise-dir models\\noise_bank\\real --file-noise-ratio 0.6")
    return 0


if __name__ == "__main__":
    sys.exit(main())
