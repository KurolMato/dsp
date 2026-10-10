"""训练降噪神经网络（纯 NumPy，无需 PyTorch）。

用法示例：
    # 用仓库里自带的两个 wav 训练（推荐）
    python train_model.py --clean ..\\data.wav ..\\manba.wav --epochs 60

    # 只指定一个文件、并把噪声库导出成 wav 供试听
    python train_model.py --clean ..\\data.wav --export-noise models\\noise_bank

训练数据完全不依赖外部下载：干净语音用你自己的录音，噪声由 core/data.py
程序化合成，每个 epoch 现场随机生成新的 (带噪, 干净) 配对。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Windows 控制台默认 GBK，中文/² 等字符会抛 UnicodeEncodeError，这里强制 UTF-8
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from core import data as data_mod          # noqa: E402
from core import dsp, model as mlp         # noqa: E402
from core.corpus import default_clean_paths, split_clean_corpus
from core.enhance import Denoiser          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def _resample(x: np.ndarray, fs_out: int, fs_in: int) -> np.ndarray:
    """有理数重采样（供 GTCRN 残差特征使用）。

    注意：必须用 resample_poly 并让上/下行对称。早期版本用 FFT 法且目标长度算错，
    会让 48 kHz 信号被压成 1/3 时长，特征与波形完全错位（详见 core/gtcrn.py 注释）。
    """
    from math import gcd

    from scipy import signal as sps

    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if fs_out == fs_in:
        return x
    g = gcd(int(fs_out), int(fs_in))
    return sps.resample_poly(x.astype(np.float64), fs_out // g, fs_in // g).astype(np.float32)


# ----------------------------------------------------------------------------
# 指标
# ----------------------------------------------------------------------------
def log_spectral_distance(ref: np.ndarray, test: np.ndarray, fs: int = 16000) -> float:
    """对数谱失真（dB，越小越好），衡量频谱包络保真度。"""
    n = min(len(ref), len(test))
    sr = np.abs(dsp.stft(ref[:n])) ** 2
    st = np.abs(dsp.stft(test[:n])) ** 2
    m = min(sr.shape[1], st.shape[1])
    sr, st = sr[:, :m], st[:, :m]
    num = np.sum(sr, axis=1, keepdims=True)
    den = np.sum(st, axis=1, keepdims=True)
    ratio = np.maximum(sr, 1e-12) * np.maximum(den, 1e-12) / np.maximum(st * num, 1e-12)
    return float(np.mean(np.sqrt(np.mean((10 * np.log10(np.maximum(ratio, 1e-12))) ** 2, axis=0))))


def segmental_snr(ref: np.ndarray, test: np.ndarray, fs: int = 16000,
                  frame: float = 0.02, hop: float = 0.01) -> float:
    """分段信噪比（dB，越大越好）。"""
    fs_ = int(frame * fs)
    hs = int(hop * fs)
    n = min(len(ref), len(test))
    r, t = ref[:n], test[:n]
    idx = np.arange(fs_)[None, :] + hs * np.arange(max(1, (n - fs_) // hs + 1))[:, None]
    r, t = r[idx], t[idx]
    pr, pe = np.sum(r ** 2, axis=1), np.sum((r - t) ** 2, axis=1)
    valid = pr > 1e-6
    if not np.any(valid):
        return float("nan")
    snr = 10 * np.log10(np.maximum(pr[valid], 1e-12) / np.maximum(pe[valid], 1e-12))
    return float(np.mean(np.clip(snr, -10, 35)))


def evaluate(denoiser: Denoiser, pool: list[np.ndarray], fs: int, n: int = 8, seed: int = 123) -> dict:
    """在一批随机合成的带噪样本上评估降噪效果。"""
    rows = []
    rng = np.random.default_rng(seed)
    for i in range(n):
        clean = pool[int(rng.integers(0, len(pool)))]
        mix = data_mod.random_mixture(clean, fs, seed=int(rng.integers(0, 2 ** 31 - 1)), speech_pool=pool)
        noisy, c = mix["noisy"], mix["clean"]
        out = denoiser.enhance(noisy, fs=fs)
        l = min(len(c), len(noisy), len(out))
        rows.append({
            "kind": mix["kind"],
            "snr_in": dsp.mixing_snr(noisy[:l], c[:l]),
            "snr_out": dsp.mixing_snr(out[:l], c[:l]),
            "lsd_raw": log_spectral_distance(c[:l], noisy[:l], fs),
            "lsd_den": log_spectral_distance(c[:l], out[:l], fs),
            "ssnr_raw": segmental_snr(c[:l], noisy[:l], fs),
            "ssnr_den": segmental_snr(c[:l], out[:l], fs),
        })
    snr_in = float(np.mean([r["snr_in"] for r in rows]))
    snr_out = float(np.mean([r["snr_out"] for r in rows]))
    return {
        "snr_in_db": snr_in,
        "snr_out_db": snr_out,
        "snr_improve_db": snr_out - snr_in,
        "lsd_noisy_db": float(np.mean([r["lsd_raw"] for r in rows])),
        "lsd_denoised_db": float(np.mean([r["lsd_den"] for r in rows])),
        "ssnr_noisy_db": float(np.mean([r["ssnr_raw"] for r in rows])),
        "ssnr_denoised_db": float(np.mean([r["ssnr_den"] for r in rows])),
        "per_sample": rows,
    }


# ----------------------------------------------------------------------------
# 训练
# ----------------------------------------------------------------------------
def train(args) -> dict:
    if args.fs < 8000 or min(args.epochs, args.steps_per_epoch, args.batch_size, args.val_size, args.eval_samples) <= 0:
        raise ValueError("采样率至少为 8000，训练轮数、批次和评估样本数必须大于零")
    if args.context < 0 or args.lr <= 0 or not np.isfinite(args.lr):
        raise ValueError("上下文必须非负，学习率必须是有限正数")
    if not 0 <= args.file_noise_ratio <= 1 or not np.isfinite([args.snr_min, args.snr_max]).all() or args.snr_min > args.snr_max:
        raise ValueError("真实噪声比例或 SNR 范围无效")
    fs = args.fs
    out_dir = os.path.abspath(args.out_dir or os.path.join(HERE, "models", "runs", time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1000000:06d}"))
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, mlp.MODEL_FILE)

    clean_paths = [os.path.abspath(p) for p in args.clean]
    print("[1/5] 读取干净语音：")
    # 训练前筛选：静音误录 / 底噪过大的素材不该当"干净参考"（否则教模型保留噪声）
    from corpus_tool import prepare_clean_list
    n_before = len(clean_paths)
    clean_paths = prepare_clean_list(clean_paths)
    if not clean_paths:
        print(f"      [错误] {n_before} 个素材全部不合格（近乎静音或底噪过大），无法训练。")
        print("             请换个安静环境重录，或运行 python check_corpus.py 看体检结果。")
        raise ValueError("训练条件不满足，请查看上方日志")
    train_pool, val_pool, split_manifest = split_clean_corpus(
        clean_paths, fs=fs, val_ratio=args.val_ratio, seed=args.seed)
    pool = train_pool + val_pool
    split_path = os.path.join(out_dir, "split_manifest.json")
    with open(split_path, "w", encoding="utf-8") as stream:
        json.dump(split_manifest, stream, ensure_ascii=False, indent=2)
    print(f"      数据划分：{split_manifest['mode']}，训练 {len(train_pool)} 段 / 验证 {len(val_pool)} 段")
    if split_manifest["mode"] == "single_source_temporal":
        print("      [提示] 单来源按时间隔离，结果不代表跨说话人泛化能力")
    total_sec = sum(len(p) for p in pool) / fs
    print(f"      {len(clean_paths)} 个文件 -> {len(pool)} 段语音，共 {total_sec:.1f} 秒")
    if total_sec < 60:
        print(f"      [提示] 干净语音只有 {total_sec:.0f} 秒，偏少；"
              "在界面「① 录音」多录几分钟、或导入更多语音后重训，效果会明显提升")

    if args.export_noise:
        paths = data_mod.synthesize_noise_bank(os.path.abspath(args.export_noise), fs=fs, speech_pool=pool)
        print(f"[2/5] 噪声库已导出 {len(paths)} 个文件 -> {args.export_noise}")
    else:
        print("[2/5] 跳过噪声库导出（--export-noise 可导出 wav 试听）")

    # 真实噪声数据集（可选）：DEMAND / MUSAN / MS-SNSD / ESC-50 等目录
    noise_pool: list[np.ndarray] = []
    for d in (args.noise_dir or []):
        got = data_mod.load_noise_pool(os.path.abspath(d), fs=fs)
        print(f"      真实噪声：{d} -> {len(got)} 条"
              f"（共 {sum(len(x) for x in got) / fs:.0f} 秒）" if got
              else f"      真实噪声：{d} 未找到可用文件，已忽略")
        noise_pool.extend(got)
    if noise_pool:
        scene_dirs = set()
        for d in (args.noise_dir or []):
            base = os.path.abspath(d)
            for root, _dirs, files in os.walk(base):
                if any(f.endswith(".wav") for f in files):
                    scene_dirs.add(os.path.relpath(root, base))
        print(f"      真实噪声合计 {len(noise_pool)} 条，训练时以 "
              f"{args.file_noise_ratio:.0%} 概率替代合成噪声")
        if len(scene_dirs) < 3:
            print(f"      [提示] 真实噪声只覆盖 {len(scene_dirs)} 个场景"
                  f"（{', '.join(sorted(scene_dirs)) or '未知'}），覆盖太窄容易让模型变差；"
                  "建议再下几个场景：python download_datasets.py --scenes TCAR TMETRO STRAFFIC")
    elif not args.noise_dir:
        print("      未指定真实噪声（--noise-dir），全部使用程序化合成噪声")

    # 噪声底特征来源：默认自研的平滑最小值统计；可选 GTCRN 预训练模型的残差
    noise_provider = None
    if args.noise_prefix == "gtcrn":
        from core.gtcrn import GTCRNDenoiser
        _gtcrn = GTCRNDenoiser()
        if not _gtcrn.is_ready:
            print(f"      [警告] GTCRN 不可用（{_gtcrn.load_error}），改用内置最小值统计")
        else:
            def noise_provider(noisy, clean, _g=_gtcrn):
                """用预训练 GTCRN 的"输入−输出"残差作为噪声底（dB）。

                实测意义：自研的平滑最小值统计在合成噪声上够用，但遇到没见过的
                噪声类型会偏；GTCRN 在真实噪声（DNS3）上训练过，它的残差是更准的
                噪声底，让自训练模型也能泛化到未见噪声——这就是"用别人的模型
                当老师"的思路（知识蒸馏的轻量版）。
                """
                enhanced = _g.enhance(noisy, fs)
                residual = noisy - enhanced
                spec = dsp.stft(residual, n_fft=dsp.N_FFT, hop=dsp.HOP)
                return (20.0 * np.log10(np.maximum(np.abs(spec), 1e-10))).astype(np.float32)

            print(f"      噪声底特征改用 GTCRN 残差：{_gtcrn.describe()}")

    # 留出实验：故意排除若干合成噪声类型，用于检验对"未见噪声"的泛化能力
    kinds = tuple(k for k in data_mod.NOISE_TYPES if k not in (args.holdout_kinds or []))
    if args.holdout_kinds:
        missing = [k for k in args.holdout_kinds if k not in data_mod.NOISE_TYPES]
        if missing:
            print(f"      [警告] 未知噪声类型将被忽略：{missing}")
        print(f"      [留出实验] 训练噪声仅用 {len(kinds)} 类："
              f"{'、'.join(data_mod.NOISE_LABELS.get(k, k) for k in kinds)}")
        print(f"                 排除（留作测试）："
              f"{'、'.join(data_mod.NOISE_LABELS.get(k, k) for k in args.holdout_kinds)}")
        if not kinds:
            print("      [错误] 所有噪声类型都被排除了，无法训练")
            raise ValueError("训练条件不满足，请查看上方日志")

    ds = data_mod.MixtureDataset(train_pool, fs=fs, snr_range=(args.snr_min, args.snr_max),
                                 seed=args.seed, context=args.context, kinds=kinds,
                                 use_noise_floor=not args.mag_only, use_snr=not args.mag_only,
                                 noise_pool=noise_pool, file_noise_ratio=args.file_noise_ratio,
                                 noise_db_provider=noise_provider)
    val_ds = data_mod.MixtureDataset(val_pool, fs=fs, snr_range=(args.snr_min, args.snr_max),
                                     seed=args.seed + 999, context=args.context, kinds=kinds,
                                     use_noise_floor=not args.mag_only, use_snr=not args.mag_only,
                                     noise_pool=noise_pool, file_noise_ratio=args.file_noise_ratio,
                                     noise_db_provider=noise_provider)
    # 固定验证集：各 epoch 用同一批数据，验证损失才可比（否则噪声类型/SNR 每轮都变，曲线乱跳）
    Xv, Yv = val_ds.fixed_batch(batch_size=args.val_size, seed=20240)

    dims = [ds.input_dim] + [int(x) for x in args.hidden.split(",")] + [mlp.N_BINS]
    params = mlp.init_params(dims, seed=args.seed)
    opt = mlp.Adam(params, lr=args.lr)
    feat_desc = "仅带噪对数谱" if args.mag_only else "带噪对数谱 + 噪声底估计 + 局部SNR"
    print(f"[3/5] 初始化网络：{mlp.param_summary(params)}")
    print(f"      输入 = {mlp.N_BINS} 频点 × {ds.n_streams} 类特征 × {2 * args.context + 1} 帧"
          f" = {ds.input_dim} 维（{feat_desc}，时间上下文 ±{args.context} 帧）")
    print(f"      验证集 {Xv.shape[0]} 帧（固定不变，便于比较各轮损失）")

    # 特征归一化统计：从若干批次估计（对数幅度谱的均值/标准差）
    xs = []
    for _ in range(6):
        X, _, _ = ds.batch(batch_size=args.batch_size)
        xs.append(X)
    allX = np.concatenate(xs, axis=0)
    norm = {"mean": float(allX.mean()), "std": float(allX.std())}
    print(f"      特征归一化：mean={norm['mean']:.3f}, std={norm['std']:.3f}（样本 {allX.shape[0]} 帧）")

    batches_per_epoch = max(1, args.steps_per_epoch)
    history = []
    best_val = float("inf")
    t0 = time.time()
    print(f"[4/5] 开始训练：{args.epochs} epochs × {batches_per_epoch} steps，batch={args.batch_size}")
    for ep in range(1, args.epochs + 1):
        tr_loss = 0.0
        for _ in range(batches_per_epoch):
            X, Y, _ = ds.batch(batch_size=args.batch_size)
            Xn = mlp.normalize_features(X, norm["mean"], norm["std"])
            loss, grads = mlp.backward(params, mlp.forward(params, Xn)[1], Y)
            opt.step(params, grads)
            tr_loss += loss
        tr_loss /= batches_per_epoch

        Xvn = mlp.normalize_features(Xv, norm["mean"], norm["std"])
        va_loss, _ = mlp.backward(params, mlp.forward(params, Xvn)[1], Yv)

        history.append({"epoch": ep, "train_loss": tr_loss, "val_loss": va_loss})
        flag = ""
        if va_loss < best_val:
            best_val = va_loss
            mlp.save_model(model_path, params, norm,
                           meta={"epochs": ep, "val_loss": va_loss, "dims": ",".join(map(str, dims)),
                                 "context": args.context, "use_noise_floor": int(not args.mag_only),
                                 "use_snr": int(not args.mag_only),
                                 "noise_prefix": args.noise_prefix if noise_provider else "ms",
                                 "snr_range": f"{args.snr_min}~{args.snr_max}", "fs": fs,
                                 "label_version": "irm_actual_noise_v2",
                                 "split_id": split_manifest["split_id"]})
            flag = "  <- 保存最佳模型"
        if ep % max(1, args.epochs // 20) == 0 or ep == 1 or flag:
            print(f"      epoch {ep:4d}/{args.epochs}  train={tr_loss:.4f}  val={va_loss:.4f}"
                  f"  {time.time() - t0:6.1f}s{flag}")

    # 保存日志（可供 GUI 画训练曲线）
    with open(os.path.join(out_dir, "training_log.json"), "w", encoding="utf-8") as f:
        json.dump({"history": history, "best_val_loss": best_val, "dims": dims,
                   "norm": norm, "seconds": time.time() - t0}, f, ensure_ascii=False, indent=2)

    print("[5/5] 评估降噪效果（8 条随机合成的带噪语音）：")
    den = Denoiser(model_path)
    rep = evaluate(den, val_pool, fs, n=args.eval_samples)
    print(f"      输入 SNR = {rep['snr_in_db']:6.2f} dB")
    print(f"      输出 SNR = {rep['snr_out_db']:6.2f} dB  （提升 {rep['snr_improve_db']:+.2f} dB）")
    print(f"      对数谱失真 LSD：带噪 {rep['lsd_noisy_db']:.2f} dB -> 降噪 {rep['lsd_denoised_db']:.2f} dB")
    print(f"      分段信噪比 SSNR：带噪 {rep['ssnr_noisy_db']:.2f} dB -> 降噪 {rep['ssnr_denoised_db']:.2f} dB")
    with open(os.path.join(out_dir, "eval_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)

    print(f"\n完成：模型 {model_path}\n      日志 {os.path.join(out_dir, 'training_log.json')}"
          f"\n      评估 {os.path.join(out_dir, 'eval_report.json')}")
    if args.activate:
        mlp.set_active_model_path(model_path)
    return rep


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="训练纯 NumPy 语音增强（降噪）神经网络")
    default_clean = default_clean_paths()
    p.add_argument("--clean", nargs="+", default=default_clean, help="干净语音文件（可多个）")
    p.add_argument("--activate", action="store_true", help="成功训练并评估后将新版本设为当前模型")
    p.add_argument("--fs", type=int, default=16000, help="采样率（默认 16000）")
    p.add_argument("--epochs", type=int, default=18, help="训练轮数（默认 18，约 8 分钟）")
    p.add_argument("--steps-per-epoch", type=int, default=40, help="每轮批次数（默认 40）")
    p.add_argument("--batch-size", type=int, default=8, help="每个批次包含的语音条数（默认 8）")
    p.add_argument("--lr", type=float, default=2.5e-3, help="学习率（默认 2.5e-3）")
    p.add_argument("--hidden", type=str, default="256,128", help="隐藏层宽度，逗号分隔（默认 256,128）")
    p.add_argument("--context", type=int, default=6,
                   help="时间上下文帧数（前后各 N 帧，默认 6；0 表示只用单帧，效果明显更差）")
    p.add_argument("--mag-only", action="store_true",
                   help="只用带噪对数谱作输入（不加噪声底/局部SNR特征，效果会明显变差，仅用于对比实验）")
    p.add_argument("--noise-dir", nargs="*", default=None,
                   help="真实噪声数据集目录（可多个），如 DEMAND/MUSAN/MS-SNSD 解压后的文件夹；"
                        "用 download_datasets.py 自动下载")
    p.add_argument("--file-noise-ratio", type=float, default=0.3,
                   help="使用真实噪声文件的概率（默认 0.3，其余用程序化合成噪声）。"
                        "实测：只用单一场景的少量真实噪声（如 160 秒）反而会明显变差，"
                        "建议真实噪声覆盖 3 个以上场景时再调高到 0.5")
    p.add_argument("--noise-prefix", choices=["ms", "gtcrn"], default="ms",
                   help="噪声底特征的来源：ms = 自研平滑最小值统计（默认）；"
                        "gtcrn = 预训练 GTCRN 的输入−输出残差（更准，泛化更好，但每批多一次前向）")
    p.add_argument("--holdout-kinds", nargs="*", default=None,
                   help="故意排除的合成噪声类型（用于留出实验，检验模型对未见噪声的泛化能力）。"
                        f"可选：{', '.join(data_mod.NOISE_TYPES)}")
    p.add_argument("--resample-fs", type=int, default=48000, help="旧命令兼容参数；GTCRN 现按模型原生采样率处理")
    p.add_argument("--val-size", type=int, default=8, help="固定验证集语音条数（默认 8）")
    p.add_argument("--snr-min", type=float, default=-5.0, help="训练用最低 SNR（dB）")
    p.add_argument("--snr-max", type=float, default=20.0, help="训练用最高 SNR（dB）")
    p.add_argument("--val-ratio", type=float, default=0.25, help="验证集语音段比例")
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--eval-samples", type=int, default=8, help="评估样本数")
    p.add_argument("--out-dir", type=str, default=None, help="输出目录（默认 models/runs/时间戳，保留旧版）")
    p.add_argument("--export-noise", type=str, default=None, help="把合成噪声库导出到该目录")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if not args.clean:
        print("错误：没有找到干净语音，请用 --clean 指定音频文件", file=sys.stderr)
        sys.exit(2)
    try:
        train(args)
    except (ValueError, RuntimeError) as exc:
        print(f"训练失败：{exc}", file=sys.stderr)
        sys.exit(2)
