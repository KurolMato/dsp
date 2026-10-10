"""冒烟测试：不启动 GUI，快速验证各算法模块是否与 MATLAB 版行为一致。

运行：
    python smoke_test.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import audio_io, data as data_mod, dsp, model as mlp  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OK, FAIL = "  [OK] ", "  [FAIL] "

# Windows 控制台默认 GBK，输出中文/特殊符号会抛 UnicodeEncodeError，这里强制 UTF-8
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def check(name: str, cond: bool, extra: str = "") -> bool:
    print((OK if cond else FAIL) + name + (f"  {extra}" if extra else ""))
    return bool(cond)


def main() -> int:
    fs = 16000
    results: list[bool] = []

    # 1. 素材
    print("\n== 1. 音频文件读取 ==")
    from core.corpus import default_clean_paths
    cand = default_clean_paths()
    cand = [os.path.abspath(p) for p in cand if os.path.isfile(p)]
    y, sr = audio_io.load_audio(cand[0])
    ym = audio_io.to_mono(y)
    results.append(check("读取音频", ym.size > 0, f"{os.path.basename(cand[0])} {sr} Hz, {ym.size / sr:.2f}s, {ym.dtype}"))

    # 2. STFT 往返一致性（关键：机器学习降噪依赖它）
    print("\n== 2. STFT / ISTFT 往返 ==")
    recon = dsp.istft(dsp.stft(ym), length=len(ym))
    err = np.abs(recon - ym)
    interior = float(err[dsp.N_FFT:-dsp.N_FFT].max())
    boundary = float(max(err[:dsp.N_FFT].max(), err[-dsp.N_FFT:].max()))
    results.append(check("内部重建误差 < 1e-6（浮点精度级）", interior < 1e-6,
                         f"max|err|={interior:.2e}"))
    results.append(check("首尾边界误差可忽略（< 1e-3，仅首尾各一帧）", boundary < 1e-3,
                         f"max|err|={boundary:.2e}（OLA 窗和在边界不足一帧所致）"))

    # 3. 端点检测（pushbutton6）
    print("\n== 3. 短时能量 + 端点检测 ==")
    det = dsp.endpoint_detect(ym, sr, energy_threshold=0.02)
    results.append(check("检测到语音段", det["detected"],
                         f"起始 {det['start_time']:.3f}s -> 结束 {det['end_time']:.3f}s，"
                         f"帧长 {det['frame_size']} 帧移 {det['frame_step']}"))
    thr = (ym ** 2).mean() * det["frame_size"]
    det2 = dsp.endpoint_detect(ym, sr, energy_threshold=thr)
    results.append(check("换阈值后结果随之变化", det2["detected"] and
                         (det2["start_sample"] != det["start_sample"] or det2["end_sample"] != det["end_sample"]),
                         f"能量自适应阈值 {thr:.2e}"))

    # 4. 带阻滤波器（pushbutton5 原实现）
    print("\n== 4. 巴特沃斯带阻滤波器（1500-2100 Hz 阻带）==")
    bs = dsp.design_bandstop(sr)
    f = bs["freqs"]
    band = (f > 1750) & (f < 1850)
    outside = ((f > 200) & (f < 1200)) | ((f > 3000) & (f < 7000))
    results.append(check("阻带内衰减 > 25 dB 且通带接近 0 dB",
                         bs["mag_db"][band].max() < -25 and bs["mag_db"][outside].max() < 1.0,
                         f"阶数 n={bs['order']}，阻带最大 {bs['mag_db'][band].max():.1f} dB"))

    # 5. 频谱 / 谱图
    print("\n== 5. 频谱与谱图 ==")
    sp = dsp.spectrum(ym, sr)
    sg = dsp.spectrogram(ym, sr)
    results.append(check("频谱点数 > 0 且频率上限 = fs/4", sp["mag"].size > 0 and abs(sp["freqs"].max() - sr / 4) < 10,
                         f"{sp['mag'].size} 点，df={sp['df']:.2f} Hz"))
    results.append(check("谱图形状 (bins, frames)", sg["db"].shape[0] == dsp.N_FFT // 2 + 1,
                         f"{sg['db'].shape}，{sg['times'][-1]:.2f}s"))

    # 6. 经典降噪（对比基线）
    print("\n== 6. 经典谱减法降噪（对比基线）==")
    clean = ym[:4 * sr]
    # 固定构造：白噪声 @ 5 dB（可复现，不随随机抽样波动）
    noise = data_mod.make_noise("white", len(clean), sr, seed=1)
    noisy = data_mod.mix_at_snr(clean, noise, 5.0)
    den = dsp.classic_denoise(noisy, sr)
    snr_in = dsp.mixing_snr(noisy, clean)
    snr_out = dsp.mixing_snr(den[:len(clean)], clean)
    results.append(check("谱减法 SNR 提升 > 3 dB（白噪声 5 dB 输入）", snr_out - snr_in > 3.0,
                         f"{snr_in:.2f} dB -> {snr_out:.2f} dB（提升 {snr_out - snr_in:+.2f} dB，"
                         "参数：平滑最小值统计噪声估计 + α=2.0 + 维纳平滑）"))

    # 7. 回声 / 倒放（pushbutton4 / 7）
    print("\n== 7. 回声与倒放 ==")
    ec = dsp.echo(ym, sr, delay=10000, a=0.3)
    results.append(check("回声长度 = 原长 + 延迟", len(ec) == len(ym) + 10000,
                         f"{len(ym)} -> {len(ec)} 采样点，0.625s 后出现 0.3 倍回声"))
    rv = dsp.reverse(ym)
    results.append(check("倒放首尾互换", abs(rv[0] - ym[-1]) < 1e-6))

    # 8. 噪声库与数据集
    print("\n== 8. 本地合成数据集 ==")
    pool = data_mod.load_clean_pool(cand, fs=fs)
    ds = data_mod.MixtureDataset(pool, fs=fs, seed=0, context=2)
    X, Y, meta = ds.batch(batch_size=4)
    n_in = ds.input_dim                      # 257 频点 × 3 类特征 × 5 帧 = 3855
    ok = X.shape[1] == n_in and X.shape[0] == Y.shape[0] and Y.shape[1] == mlp.N_BINS \
        and 0.0 <= Y.min() and Y.max() <= 1.0
    results.append(check(f"批次特征 (n_frames, {n_in}) 且掩蔽值在 [0,1]", ok,
                         f"X{X.shape} Y{Y.shape}，噪声类型示例 {meta[0]['kind']}，"
                         f"SNR {meta[0]['snr_db']:+.1f} dB"))
    kinds = []
    for k in data_mod.NOISE_TYPES:
        nz = data_mod.make_noise(k, sr, sr, seed=7, speech_pool=pool)
        kinds.append(np.isfinite(nz).all() and np.abs(nz).max() <= 1.0 + 1e-6)
    results.append(check(f"全部 {len(data_mod.NOISE_TYPES)} 种噪声可合成且幅度归一",
                         all(kinds), "、".join(data_mod.NOISE_LABELS[k] for k in data_mod.NOISE_TYPES)))

    # 9. 神经网络前向 + 反向 + 一步优化
    print("\n== 9. 神经网络前向/反向/优化 ==")
    params = mlp.init_params([n_in, 64, mlp.N_BINS], seed=0)
    Xn = mlp.normalize_features(X, X.mean(), X.std())
    out, caches = mlp.forward(params, Xn)
    loss0, grads = mlp.backward(params, caches, Y)
    opt = mlp.Adam(params, lr=5e-3)
    for _ in range(30):
        out, caches = mlp.forward(params, Xn)
        loss, grads = mlp.backward(params, caches, Y)
        opt.step(params, grads)
    results.append(check("损失下降（反向传播有效）", loss < loss0 * 0.9,
                         f"{loss0:.4f} -> {loss:.4f}，输出范围 [{out.min():.3f}, {out.max():.3f}]"))
    gnorm = float(np.sqrt(sum(float((g ** 2).sum()) for g in grads.values())))
    results.append(check("梯度有限（无 NaN/Inf）", np.isfinite(gnorm), f"|grad|={gnorm:.3e}"))

    # 10. 模型存取
    print("\n== 10. 模型保存 / 加载 ==")
    tmp = os.path.join(HERE, "models", "_smoke_test.npz")
    mlp.save_model(tmp, params, {"mean": float(X.mean()), "std": float(X.std())}, meta={"epochs": 1})
    p2, n2, m2 = mlp.load_model(tmp)
    same = all(np.allclose(params[k], p2[k]) for k in params if k.startswith(("W", "b")))
    results.append(check("保存后重新加载参数一致", same and m2.get("epochs") == 1, f"meta={m2}"))
    os.remove(tmp)

    print(f"\n===== 通过 {sum(results)}/{len(results)} 项 =====")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
