"""dsp_py 核心算法包。

模块划分（对应原 MATLAB 项目的各处功能）：
    audio_io  —— 录音 / 播放 / 音频文件读写
    dsp       —— 分帧、短时能量、端点检测、带阻滤波器、频域分析、回声、倒放、经典降噪
    model     —— 纯 NumPy 实现的降噪神经网络（MLP + 手写反向传播 + Adam）
    data      —— 本地合成数据集（噪声库 + 随机 SNR 混合 + 理想比值掩蔽标签）
    enhance   —— 自训练 MLP 降噪推理（STFT -> 掩蔽 -> ISTFT）
    gtcrn     —— 预训练 GTCRN 降噪（ICASSP 2024，onnxruntime 推理，无需 PyTorch）
    metrics   —— 无参考降噪评估（只有带噪录音时也能量化效果）
"""

__all__ = ["audio_io", "dsp", "model", "data", "enhance", "gtcrn", "metrics"]
