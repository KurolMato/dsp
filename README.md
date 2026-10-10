# 语音处理与音乐分离系统

Python / Tkinter 桌面应用，包含语音录制、时域/频谱分析、端点检测、回声、倒放、四种降噪方案，以及音乐人声/伴奏分离和分轨混音。

功能及修复详见 [2026-10-05 更新说明](docs/更新说明_2026-10-05.md)。历史说明中的本机备份、测试输出和下载缓存不随 Git 仓库分发。

2026-10-10 新增「音效」页：可调房间混响，以及 WPE 去混响。使用方法及验证见 [混响更新说明](docs/更新说明_2026-10-10_混响.md)。无需安装新依赖。

2026-10-10 已从提供的版本合并自动带阻功能：「处理」页默认勾选「谱减法自动陷波」，点击⑦执行窄带检测、按需陷波和谱减；取消勾选则使用四个频率手动带阻。详见 [版本对比与合并说明](docs/合并说明_2026-10-10_自动带阻.md)。

2026-10-10 新增独立「双人语音分离」工作区：将两人同时说话的录音分成两条匿名音轨，可同步试听、导出、取消任务和打开历史。附带六秒混合参考、两路原音及真实模型分离结果；点「加载参考混合音频」即可演示。[双人分离使用与更新说明](docs/更新说明_2026-10-10_双人语音分离.md)

双人分离使用独立 `.venv-speakers` 和 `models/speakers/`。Windows 电脑先运行 `setup_speakers_windows.bat`，再启动 `run_windows.bat`。音乐人声/伴奏分离使用主环境，两者独立。

双人分离页新增「逐轨降噪 / 噪声门」：两轨独立开关和强度，可切换原始/增强对比试听，导出同时保留原始两路。默认 GTCRN 35% 混合，噪声门关闭；每次重新处理都从原始分离结果开始。[逐轨后处理说明](docs/更新说明_2026-10-10_逐轨后处理.md)

左侧「音效」→ 选择原信号或当前处理结果 →「添加混响」/「去混响」→ 试听或导出。添加混响后会自动选择当前结果，便于继续去混响。音效页处理语音工作区的单声道信号；算法接口支持单/双声道。输出保留原时长，超出原文件末尾的混响尾音截断。

## 启动

Windows 电脑安装 **64 位 Python 3.12** 后，首次使用执行：

```powershell
git clone https://github.com/KurolMato/dsp.git
cd dsp
.\setup_music_windows.bat
# 需要双人语音分离时，再执行：
.\setup_speakers_windows.bat
.\run_windows.bat
```

以后直接双击 `run_windows.bat`。已有项目目录可以直接使用，不需要再克隆一份。

**换电脑不要携带虚拟环境作为可运行依赖。** `.venv` / `.venv-speakers` 的 `pyvenv.cfg` 会记录创建时的 Python 安装位置，复制后可能出现 `No Python at ...`。新启动器已加入检测：发现失效环境后改名保留，使用目标电脑的 64 位 Python 3.12 重建；模型和音频保留。详见 [换电脑启动修复说明](docs/更新说明_2026-10-10_换电脑启动.md)。

在其他 Windows 电脑上，安装 64 位 Python 3.12，先运行 `setup_music_windows.bat`，双人分离再运行 `setup_speakers_windows.bat`，最后运行 `run_windows.bat`。三个脚本均通过同目录 `bootstrap_windows.ps1` 检测/重建环境；需要一起复制。安装依赖需联网，模型已经带上时无需重新下载模型。

```powershell
# 在克隆后的项目根目录执行；依赖只安装到项目环境
py -3.12 -m venv .venv
.venv\Scripts\python.exe -X utf8 -m pip install "torch>=2.7.1" --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -X utf8 -m pip install -r requirements-music.txt
.venv\Scripts\python.exe -X utf8 check_env.py
.venv\Scripts\python.exe -X utf8 app.py
```

只使用语音处理功能时安装 `requirements.txt` 即可，无需 PyTorch。音乐扩展使用 `pymss==2.1.7` / PyTorch / PyAV。没有安装扩展时可使用语音工作区，音乐任务会提示缺失依赖。

macOS / Linux 可用 `python3 -m venv .venv` 建环境，安装适合系统的 PyTorch 和 `requirements-music.txt`，再执行 `bash run_unix.sh`。本次实际验证平台为 Windows / Python 3.12 / CPU；其他平台和 CUDA 未验证。Linux 录音、播放还需要 PortAudio 系统库。

## 音乐分离

1. 切换右侧「音乐分离 / 分轨混音」，导入歌曲。
2. 选择模型和设备，点击「开始分离」。首次使用自动下载，也可以先点「下载所选模型」。
3. 使用同一进度条定位，调整各轨音量、静音或独奏；快捷切换原曲、人声、伴奏。
4. 「导出混音」保存当前音量组合；「导出音轨」保存全部分离轨道。

| 模型 | 输出 | 获取方式 |
|---|---|---|
| VR：`1_HP-UVR` | 人声、伴奏 | 首次使用下载，之后复用本机缓存 |
| RoFormer：`bs_roformer_voc_hyperacev2` | 人声、伴奏 | 首次使用下载，之后复用本机缓存 |
| HTDemucs4 | 人声、鼓、贝斯、其他；额外合成伴奏 | 首次使用下载，之后复用本机缓存 |

支持 WAV、MP3、FLAC、M4A、OGG、AIFF 等输入，保留单/双声道、原采样率和原长度。M4A 等格式通过 PyAV 解码，具体支持范围取决于编码器。

结果自动保存在 `results/music/时间戳_任务ID/`，包含浮点 WAV、`result.json` 和 `worker.log`。历史结果可重新打开。每次任务目录独立；取消任务终止处理进程，未完成目录标记为取消，不作为成功结果显示。

混音导出支持浮点 WAV 和 24 位 FLAC，并保存音量与来源 JSON。默认启用峰值保护，超过满幅时对全部声道统一降低增益。四轨混音默认不重复叠加派生伴奏和原曲。

## 语音降噪与训练

语音工作区仍支持 GTCRN、自训练 MLP、经典谱减法，以及真正的 GTCRN → MLP 顺序处理。多方案对比后可试听并选择采用哪一个结果。

新版 MLP 处理前按训练采样率重采样，处理后恢复原采样率与长度。训练使用实际噪声残差建立 IRM 标签，按素材内容分组划分训练/验证集，再切非重叠片段；不会把同一音频的重叠窗口混入两个集合。

```powershell
# 自动寻找 recordings 与示例语音，输出新版本并启用
.venv\Scripts\python.exe -X utf8 train_model.py --activate

# 可选真实噪声；每次训练保留独立版本
.venv\Scripts\python.exe -X utf8 train_model.py --noise-dir models\noise_bank\real --file-noise-ratio 0.3 --activate
```

输出位于 `models/runs/版本目录/`，包含模型、训练日志、评估报告和 `split_manifest.json`。模型页可取消训练、重载或选择旧版。当前指针保存在 `models/active_model.json`，项目内模型使用相对路径。

仓库包含 `models/denoise_mlp.npz` 和 `models/active_model.json` 中的相对路径，以及运行 GTCRN 所需的小型 ONNX 模型。各电脑自行训练的 `models/runs/` 不自动上传；当前本机训练版本仍保留。训练素材较少，不能据此声称跨说话人泛化；指标及对比条件见历史更新说明。

本机大模型、训练噪声库及实验/训练版本已另外提供为 [Releases 资源备份](https://github.com/KurolMato/dsp/releases/tag/assets-20261010)，下载和恢复方法见 [资源上传说明](docs/更新说明_2026-10-10_完整资源.md)。虚拟环境不上传，新电脑运行安装脚本重建。

真实录音没有干净参考时，界面显示高低能量对比、低能量帧下降、高能量帧保留和频谱改变量。这些是近似观察值，不能当作真实 SNR 或人声失真；静音、过短或高低能量难区分时不输出可靠估计。

## 验证与文件

```powershell
.venv\Scripts\python.exe -X utf8 smoke_test.py
.venv\Scripts\python.exe -X utf8 verify_interactions.py --plots-only
.venv\Scripts\python.exe -X utf8 verify_portable_startup.py
# 以下完整回归需要本机模型缓存及历史验证资产：
.venv\Scripts\python.exe -X utf8 verify_update.py
.venv\Scripts\python.exe -X utf8 bench_models.py --model models\denoise_mlp.npz --model models\runs\20261005_fixed_full\denoise_mlp.npz --manifest models\runs\20261005_fixed_full\split_manifest.json
```

`verify_update.py` 实际运行已缓存的 VR 模型，并使用本次 `results/verification_20261005/` 内的四轨、RoFormer、单声道输出和小型教师模型检查结果；需保留这些验证资产。测试不录音、不实际播放扬声器，使用临时文件验证导出和界面逻辑。

| 文件 | 职责 |
|---|---|
| `app.py` | 语音主界面及两个工作区 |
| `music_panel.py` | 音乐界面、任务状态、混音、历史、导出 |
| `music_worker.py` | 独立进程调用 pymss，下载/分离/保存 |
| `core/music.py` | 分离任务控制、共享播放游标、混音 |
| `core/tasks.py` | 带素材版本号的语音后台任务 |
| `core/corpus.py` | 去重、分组划分与验证清单 |
| `core/enhance.py` / `core/gtcrn.py` | 降噪、采样率恢复与模型缓存隔离 |
| `train_model.py` | 新版模型训练及版本保存 |
| `results/verification_20261005/` | 本次实际测试、模型对比、音轨和绘图 |

参考 [pymss-studio](https://github.com/pymss-project/pymss-studio) 的分离与分轨操作方式，使用 [pymss](https://github.com/pymss-project/pymss) 的公开 Python API。音乐功能在本项目中独立实现，没有复制 Studio 前端源码。

## 两台电脑同步

另一台电脑按上面的步骤克隆并安装。每次修改前执行 `git pull --ff-only`，完成后执行：

```powershell
git add .
git commit -m "说明本次修改"
git push
```

Git 同步提交后的文件；编辑中的内容不会实时同步。两台电脑同时修改同一处时，需要解决冲突后再提交。

源码、依赖清单、小型降噪模型、参考音频、个人录音和说明通过 Git 同步。`.gitignore` 继续排除虚拟环境、大模型、训练噪声库、运行结果、缓存和本机训练版本；模型和训练数据的本次快照在 Releases 下载，`git pull` 不会自动下载 Release 附件。新电脑也可从界面单独下载音乐及双人分离模型。两套虚拟环境留在本地，每台电脑自行安装依赖。

精简内容和本地保留项见 [GitHub 上传与精简说明](docs/更新说明_2026-10-10_GitHub上传.md)。
