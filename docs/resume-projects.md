# Resume Project Writeups

## 中文版

### AI + 嵌入式 Linux 工业音频异常检测服务

一句话介绍：构建了一个无硬件可运行的嵌入式 Linux 工业音频异常检测服务，支持 WAV 回放/上传、轻量音频特征、ONNX 推理后端、报警去抖、SQLite 离线缓存、FastAPI 接口和 systemd 部署。

技术栈：Python, NumPy, Pillow, FastAPI, SQLite, ONNX Runtime, pytest, systemd, GitHub Actions

简历 bullet：

- 实现工业音频异常检测服务，支持滑窗音频特征提取、centroid 异常评分、可选 ONNX Runtime 后端，并生成模型部署/延迟/一致性报告。
- 设计边缘设备事件链路，将原始异常窗口与设备报警状态分离，加入连续窗口报警去抖，使用 SQLite 持久化事件并模拟 `uploaded/ack` 离线补传机制。
- 接入 MIMII/ToyADMOS 风格数据目录，输出窗口级 Precision/Recall/F1、ROC-AUC 和混淆矩阵，支持本地 synthetic fixture 与真实数据集路径切换。

面试展开讲法：

先讲输入侧没有硬件时如何用 WAV replay 和 upload 模拟 ALSA/UDP 数据源；再讲 9 个轻量特征为什么适合边缘设备；接着讲模型后端如何从 centroid 切到 ONNX Runtime；最后讲 systemd、SQLite、healthz/metrics 和 ack 如何体现嵌入式 Linux 服务化能力。

可量化指标：

- Public-audio fixture: F1 `0.884`, ROC-AUC `0.990`
- ONNX parity: decision mismatch `0`
- Feature vector: 9 float32 values
- Default window: 0.25 s at 16 kHz

### TinyML 预测性维护系统

一句话介绍：构建了一个模拟 MCU/RTOS 预测性维护节点，覆盖振动窗口、特征提取、异常检测、CSV 回放、fixed-point 分析、C 参数导出和 PHM2008 更难数据路径。

技术栈：Python, NumPy, scikit-learn, ONNX, C header export, pytest, GitHub Actions

简历 bullet：

- 实现 TinyML 风格预测维护 pipeline，从 synthetic/CWRU/PHM2008 数据到模型评估、遥测回放、报警去抖和报告图表生成。
- 增加 INT8/定点化模拟与 C inference 参数导出，比较模型大小、延迟、误差和异常决策 mismatch。
- 编写一键 portfolio demo、CI 和 MCU 迁移文档，说明从 laptop prototype 到 FreeRTOS/CMSIS-DSP 的移植边界。

面试展开讲法：

强调这是 MCU/TinyML anchor 项目：虽然当前在 PC 上模拟，但数据窗口、特征、模型参数和 C 头文件都按 MCU 迁移路径设计。然后连接到 edge-audio 和 gateway，说明 Linux 侧如何接收、缓存和展示边缘事件。

可量化指标：

- 支持 CSV replay、fixed-point report、PHM2008 sample comparison
- 支持 C parameter export and parity checks
- GitHub Actions 覆盖 lite/full test path

## English Version

### Embedded Linux Industrial Audio Anomaly Service

One-line summary: Built a hardware-free embedded Linux audio anomaly service with WAV replay/upload, lightweight audio features, optional ONNX Runtime inference, alarm debounce, SQLite offline buffering, FastAPI contracts, and systemd deployment files.

Tech stack: Python, NumPy, Pillow, FastAPI, SQLite, ONNX Runtime, pytest, systemd, GitHub Actions

Resume bullets:

- Built an industrial audio anomaly service with sliding-window feature extraction, centroid anomaly scoring, optional ONNX Runtime backend, and model deployment latency/parity reports.
- Designed an edge event pipeline that separates raw anomaly windows from stable equipment alarm state, adds consecutive-window debounce, persists events in SQLite, and simulates offline upload acknowledgement with `uploaded/ack` fields.
- Added MIMII/ToyADMOS-style dataset support with window-level Precision/Recall/F1, ROC-AUC, and confusion-matrix reports, while keeping a local fixture for hardware-free demos.

Interview expansion:

Start with the input boundary: WAV replay and upload stand in for ALSA/UDP capture. Then explain the nine lightweight audio features, the centroid-to-ONNX backend path, and why alarm debounce matters. Finish with service details: SQLite buffering, `/healthz`, `/metrics`, upload/ack endpoints, systemd, and CI.

Quantified evidence:

- Public-audio fixture: F1 `0.884`, ROC-AUC `0.990`
- ONNX parity: decision mismatch `0`
- Feature vector: 9 float32 values
- Default window: 0.25 s at 16 kHz

### TinyML Predictive Maintenance System

One-line summary: Built a simulated MCU/RTOS predictive-maintenance node covering vibration windows, feature extraction, anomaly detection, CSV replay, fixed-point analysis, C parameter export, and harder PHM2008-style data.

Tech stack: Python, NumPy, scikit-learn, ONNX, C header export, pytest, GitHub Actions

Resume bullets:

- Implemented a TinyML predictive-maintenance pipeline from synthetic/CWRU/PHM2008-style data to evaluation, telemetry replay, alarm debounce, and static reporting.
- Added INT8/fixed-point simulation and C inference parameter export, comparing model size, latency, numerical drift, and anomaly decision mismatch.
- Built a portfolio demo, CI, and MCU migration notes describing the boundary from laptop prototype to FreeRTOS/CMSIS-DSP implementation.

Interview expansion:

Position this as the MCU/TinyML anchor project. The code runs on a PC today, but the windowing, features, model parameters, and C header export are designed around MCU migration. Then connect it to the Linux-side services that ingest, buffer, and expose edge events.

Quantified evidence:

- CSV replay, fixed-point report, PHM2008 sample comparison
- C parameter export and parity checks
- GitHub Actions lite/full test paths
