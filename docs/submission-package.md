# Submission Package

Use this page as the copy/paste source for internship applications.

## GitHub Link

https://github.com/chengzi030109/edge-ai-embedded-portfolio

## Boss / 实习僧 / 牛客一句话

AI + 嵌入式 Linux 边缘智能作品集：包含 TinyML 预测性维护、工业音频异常检测、边缘网关、视觉质检和边缘推理 benchmark。重点项目为工业音频异常检测服务，支持 Docker 一键启动、WAV 上传/回放、滑窗特征、异常检测、报警去抖、SQLite 离线缓存、FastAPI API、Prometheus 风格 metrics 和 systemd 部署。

## 中文简历 3 条 Bullet

- 构建 AI + 嵌入式 Linux 作品集，覆盖 TinyML 预测性维护、工业音频异常检测、边缘网关、视觉质检和 edge benchmark，所有项目均支持无硬件本地运行与 GitHub Actions 验证。
- 实现嵌入式 Linux 工业音频异常检测服务，支持 Docker Compose 一键运行、WAV 回放/上传、滑窗音频特征、ONNX Runtime 后端、报警去抖、SQLite 离线缓存、FastAPI 接口、`/metrics` 监控和 systemd 部署。
- 接入 MIMII/ToyADMOS 风格工业音频数据目录，输出窗口级 Precision/Recall/F1、ROC-AUC 和混淆矩阵；当前 public-audio fixture F1 `0.884`、ROC-AUC `0.990`，ONNX 推理决策 mismatch 为 `0`。

## English Resume 3 Bullets

- Built a hardware-free edge AI + embedded portfolio covering TinyML predictive maintenance, industrial audio anomaly detection, edge gateway ingestion, visual inspection, and edge benchmarking, with GitHub Actions validation.
- Implemented an embedded Linux audio anomaly service with Docker Compose startup, WAV replay/upload, sliding-window audio features, optional ONNX Runtime backend, alarm debounce, SQLite offline buffering, FastAPI endpoints, `/metrics`, and systemd deployment files.
- Added MIMII/ToyADMOS-style industrial audio evaluation with window-level Precision/Recall/F1, ROC-AUC, and confusion-matrix reports; current fixture reaches F1 `0.884`, ROC-AUC `0.990`, and ONNX decision mismatch `0`.

## 3-Minute Demo Order

1. Open the GitHub README and show the GIF, CI badge, and project overview table.
2. Run the Docker dashboard:

```powershell
cd E:\linux
$env:PYTHON_IMAGE="docker.m.daocloud.io/library/python:3.12-slim"
docker compose up --build edge-audio
```

3. Open `http://localhost:8080/dashboard`.
4. Upload `edge-audio-anomaly-service/data/wav/anomaly/anomaly_01.wav`.
5. Show that event counts, alarm windows, recent events, and `/metrics` update.
6. Close by showing `tinyml-predictive-maintenance/docs/mcu-migration.md` and explaining the MCU/TinyML side.

## Interview Focus Questions

- Why is this embedded Linux rather than only Python ML?
- Why use a lightweight centroid model before a deep audio model?
- How can ONNX Runtime replace the current backend?
- How does alarm debounce reduce false positives?
- How does SQLite simulate offline buffering and cloud acknowledgement?
- What changes when moving from WAV replay to ALSA microphone input?
- Why provide both Docker Compose and systemd deployment files?
- What are the limitations of synthetic/public-audio-shaped fixtures?

## Pinned Repo Description

Hardware-free edge AI + embedded portfolio with TinyML maintenance, Dockerized ONNX-ready audio anomaly service, SQLite buffering, FastAPI/systemd deployment, dashboard GIF, and CI-backed demos.
