# 3-Minute Demo Script

## 0:00-0:30 GitHub 首页

打开仓库首页：

https://github.com/chengzi030109/edge-ai-embedded-portfolio

讲法：

> 这是我的 AI + 嵌入式 Linux 作品集，覆盖 TinyML 预测性维护、工业音频异常检测、边缘网关、视觉质检和 edge benchmark。所有项目都可以在没有硬件的情况下本地运行，并且有 CI 验证。

指给面试官看：

- CI badge
- 项目总览表
- Edge Audio 的架构图和两张摘要图

## 0:30-1:20 Edge Audio Demo + 图

运行：

```powershell
cd E:\linux\edge-audio-anomaly-service
..\tinyml-predictive-maintenance\.venv\Scripts\python.exe scripts\run_audio_demo.py
```

讲法：

> 这个服务把 WAV 文件当作嵌入式 Linux 上的音频流，按滑窗提取 RMS、ZCR、频谱质心、频带能量等轻量特征，再用 centroid 或 ONNX 后端输出异常分数。

展示：

- `reports/audio_score_curve.png`
- `reports/public_audio_evaluation.md`
- `reports/model_deployment_report.md`

重点数字：

- `windows analyzed: 91`
- `raw anomaly windows: 35`
- `alarm windows: 33`
- public-audio fixture F1 `0.884`, ROC-AUC `0.990`
- ONNX decision mismatch `0`

## 1:20-2:10 API / SQLite / systemd

打开：

- `edge-audio-anomaly-service/src/edge_audio/api.py`
- `edge-audio-anomaly-service/src/edge_audio/storage.py`
- `edge-audio-anomaly-service/systemd/edge-audio-anomaly-service.service`

讲法：

> 我没有只做模型脚本，而是把它做成一个 Linux 服务。它支持 WAV upload、事件查询、ack、health check 和 Prometheus 风格 metrics。事件先写 SQLite，`uploaded/ack` 字段模拟断网缓存和云端确认。

点名接口：

```text
POST /api/v1/audio/upload
POST /api/v1/audio/events/ack
GET  /healthz
GET  /metrics
```

讲报警去抖：

> 模型的单窗口输出是 `is_anomaly_raw`，真正给设备/操作员看的报警是 `is_alarm`，需要连续异常窗口才进入报警，避免一次抖动就误报。

## 2:10-3:00 TinyML + 总结

打开：

- `tinyml-predictive-maintenance/README.md`
- `tinyml-predictive-maintenance/docs/mcu-migration.md`

讲法：

> TinyML 项目是 MCU 方向的 anchor，包含振动窗口、特征、fixed-point 模拟、C 参数导出和 MCU 迁移说明。Edge Audio 是嵌入式 Linux 应用层方向的 anchor，体现服务化、SQLite 缓存、API、ONNX 后端和 systemd 部署。

收尾：

> 这个作品集重点不是单个模型分数，而是完整的边缘 AI 工程链路：可回放输入、轻量特征、推理后端、报警策略、本地持久化、API/metrics、部署文件、报告和 CI。
