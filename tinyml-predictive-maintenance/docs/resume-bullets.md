# Resume Bullets

Use one version depending on the role. Do not paste all of them into the same
resume; pick the framing that best matches the internship description.

## Conservative Version

- Built a hardware-free predictive-maintenance prototype for vibration anomaly
  detection, including synthetic signal simulation, CSV replay, feature
  extraction, model evaluation, and telemetry reports.
- Implemented alarm debounce logic and JSONL telemetry fields to separate raw
  model anomalies from stable device alarm state.
- Added reproducible reports and tests for synthetic data, CWRU bearing data,
  PHM2008/C-MAPSS-shaped degradation data, and fixed-point model drift.

## AI + Embedded Version

- Developed a TinyML predictive-maintenance demo with an RTOS-style pipeline:
  sensor/replay, windowing, feature extraction, anomaly inference, alarm
  debounce, and telemetry.
- Ported the centroid anomaly detector to portable C and Q24.8 fixed-point C,
  with Python/C parity tests covering both float and integer inference paths.
- Generated MCU resource-budget reports for model parameters, sample buffers,
  feature vectors, compact telemetry payloads, and per-window inference work.

## Strong Engineering Version

- Built an end-to-end AI + embedded maintenance project that runs without
  hardware but preserves MCU migration boundaries through C inference, fixed
  window contracts, telemetry schemas, and RTOS task mapping.
- Implemented a one-command portfolio demo that trains the model, runs
  synthetic and CSV replay nodes, evaluates metrics, refreshes fixed-point and
  PHM2008 reports, and regenerates static figures.
- Designed a lightweight normal-only anomaly detector and compared deployment
  tradeoffs against heavier baselines, emphasizing model size, latency,
  portability, and quantization behavior rather than accuracy alone.

## Short Chinese Version

- 完成一个 AI + 嵌入式预测维护项目：支持振动信号仿真、CSV 回放、特征提取、异常检测、报警去抖、遥测日志和报告生成。
- 将 Python 异常检测模型迁移到 C，并实现 Q24.8 定点 C 推理；通过 Python/C parity test 验证 float 和 fixed-point 两条路径一致。
- 输出 MCU 资源预算报告，覆盖模型参数大小、采样窗口、特征向量、遥测载荷和单窗口推理操作量，形成可讲清楚的 TinyML 迁移路线。

## Interview-Friendly Project Name

Recommended resume title:

```text
TinyML Predictive Maintenance Node, Hardware-Free MCU Prototype
```

Chinese version:

```text
TinyML 预测维护节点：面向 MCU 迁移的无硬件原型
```

## Metrics To Mention Carefully

Use these numbers as a snapshot, not as permanent claims:

- Synthetic detector: F1 around `0.983`, accuracy around `0.975`.
- CWRU centroid result: F1 near `1.000`, but explain the CWRU ceiling effect.
- PHM2008/C-MAPSS-shaped sample: F1 around `0.960`, with nonzero false positive
  behavior that is more realistic for gradual degradation.
- Fixed-point report: `0` parameter-path decision mismatches and `0`
  integer-path decision mismatches in the current generated report.
- Local tests after installing E-drive TinyCC: `23 passed, 2 skipped`.

## What Not To Overclaim

- Do not say it has been deployed on a real MCU yet.
- Do not say PHM2008 real data is bundled in the repo; the repo includes a small
  PHM-shape fixture and scripts for real-data preparation.
- Do not present CWRU near-perfect metrics as proof of industrial robustness.
- Do not claim TFLite Micro deployment is implemented; it is documented as a
  future path.

The strongest honest claim is: this project demonstrates the full software
architecture and MCU migration boundary before hardware is purchased.
