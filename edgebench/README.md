# EdgeBench

Small benchmarking and reporting tool for embedded-Linux edge AI deployment.

It is intentionally lightweight so it can run on a laptop today and later move
to Raspberry Pi, Jetson, or other Linux boards.

## Quick Start

```bash
cd edgebench
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python -m edgebench run --builtin centroid --repeat 200 --out runs/demo.json
python -m edgebench report --input runs/demo.json --out runs/demo.md
```

On Windows you can also run:

```powershell
.\setup.ps1
```

You can also benchmark the predictive-maintenance JSON model:

```bash
python -m edgebench run ^
  --model ..\tinyml-predictive-maintenance\artifacts\model.json ^
  --input-size 10 ^
  --repeat 500 ^
  --out runs/tpm-model.json
```

## Metrics

- average latency
- p50, p95, p99 latency
- throughput
- model file size
- backend metadata
- run configuration

## Internship Value

This project demonstrates the deployment side of embedded AI:

- benchmark methodology: warmup, repeat count, percentile latency
- CLI design
- reproducible JSON reports
- portable Linux-friendly code
- ready to extend with ONNX Runtime, TFLite, TensorRT, or OpenVINO
