# Edge Vision Inspection

Embedded Linux visual inspection demo for defect detection. The first version
runs without a camera by generating synthetic product-surface images.

## Quick Start

```powershell
cd E:\linux\edge-vision-inspection
..\tinyml-predictive-maintenance\.venv\Scripts\python.exe scripts\run_vision_demo.py
..\tinyml-predictive-maintenance\.venv\Scripts\python.exe -m pytest -q
```

Outputs:

- `data/images/`
- `artifacts/vision_model.json`
- `reports/vision_inspection_report.md`
- `reports/annotated/`
- `reports/vision_score_distribution.png`

## API Contract

- `POST /api/v1/images/analyze`
- `GET /api/v1/images/results`
- `GET /api/v1/images/summary`

FastAPI is optional for the first local demo.

