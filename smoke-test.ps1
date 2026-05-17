param(
  # Allows this script to run with either the system Python or a project venv.
  # Example:
  #   .\smoke-test.ps1 -Python "E:\linux\tinyml-predictive-maintenance\.venv\Scripts\python.exe"
  [string]$Python = "python"
)

# Stop immediately if any command fails. This makes the smoke test useful in CI
# or before recording a project demo because failures are not hidden.
$ErrorActionPreference = "Stop"

# 1. Train the predictive-maintenance model and run the simulated node.
# The short duration keeps verification fast while still producing normal and
# faulty telemetry windows.
Push-Location "$PSScriptRoot\tinyml-predictive-maintenance"
& $Python scripts\train_model.py --out artifacts\model.json --windows 120
& $Python scripts\run_simulated_node.py --model artifacts\model.json --duration 4
& $Python scripts\run_simulated_node.py --model artifacts\model.json --source csv --input data\examples\vibration_demo.csv --telemetry runs\csv_telemetry.jsonl
& $Python scripts\evaluate_model.py --model artifacts\model.json --windows-per-state 40
& $Python scripts\fixed_point_report.py --model artifacts\model.json --windows-per-state 20
& $Python scripts\export_model_to_c.py --model artifacts\model.json --out firmware\model_params.h --fixed-out firmware\model_params_fixed.h
& $Python scripts\mcu_resource_report.py --model artifacts\model.json
& $Python scripts\prepare_phm2008.py synthetic --out data\phm2008_sample\train_FD001.txt --units 12 --cycles 180 --sensors 6 --seed 2027
& $Python scripts\compare_phm2008.py --data-root data\phm2008_sample
& $Python scripts\generate_figures.py
Pop-Location

# 2. Run the embedded Linux application-layer demos. They intentionally use the
# same Python venv as the TinyML project so the whole portfolio remains easy to
# verify on one laptop.
Push-Location "$PSScriptRoot\edge-ai-maintenance-gateway"
$env:PYTHONPATH = "src"
& $Python scripts\run_gateway_demo.py
Pop-Location

Push-Location "$PSScriptRoot\edge-audio-anomaly-service"
$env:PYTHONPATH = "src"
& $Python scripts\run_audio_demo.py
Pop-Location

Push-Location "$PSScriptRoot\edge-vision-inspection"
$env:PYTHONPATH = "src"
& $Python scripts\run_vision_demo.py
Pop-Location

# 3. Benchmark the model produced by the first project. PYTHONPATH is set so the
# edgebench package can be executed directly from source without installation.
Push-Location "$PSScriptRoot\edgebench"
$env:PYTHONPATH = "src"
& $Python -m edgebench run --model ..\tinyml-predictive-maintenance\artifacts\model.json --input-size 10 --repeat 100 --out runs\tpm-model.json
& $Python -m edgebench report --input runs\tpm-model.json --out runs\tpm-model.md
Pop-Location

Write-Host "Smoke test complete."
