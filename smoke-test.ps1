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
& $Python scripts\prepare_phm2008.py synthetic --out data\phm2008_sample\train_FD001.txt --units 12 --cycles 180 --sensors 6 --seed 2027
& $Python scripts\compare_phm2008.py --data-root data\phm2008_sample
& $Python scripts\generate_figures.py
Pop-Location

# 2. Benchmark the model produced by the first project. PYTHONPATH is set so the
# edgebench package can be executed directly from source without installation.
Push-Location "$PSScriptRoot\edgebench"
$env:PYTHONPATH = "src"
& $Python -m edgebench run --model ..\tinyml-predictive-maintenance\artifacts\model.json --input-size 10 --repeat 100 --out runs\tpm-model.json
& $Python -m edgebench report --input runs\tpm-model.json --out runs\tpm-model.md
Pop-Location

Write-Host "Smoke test complete."
