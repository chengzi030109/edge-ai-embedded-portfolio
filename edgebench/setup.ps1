# Create an isolated Python environment for EdgeBench.
# This lets the benchmark tool evolve independently from other projects.
$ErrorActionPreference = "Stop"

# Create .venv in the current project directory.
python -m venv .venv

# Install using the venv's Python so package locations are predictable.
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Show the activation command for manual development after setup finishes.
Write-Host "Setup complete. Activate with:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
