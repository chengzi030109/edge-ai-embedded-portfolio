# Create an isolated Python environment for the TinyML project.
# This keeps project dependencies away from the global Python installation.
$ErrorActionPreference = "Stop"

# Create .venv in the current project directory.
python -m venv .venv

# Always use the venv's Python explicitly so the script does not accidentally
# install packages into the global interpreter.
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Show the activation command for manual development after setup finishes.
Write-Host "Setup complete. Activate with:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
