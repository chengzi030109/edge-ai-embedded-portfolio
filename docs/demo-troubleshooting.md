# Demo Troubleshooting

This page records the practical issues hit while making the Docker dashboard
demo reproducible on Windows.

## Docker Hub Cannot Pull Images

Symptom:

```text
failed to fetch oauth token: Post "https://auth.docker.io/token"
```

Meaning: Docker Engine is running, but it cannot reach Docker Hub.

Fix for domestic networks:

```powershell
cd E:\linux
$env:PYTHON_IMAGE="docker.m.daocloud.io/library/python:3.12-slim"
docker compose up --build edge-audio
```

The Compose file passes `PYTHON_IMAGE` into the Dockerfile, so only the base
image source changes. The application code and service command stay the same.

## Docker Engine Is Not Running

Symptom:

```text
failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine
```

Fix:

```powershell
wsl --shutdown
```

Then reopen Docker Desktop and wait until it shows `Engine running`. After that:

```powershell
docker version
docker info
```

Both commands should show server-side Docker information before running Compose.

## Dashboard Stays On `loading...`

First check whether the API endpoints are alive:

```powershell
curl http://localhost:8080/healthz
curl http://localhost:8080/metrics
curl http://localhost:8080/api/v1/audio/summary
```

If the browser shows the page but metrics/events do not update, inspect logs:

```powershell
docker logs --tail 120 linux-edge-audio-1
```

Past root cause: FastAPI thread-pool handlers reused a SQLite connection that
was created in a different thread. The service now opens SQLite with
`check_same_thread=False` and serializes API database access with a lock.

## Upload WAV Has No Response

First confirm the upload endpoint works:

```powershell
curl -F "label=anomaly" -F "file=@E:\linux\edge-audio-anomaly-service\data\wav\anomaly\anomaly_01.wav" http://localhost:8080/api/v1/audio/upload
```

If the web page stays on `uploading and analyzing...`, open:

```powershell
docker logs --tail 120 linux-edge-audio-1
```

Past root cause: FastAPI/Pydantic in the container could not resolve
`UploadFile` because it was imported inside the app factory while annotations
were postponed. The API now exposes `UploadFile` to the module namespace and the
dashboard shows upload errors instead of waiting forever.

## `docker compose` vs `docker-compose.exe`

Docker Desktop usually provides both:

```powershell
docker compose version
docker-compose.exe version
```

Use the modern command when available:

```powershell
docker compose up --build edge-audio
```

If the current shell cannot find the plugin, use Docker Desktop's bundled path:

```powershell
$env:PATH="E:\tools\docker\Docker Desktop\resources\bin;$env:PATH"
docker-compose.exe up --build edge-audio
```

## Docker Installed On C Drive By Mistake

Do not delete Docker folders while Docker Desktop is running. First uninstall
normally from Windows Settings or:

```powershell
winget uninstall Docker.DockerDesktop
```

Then remove Docker WSL data only if you are sure no images/containers matter:

```powershell
wsl --unregister docker-desktop
wsl --unregister docker-desktop-data
```

For an E-drive install, use Docker Desktop's installer option when available:

```powershell
& "E:\tools\docker\Docker Desktop Installer.exe" install `
  --accept-license `
  --installation-dir="E:\tools\docker\Docker Desktop" `
  --backend=wsl-2
```

Set Docker's disk image location to E drive in Docker Desktop:

```text
Settings -> Resources -> Advanced -> Disk image location
```

Recommended:

```text
E:\tools\docker\docker_data
```
