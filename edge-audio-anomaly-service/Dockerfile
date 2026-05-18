# Pin the default image to a digest so builds are reproducible during review.
# Reviewers behind a domestic mirror can still override it:
#   PYTHON_IMAGE=docker.m.daocloud.io/library/python:3.12-slim docker compose up --build edge-audio
ARG PYTHON_IMAGE=python:3.12.13-slim@sha256:7026274c107626d7e940e0e5d6730481a4600ae95d5ca7eb532dd4180313fea9

# ---------------------------------------------------------------------------
# Stage 1: builder. Resolves wheels into a relocatable virtual environment so
# the final image only carries the runtime files. Keeping pip + build cache
# out of the runtime layer trims the image and keeps the supply-chain story
# easy to talk about in interviews.
# ---------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

COPY pyproject.toml requirements.txt README.md ./
COPY src ./src
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: runtime. Non-root user, healthcheck, and only the files needed at
# runtime. The venv from the builder stage is copied verbatim so the runtime
# image never installs build tooling.
# ---------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PATH="/opt/venv/bin:${PATH}"

# UID/GID are matched in docker-compose.yml so bind-mounted volumes don't end
# up root-owned on Linux hosts. The numeric values still need to exist inside
# the image so uvicorn can drop into a real account.
RUN groupadd --system --gid 1000 app \
    && useradd --system --uid 1000 --gid app --home-dir /app --shell /usr/sbin/nologin app

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY pyproject.toml requirements.txt README.md ./

# data/, artifacts/, reports/ are bind-mounted via docker-compose.yml at run
# time. Pre-creating them with the right ownership avoids a permission error
# on the very first start when the host directories are empty.
RUN mkdir -p /app/data /app/artifacts /app/reports \
    && chown -R app:app /app

USER app

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=2).status == 200 else 1)"

CMD ["python", "scripts/start_dashboard.py", "--host", "0.0.0.0", "--port", "8080"]
