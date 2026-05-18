ARG PYTHON_IMAGE=python:3.12-slim
FROM ${PYTHON_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

WORKDIR /app

# Install the lightweight API runtime first so Docker can cache dependencies
# while source files change during portfolio iteration.
COPY pyproject.toml requirements.txt ./
COPY src ./src
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY configs ./configs
COPY scripts ./scripts
COPY README.md ./

EXPOSE 8080

CMD ["python", "scripts/start_dashboard.py", "--host", "0.0.0.0", "--port", "8080"]
