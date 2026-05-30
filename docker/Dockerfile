# =============================================================================
# Stage 1 — builder
# Installs all Python dependencies into an isolated venv.
# =============================================================================
FROM python:3.12-slim AS builder

# Build tools needed to compile wheels (numpy, insightface C extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy source last so dependency layers are cached independently of code changes.
# pyproject.toml changes → reinstall all deps; src/ changes → only reinstall the package.
COPY pyproject.toml .
RUN python -m venv /venv && \
    /venv/bin/pip install --upgrade pip && \
    # Pre-install all declared dependencies before the source is present
    /venv/bin/pip install --no-cache-dir \
        insightface \
        "onnxruntime>=1.17" \
        "numpy>=1.26" \
        "fastapi>=0.110" \
        "uvicorn[standard]>=0.29" \
        "python-multipart>=0.0.9" \
        "pydantic-settings>=2.2" \
        "scikit-learn>=1.4" && \
    # Headless OpenCV — no Qt/display needed in Docker (replaces opencv-python)
    /venv/bin/pip install --no-cache-dir opencv-python-headless

# Now copy source and install the package itself (no deps re-download)
COPY src/ ./src/
RUN /venv/bin/pip install --no-cache-dir --no-deps .


# =============================================================================
# Stage 2 — runtime
# Minimal image: only the venv and application source.
# Runs as a non-root user.
# =============================================================================
FROM python:3.12-slim AS runtime

# Runtime system libraries required by OpenCV headless and ONNX Runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN groupadd --gid 1000 eidon && \
    useradd --uid 1000 --gid eidon --shell /bin/bash --create-home eidon

WORKDIR /app

# Copy the complete venv from the builder stage
COPY --from=builder /venv /venv

# Copy application source and the server entrypoint
COPY src/ ./src/
COPY scripts/serve.py ./scripts/serve.py

# Create mount-point directories with correct ownership
RUN mkdir -p models data/gallery data/raw data/processed && \
    chown -R eidon:eidon /app

USER eidon

# Make the venv the active Python environment
ENV PATH="/venv/bin:$PATH"
ENV PYTHONPATH="/app/src"
ENV PYTHONDONTWRITEBYTECODE="1"
ENV PYTHONUNBUFFERED="1"

EXPOSE 8000

# Models and data are expected to be mounted:
#   docker run -v ./models:/app/models:ro -v ./data:/app/data eidon
CMD ["python", "scripts/serve.py", "--host", "0.0.0.0"]
