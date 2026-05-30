# Eidon

Real-time face recognition built on [ArcFace](https://arxiv.org/abs/1801.07698) embeddings and [RetinaFace / SCRFD](https://arxiv.org/abs/1905.00641) detection. Runs entirely on CPU with no cloud dependency.

[![CI](https://github.com/FrancescoCitti/eidon/actions/workflows/ci.yml/badge.svg)](https://github.com/FrancescoCitti/eidon/actions/workflows/ci.yml)
[![Pages](https://github.com/FrancescoCitti/eidon/actions/workflows/pages.yml/badge.svg)](https://github.com/FrancescoCitti/eidon/actions/workflows/pages.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Rust](https://img.shields.io/badge/rust-stable-orange)
![License](https://img.shields.io/badge/license-MIT-green)

## What it does

Eidon identifies faces in a live webcam feed or a still image and matches them against a local gallery of enrolled identities using cosine similarity over 512-dimensional ArcFace embeddings.

**Two ways to use it:**

| Mode | Stack | Where data lives |
|---|---|---|
| **Browser app** | Rust compiled to WASM, onnxruntime-web | Entirely in your browser (IndexedDB) |
| **Python backend** | FastAPI REST API + OpenCV webcam | Your machine |

## Architecture

```
Input (webcam / image / uploaded photo)
        │
        ▼
SCRFD RetinaFace detector  ──► 5-point landmark alignment → 112×112 crop
        │
        ▼
ArcFace MobileNet (w600k_mbf)  ──► 512-dim L2-normalised embedding
        │
        ▼
Cosine similarity vs stored embeddings  ──► identity + confidence score
```

Models: `det_500m.onnx` (SCRFD face detector) and `w600k_mbf.onnx` (ArcFace MobileNet backbone), both from the [InsightFace buffalo\_s pack](https://github.com/deepinsight/insightface).

## Browser App

A fully client-side face recognition app. No server, no accounts, no data leaves your device.

**Stack:**
- Rust compiled to `wasm32-unknown-unknown` via [Trunk](https://trunkrs.dev/)
- [Leptos](https://leptos.dev/) reactive UI framework (CSR mode)
- [onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/) for ONNX inference (loaded from CDN)
- IndexedDB for local embedding storage via [Rexie](https://github.com/devashishdxt/rexie)

**Features:**
- Live webcam recognition with face bounding-box overlay
- Guided 5-pose camera enrollment with geometric head-pose validation (yaw + roll from SCRFD landmarks)
- Photo upload enrollment: select up to 10 images from disk
- Profiles management: view and delete stored identities
- Privacy-first: all computation is local, nothing is transmitted

### Run locally

```bash
# Prerequisites: Rust stable + wasm32 target
rustup target add wasm32-unknown-unknown

# Download Trunk prebuilt binary
curl -sL https://github.com/trunk-rs/trunk/releases/download/v0.21.7/trunk-x86_64-unknown-linux-gnu.tar.gz \
  | tar -xz -C ~/.cargo/bin/

# Download ONNX models (requires Python deps — see below)
python scripts/download_models.py

# Copy models where the dev server can find them
mkdir -p web/public/models/buffalo_s
cp models/buffalo_s/det_500m.onnx models/buffalo_s/w600k_mbf.onnx web/public/models/buffalo_s/

# Start the dev server
cd web && trunk serve
# Open http://localhost:8080
```

> The release build (`trunk build --release`) is memory-intensive and runs only in CI.
> The dev build is fine for local testing on modest hardware.

### Deploy to GitHub Pages

Push to `main`. The `pages.yml` workflow downloads models, runs `trunk build --release`, and deploys `dist/` automatically.

## Python Backend

### Install

```bash
pip install -e .
python scripts/download_models.py
```

### Live webcam

```bash
python scripts/webcam.py
# --threshold 0.45    cosine similarity cutoff (default 0.40)
# --skip-frames 2     run inference every N frames
```

### Enroll an identity

```bash
python scripts/enroll.py --name "Alice" --images path/to/photos/
```

### REST API

```bash
python scripts/serve.py --host 0.0.0.0 --port 8000
```

| Endpoint | Method | Description |
|---|---|---|
| `GET /health` | | Liveness check |
| `POST /recognize` | | Identify faces in an uploaded image |
| `POST /enroll` | | Add a new identity from an uploaded image |
| `GET /identities` | | List enrolled identities |
| `DELETE /identities/{name}` | | Remove an identity |

Full API reference: [`docs/api.md`](docs/api.md)

### Docker

```bash
docker compose -f docker/docker-compose.yml up
```

## Project Structure

```
eidon/
├── src/eidon/           Python package (detection, embedding, matching, API)
├── web/                 Rust/WASM browser frontend
│   ├── src/             Leptos app + ONNX inference bridge
│   ├── Cargo.toml
│   └── Trunk.toml
├── scripts/             CLI entry points
├── tests/               Python unit and integration tests
├── docker/              Dockerfile + compose
└── .github/workflows/   CI (lint/test) + GitHub Pages deployment
```

## Hardware

Developed on an Intel N100 (4-core, no GPU). ONNX inference uses `CPUExecutionProvider` only. A single detection + embedding pass takes roughly 80-150 ms on this hardware.

## Privacy

All facial data processed by the browser app stays on your device. No images, video frames, or biometric embeddings are transmitted to any server. Embeddings are stored exclusively in browser IndexedDB and can be deleted at any time from the Profiles section.

This design follows the principles of data minimisation and privacy by design as defined in the EU General Data Protection Regulation (Regulation (EU) 2016/679, Art. 5(1)(c) and Art. 25).

## License

MIT
