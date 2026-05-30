# Facial Recognition ML — Project Plan

**Status:** PLANNING — not yet started  
**Owner:** Francesco  
**Last updated:** 2026-05-30  
**Goal:** Production-grade real-time webcam facial recognition pipeline based on ArcFace / InsightFace, built to senior engineering standards.

---

## Context for a Resuming Agent

This project was started fresh. The user reviewed:
- FaceNet paper (arXiv:1503.03832, Google 2015) — 128-dim embeddings, triplet loss
- A 2017 Hackernoon tutorial (Cole Murray) — Dlib + TF1.x pipeline

**Decision:** Skip the outdated 2017 stack entirely. Build a modern pipeline using:
- **RetinaFace** — face detection and alignment (replaces Dlib)
- **ArcFace** (via InsightFace) — face embeddings (replaces FaceNet weights)
- **ONNX Runtime** — inference engine (fast, framework-agnostic)
- **SVM / cosine similarity** — identity classification

The user wants **production-level, senior engineer quality** code. Details of use case (webcam vs batch vs API) and GPU availability are TBD — confirm with Francesco before implementing.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT SOURCES                         │
│         Webcam  │  Image file  │  Video  │  API call    │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│               PREPROCESSING PIPELINE                     │
│  1. Frame capture / image load                          │
│  2. RetinaFace: detect faces + 5-point landmarks        │
│  3. Affine transform → 112×112 aligned crop             │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│               EMBEDDING GENERATION                       │
│  ArcFace (R100 or MobileNet for edge) via ONNX          │
│  Output: L2-normalised 512-dim vector per face          │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│               IDENTITY MATCHING                          │
│  Gallery DB (embeddings + metadata)                     │
│  Cosine similarity → threshold → identity / unknown     │
│  Optional: SVM classifier for closed-set recognition    │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│               OUTPUT / SERVING                           │
│  REST API (FastAPI)  │  CLI tool  │  Annotated frames   │
└─────────────────────────────────────────────────────────┘
```

---

## Project Structure (target)

```
facial-recognition-ML/
├── PLAN.md                          ← this file
├── README.md
├── pyproject.toml                   ← dependencies (uv / pip)
├── .env.example
├── .gitignore
│
├── src/
│   └── eidon/
│       ├── __init__.py
│       ├── config.py                ← pydantic-settings config
│       ├── pipeline.py              ← orchestrates detect → embed → match
│       │
│       ├── detection/
│       │   ├── __init__.py
│       │   └── retinaface.py        ← RetinaFace wrapper
│       │
│       ├── embedding/
│       │   ├── __init__.py
│       │   └── arcface.py           ← ArcFace ONNX wrapper
│       │
│       ├── matching/
│       │   ├── __init__.py
│       │   ├── gallery.py           ← embedding store (numpy .npz, 10-person scale)
│       │   └── matcher.py           ← cosine sim + threshold logic
│       │
│       ├── webcam/
│       │   ├── __init__.py
│       │   └── live.py              ← OpenCV capture loop, annotated frame output
│       │
│       ├── serving/
│       │   ├── __init__.py
│       │   ├── api.py               ← FastAPI app
│       │   └── schemas.py           ← Pydantic request/response models
│       │
│       └── utils/
│           ├── __init__.py
│           ├── image.py             ← image I/O helpers
│           └── logging.py           ← structured logging setup
│
├── scripts/
│   ├── enroll.py                    ← CLI: add identities to gallery
│   ├── evaluate.py                  ← CLI: benchmark on LFW / custom set
│   └── download_models.py           ← download ONNX weights
│
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_detection.py
│   │   ├── test_embedding.py
│   │   └── test_matching.py
│   └── integration/
│       └── test_pipeline.py
│
├── models/                          ← ONNX model files (gitignored)
│   ├── det_10g.onnx                 ← RetinaFace detector
│   └── w600k_mbf.onnx               ← ArcFace MobileNet backbone (CPU-optimised)
│
├── data/                            ← gitignored
│   ├── raw/                         ← original images
│   ├── processed/                   ← aligned 112×112 crops
│   └── gallery/                     ← enrolled embeddings (gallery.npz)
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── web/                             ← Phase 9: browser frontend (GitHub Pages)
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── index.html
│   └── src/
│       ├── main.ts                  ← app shell: loading screen, model progress, routing
│       ├── types.ts                 ← Detection, MatchResult, RecognitionResult (TS)
│       ├── pipeline.ts              ← detect → align → embed → match orchestrator
│       ├── app.css                  ← dark-theme responsive styles
│       ├── inference/
│       │   ├── nms.ts               ← IoU + NMS
│       │   ├── alignment.ts         ← similarity transform + Canvas 2D warpAffine
│       │   ├── detector.ts          ← SCRFD anchor decoding + onnxruntime-web
│       │   └── embedder.ts          ← ArcFace ONNX + L2 normalise
│       ├── gallery/
│       │   └── db.ts                ← IndexedDB gallery (idb)
│       ├── pages/
│       │   ├── recognize.ts         ← live webcam overlay
│       │   └── enroll.ts            ← 5-pose guided enrollment
│       └── ui/
│           └── overlay.ts           ← canvas bbox + label drawing
│
└── .github/
    └── workflows/
        ├── ci.yml                   ← lint + test on push
        └── pages.yml                ← build + deploy to GitHub Pages
```

---

## Implementation Phases

### Phase 0 — Repository Bootstrap  [x] DONE 2026-05-30
- [x] `git init` + `.gitignore` (models/, data/, __pycache__, .env)
- [x] `pyproject.toml` with full deps + ruff / mypy / pytest config
- [x] `src/eidon/config.py` — pydantic-settings, singleton `settings`, validated fields
- [x] `scripts/download_models.py` — downloads buffalo_s pack via InsightFace
- [x] `src/eidon/utils/logging.py` — JSON (prod) + coloured-text (dev) formatters
- [x] `.env.example` — documents all environment variables
- [x] `.github/workflows/ci.yml` — ruff + mypy + pytest on Python 3.12
- [x] Full package skeleton (`__init__.py` in every sub-package)
- [x] `tests/conftest.py` — shared fixtures
- [x] `.python-version` = 3.12

### Phase 1 — Detection Module  [x] DONE 2026-05-30
- [x] `src/eidon/types.py` — shared `Detection` + `RecognitionResult` dataclasses
- [x] `detection/alignment.py` — Umeyama similarity transform, `align_face()`, `ARCFACE_DST` template
- [x] `detection/retinaface.py` — `RetinaFaceDetector` wrapping InsightFace; `detect()` + `detect_largest()`
- [x] `detection/__init__.py` — clean public exports
- [x] `tests/unit/test_detection.py` — 13 unit tests (alignment + mocked detector), no model files needed

### Phase 2 — Embedding Module  [x] DONE 2026-05-30
- [x] `embedding/arcface.py` — `ArcFaceEmbedder` with `embed()` + `embed_batch()`; pure-function `_preprocess` and `_l2_normalize`
- [x] `embedding/__init__.py` — clean export
- [x] `tests/unit/test_embedding.py` — 21 unit tests: preprocessing math, L2 norm, shape/dtype contracts, cosine-similarity identity check, all error paths

### Phase 3 — Matching / Gallery  [x] DONE 2026-05-30
- [x] `src/eidon/types.py` — added `MatchResult` dataclass
- [x] `matching/gallery.py` — `Gallery`: add/remove/save/load, .npz persistence, validation
- [x] `matching/matcher.py` — `Matcher`: nearest-neighbour cosine similarity, configurable threshold, "unknown" fallback
- [x] `matching/__init__.py` — clean exports
- [x] `scripts/enroll.py` — CLI to register identities from an image directory
- [x] `tests/unit/test_matching.py` — 34 unit tests (gallery mutation, persistence round-trip, matcher logic)

### Phase 4 — Pipeline Orchestration  [x] DONE 2026-05-30
- [x] `src/eidon/pipeline.py` — `FaceRecognitionPipeline`: `recognize()`, `update_gallery()`, `from_settings()` factory
- [x] `tests/unit/test_pipeline.py` — 12 unit tests (composition, crop shape, threshold forwarding, validation)
- [x] `tests/integration/test_pipeline.py` — 7 integration tests (real gallery+matcher, full data flow, gallery hot-swap, persistence round-trip)
- [x] Added `tests/__init__.py`, `tests/unit/__init__.py`, `tests/integration/__init__.py` to fix pytest package collision

### Phase 5 — Webcam Live Mode  [x] DONE 2026-05-30
- [x] `webcam/live.py` — `LiveRecognition` loop + `annotate_frame()` + `_FPSCounter`
- [x] Per-frame: detect → align → batch-embed → match → draw bbox + name + confidence
- [x] FPS rolling-window counter overlay (top-left corner)
- [x] Graceful exit on `q` key and OS window-close button
- [x] Frame-skip: inference every N frames, cached results redrawn on all frames between
- [x] Label flip: labels near top edge automatically draw below the box instead of above
- [x] `scripts/webcam.py` — CLI entry point with --webcam / --skip-frames / --threshold flags
- [x] `tests/unit/test_webcam.py` — 24 unit tests (fps counter, annotate_frame, loop lifecycle, all with mocked cv2)

### Phase 6 — REST API  [x] DONE 2026-05-30
- [x] `serving/schemas.py` — BoundingBox, FaceResult, RecognizeResponse, EnrollResponse, IdentitiesResponse, DeleteResponse, HealthResponse
- [x] `serving/api.py` — FastAPI: GET /health, POST /recognize, POST /enroll, GET /identities, DELETE /identities/{name}
- [x] `serving/__init__.py` — exports app
- [x] Auth: X-API-Key header, enabled by FACEREC_API_KEY env var, disabled when unset
- [x] `scripts/serve.py` — uvicorn entry point with --host/--port flags
- [x] `tests/integration/test_api.py` — 20 integration tests via FastAPI TestClient + dependency injection (no models needed)
- [x] Added python-multipart + httpx2 to dependencies

### Phase 7 — Docker + CI  [x] DONE 2026-05-30
- [x] `docker/Dockerfile` — multi-stage (builder + runtime), non-root user, opencv-python-headless swap
- [x] `docker/docker-compose.yml` — volume mounts for models/ + data/, healthcheck, env-based API key
- [x] `.dockerignore` — excludes models/, data/, .venv, tests, .git from build context
- [x] `.github/workflows/ci.yml` — lint-and-test job (ruff + mypy + pytest) + docker-build job (build check, no push, GHA cache)
- [x] Fixed all ruff (43 → 0) and mypy errors across the whole codebase before locking CI

### Phase 8 — Documentation  [x] DONE 2026-05-30
- [x] `README.md` — project overview, architecture diagram, quickstart, all scripts documented
- [x] `docs/api.md` — REST API reference (endpoints, request/response examples, auth)
- [x] `docs/enrollment.md` — enrollment guide (file-based vs guided webcam)
- [x] Inline docstring audit — all public classes and functions documented
- [x] `.env.example` — covers all settings including max_workers, batch_size, det_model_name, rec_model_name

### Phase 9 — Browser Frontend (GitHub Pages, zero-cost product)  [x] DONE 2026-05-30
- **Goal:** fully client-side face recognition — no server, no cost, no privacy concerns
- **Stack:** Rust → WASM (tract-onnx for inference, Leptos for the reactive UI, Trunk as build tool)
- **Hosting:** GitHub Pages (static, free forever)
- **Models:** det_500m.onnx + w600k_mbf.onnx served from the Pages deployment via HTTP fetch
- [x] `web/src/inference/nms.rs` — IoU + NMS (pure Rust)
- [x] `web/src/inference/alignment.rs` — closed-form similarity transform + Canvas 2D warpAffine (web-sys)
- [x] `web/src/inference/detector.rs` — SCRFD anchor decoding + NMS; tract-onnx backend
- [x] `web/src/inference/embedder.rs` — ArcFace MobileNet; tract-onnx backend; L2 normalised output
- [x] `web/src/gallery.rs` — IndexedDB gallery via `rexie`; data stays on device
- [x] `web/src/pipeline.rs` — detect → align → embed → match orchestrator (mirrors Python pipeline)
- [x] `web/src/app/recognize.rs` — live webcam overlay (Leptos component)
- [x] `web/src/app/enroll.rs` — 5-pose guided enrollment with dwell-bar auto-capture (Leptos)
- [x] `web/src/app/gallery_view.rs` — gallery list / remove / clear (Leptos)
- [x] `web/src/main.rs` — app shell: async model loading, tab routing (Leptos CSR)
- [x] `web/src/styles.css` — dark theme, responsive layout
- [x] `web/Cargo.toml` — tract-onnx + Leptos + web-sys + rexie + gloo-net
- [x] `web/Trunk.toml` — relative public_url for GitHub Pages subdirectory
- [x] `.github/workflows/pages.yml` — download models → trunk build → deploy on push to main
- **Stack note:** Rust compiled to wasm32-unknown-unknown; tract-onnx runs inference natively without JS interop
- **Privacy:** gallery data and face embeddings never leave the user's device (IndexedDB)

---

## Key Technical Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Inference engine | ONNX Runtime | Framework-agnostic, fast on CPU+GPU, no TF/PyTorch required at runtime |
| Face detector | RetinaFace (det_10g) | State-of-art, included in InsightFace model zoo |
| Backbone | ArcFace MobileNet (w600k_mbf) | CPU-only machine (Intel N100) — MobileNet ~80ms/face vs R50 ~2-3s/face |
| Embedding storage | numpy .npz only | Gallery is ~10 people — faiss adds complexity with no benefit at this scale |
| API framework | FastAPI | Async, auto-docs, pydantic native |
| Config | pydantic-settings | Type-safe, .env support, 12-factor app |
| Python packaging | pyproject.toml (uv) | Modern, no setup.py |

---

## Confirmed Decisions (2026-05-30)

| Question | Answer |
|---|---|
| Use case | Real-time webcam recognition |
| Gallery scale | ~10 people — numpy .npz only, no faiss needed |
| Python versioning | `.python-version` = 3.12 (CI/GitHub); local machine runs 3.14.4 |
| GitHub | Project will be pushed to GitHub |

---

## Hardware Profile (confirmed 2026-05-30)

| Component | Detail |
|---|---|
| CPU | Intel N100, 4 cores, no hyperthreading, 3.4 GHz max |
| RAM | 15 GB total, ~5 GB available at idle |
| GPU | None (Intel UHD integrated only) |
| ONNX provider | CPUExecutionProvider |
| Max workers | 4 |

---

## Open Questions (remaining — confirm before Phase 6+)

1. **Dataset for evaluation** — LFW, or a custom private dataset?
2. **Auth / security requirements** for the REST API?

---

## Dependencies (planned)

```toml
[project]
requires-python = ">=3.11"       # local machine: 3.14.4 | CI targets: 3.12

dependencies = [
    "insightface>=0.7",
    "onnxruntime>=1.17",          # CPUExecutionProvider only (no GPU on this machine)
    "opencv-python>=4.9",
    "numpy>=1.26",
    "fastapi>=0.110",
    "uvicorn[standard]>=0.29",
    "pydantic-settings>=2.2",
    "scikit-learn>=1.4",
    # no faiss — gallery is ~10 people, numpy cosine similarity is sufficient
]

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "pytest-asyncio>=0.23",
    "httpx>=0.27",
    "ruff>=0.4",
    "mypy>=1.9",
]
```

### Python version strategy
- `.python-version` → `3.12` — used by GitHub Actions CI and pyenv if installed
- Local machine runs Python 3.14.4 (system) — use a venv to isolate deps
- `requires-python = ">=3.11"` in pyproject.toml — allows either

---

## Non-Goals (explicitly out of scope for v1)

- Training a new backbone from scratch (we use pretrained ArcFace weights)
- Anti-spoofing / liveness detection
- Age / gender / emotion attributes
- Mobile / edge deployment (can be added later with MobileNet backbone swap)
