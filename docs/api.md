# Eidon REST API Reference

Base URL: `http://localhost:8000`  
Interactive docs (Swagger UI): `http://localhost:8000/docs`

---

## Authentication

Authentication is **disabled by default**.  
Set `EIDON_API_KEY` in your environment or `.env` to enable it.

```bash
# .env
EIDON_API_KEY=your-secret-key
```

When enabled, all endpoints except `/health` require the key in the `X-API-Key` header:

```bash
curl -H "X-API-Key: your-secret-key" http://localhost:8000/identities
```

---

## Endpoints

### `GET /health`

Liveness check. No auth required.

**Response `200`**
```json
{
  "status": "ok",
  "gallery_identities": 2,
  "gallery_embeddings": 20,
  "model_pack": "buffalo_s",
  "version": "0.1.0"
}
```

---

### `POST /recognize`

Detect and identify all faces in an uploaded image.

**Request** — `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `file` | image | JPEG, PNG, or BMP |

**curl example**
```bash
curl -s -X POST http://localhost:8000/recognize \
  -F "file=@photo.jpg" | python3 -m json.tool
```

**Response `200`**
```json
{
  "faces": [
    {
      "identity": "Alice",
      "similarity": 0.8712,
      "matched": true,
      "bbox": { "x1": 142.3, "y1": 58.1, "x2": 298.7, "y2": 264.5 },
      "detection_score": 0.9934
    }
  ],
  "count": 1,
  "processing_time_ms": 124.6
}
```

When no face is detected, `faces` is an empty array and `count` is `0`.  
When a face is detected but not matched, `identity` is `"unknown"` and `matched` is `false`.

**Error `422`** — image could not be decoded.

---

### `POST /enroll`

Detect the largest face in an image and add it to the gallery.

**Request** — `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `file` | image | JPEG, PNG, or BMP — must contain a face |
| `identity` | string | Name to enroll this face under |

**curl example**
```bash
curl -s -X POST http://localhost:8000/enroll \
  -F "file=@alice_photo.jpg" \
  -F "identity=Alice" | python3 -m json.tool
```

**Response `200`**
```json
{
  "identity": "Alice",
  "embeddings_count": 3,
  "message": "Successfully enrolled 'Alice' (3 embedding(s) total)"
}
```

`embeddings_count` is the total stored for that identity after this call.

**Error `422`** — no face detected in image, or image could not be decoded.

---

### `GET /identities`

List all enrolled identities and their embedding counts.

**curl example**
```bash
curl -s http://localhost:8000/identities | python3 -m json.tool
```

**Response `200`**
```json
{
  "identities": [
    { "name": "Alice", "embeddings": 10 },
    { "name": "Bob",   "embeddings": 10 }
  ],
  "total": 2
}
```

---

### `DELETE /identities/{name}`

Remove an identity and all its embeddings from the gallery.

**curl example**
```bash
curl -s -X DELETE http://localhost:8000/identities/Bob | python3 -m json.tool
```

**Response `200`**
```json
{
  "identity": "Bob",
  "message": "'Bob' removed from gallery"
}
```

**Error `404`** — identity not found.

---

## Tuning recognition accuracy

The `EIDON_SIMILARITY_THRESHOLD` setting (default `0.40`) controls the cut-off for a match.

| Threshold | Effect |
|---|---|
| `0.30` | More permissive — fewer unknowns, more false matches |
| `0.40` | Default — good balance for webcam-enrolled identities |
| `0.50` | Stricter — fewer false matches, more unknowns |

Typical similarity scores:
- Same person, webcam enrolled: **0.80 – 0.95**
- Same person, photo enrolled: **0.55 – 0.75** (domain gap)
- Different person: **< 0.35**
