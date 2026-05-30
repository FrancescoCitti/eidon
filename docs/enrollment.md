# Enrollment Guide

Enrollment is the process of adding a person's face embeddings to the gallery so Eidon can recognise them.

---

## Which method to use?

| Method | Accuracy | Speed | When to use |
|---|---|---|---|
| **Guided webcam** | ★★★ | ~30 s | Default — always prefer this |
| **File-based** | ★★ | ~1 min | You have existing photos but no webcam access |

The guided webcam method produces higher similarity scores (typically 0.85+) because training and inference use the same sensor, lighting, and distance.  
File-based enrollment from smartphone or camera photos typically yields 0.55–0.75 due to the domain gap.

---

## Guided webcam enrollment

```bash
python scripts/enroll_webcam.py --identity "Alice"
```

### What happens

The script opens a webcam window and walks you through five poses. When you hold the target pose steadily for ~0.7 seconds, a frame is captured automatically.

| Step | Instruction | Frames captured |
|---|---|---|
| 1 | Look straight at the camera | 2 |
| 2 | Slowly turn your head LEFT | 2 |
| 3 | Slowly turn your head RIGHT | 2 |
| 4 | Tilt your head to your LEFT shoulder | 2 |
| 5 | Tilt your head to your RIGHT shoulder | 2 |

**Total: 10 embeddings** stored per enrollment session.

### UI guide

- **Green bounding box** — face detected and pose is correct, dwell bar filling
- **Yellow bounding box** — face detected but pose is wrong
- **Green flash border** — frame captured successfully
- **Progress dots** at top — steps completed (grey → cyan → green)
- **Dwell bar** — fills as you hold the pose; triggers capture when full

### Tips for best results

- Sit at a natural distance from the webcam (40–70 cm)
- Ensure even, consistent lighting — avoid strong back-light
- Move slowly and hold each pose steady
- Remove glasses if you usually don't wear them (or enroll with them on)
- If a pose is hard to hold, use `--captures-per-step 1` to reduce the dwell requirement

### Options

```
--identity          Name to enroll (required)
--webcam            Camera device index (default: 0)
--captures-per-step Frames to capture per pose (default: 2, total: 10)
--min-score         Minimum RetinaFace detection confidence (default: 0.6)
--gallery-path      Path to the gallery .npz file
```

---

## File-based enrollment

```bash
python scripts/enroll.py --identity "Bob" --images-dir /path/to/photos
```

### Directory format

```
photos/
├── bob_001.jpg
├── bob_002.jpg
└── bob_003.png   # JPEG, PNG, BMP all supported
```

For each image the script:
1. Detects the largest face
2. Aligns it to the ArcFace 112×112 template
3. Generates a 512-dim embedding
4. Appends it to the gallery

Images where no face is detected are skipped with a warning.

### Tips for better file-based accuracy

- Use 10–30 diverse photos (different angles, expressions, lighting)
- Avoid group photos — only the largest face is enrolled per image
- Photos taken in similar conditions to your webcam will produce better scores
- Consider re-enrolling from webcam after testing — file-based is a fallback

### Options

```
--identity      Name to enroll (required)
--images-dir    Directory of face images (required)
--min-score     Minimum detection confidence (default: 0.5)
--gallery-path  Path to the gallery .npz file
```

---

## Gallery management

```bash
# View enrolled identities and embedding counts
python scripts/manage_gallery.py list

# Remove an identity
python scripts/manage_gallery.py delete --identity "OldName"

# Rename without re-enrolling
python scripts/manage_gallery.py rename --identity "OldName" --new-name "NewName"

# File stats
python scripts/manage_gallery.py info
```

### Re-enrolling

Enrolling the same identity multiple times **adds** embeddings — it does not replace them.  
To start fresh for an identity: delete it first, then re-enroll.

```bash
python scripts/manage_gallery.py delete --identity "Alice"
python scripts/enroll_webcam.py --identity "Alice"
```

### Gallery file

The gallery is stored as a numpy `.npz` archive at `data/gallery/gallery.npz` (configurable via `EIDON_GALLERY_PATH`).

The file is updated immediately on disk after every enroll or delete operation. It is mounted as a read-write volume in Docker, so changes from the API or scripts persist correctly.

---

## Via the REST API

You can enroll identities programmatically via the API:

```bash
curl -X POST http://localhost:8000/enroll \
  -F "file=@alice.jpg" \
  -F "identity=Alice"
```

Each API call enrolls one image. For bulk enrollment, use the CLI scripts — they handle directories and multiple files efficiently.

See [api.md](api.md) for full API reference.
