# Terroir AI — Intelligent Crop Sorting System

**Two-phase AI pipeline for potato disease detection on a conveyor belt.**  
Built by Dhruv Jain · Emerson High School, McKinney TX

---

## Quick Start

```bash
pip install -r requirements.txt

python main.py --sim              # Simulated conveyor (no camera)
python main.py --image photo.jpg  # Test a single image
python main.py --camera 0         # Live USB camera
python main.py --simple           # One-line output per tuber
```

---

## Architecture (Korchagin et al. 2021 — Agronomy 11:1980)

### Phase 1 — Tuber Detection
- **Viola-Jones Haar Cascade** on Sobel-preprocessed, 10x reduced frame
- Integral Image: `I(x,y) = Σ i(x',y')` → O(1) Haar feature lookup
- **IoU tracking** prevents classifying the same tuber twice across frames
- Falls back to background subtraction + contours if no cascade trained

### Phase 2 — Dynamic Disease Classification

| Route | Method | Diseases | Accuracy |
|-------|--------|----------|----------|
| 2A | SIFT-SVM (RBF) | Dry Rot, Gangrene, Structural | 80-95% |
| 2B | Otsu → CNN INT8 | Blights, Scabs, Surface | 85-97% |

**SIFT Math:** `D(x,y,σ) = (G(x,y,kσ) - G(x,y,σ)) * I(x,y)`  
**Otsu Math:** `σ²_b = ω₀ω₁(μ₀-μ₁)²`  
**SVM Math:** `min ||w||²/2  s.t. y_i(w·x_i + b) ≥ 1`

---

## Action Matrix (AHDB dataset)

| Disease | Action |
|---------|--------|
| Healthy | ✓ KEEP |
| Late Blight | ✗ REJECT |
| Early Blight | ✗ REJECT |
| Dry Rot | ✗ REJECT (SIFT-SVM) |
| Soft Rot | ✗ REJECT |
| Common Scab | B GRADE-B (>10% surface area → REJECT) |
| Silver Scurf | ✓ KEEP |
| Black Dot | ✓ KEEP |

---

## Training

```bash
# Download: https://www.kaggle.com/datasets/arjuntejaswi/plant-village
# Place in: data/train/Healthy/, data/train/Early_Blight/, data/train/Late_Blight/

python train.py --cnn             # Train CNN → INT8 TFLite
python train.py --svm             # Train SIFT-SVM
python train.py --both            # Train both
```

---

## System Requirements (Spec)
- Standard laptop CPU/GPU
- $80-$150 USB action camera + flashlight
- Conveyor speed: 0.8–1.0 m/s
- Processing target: <100ms per tuber (~100 tubers/sec)

---

## Logs
Every classification logged to `logs/session_YYYYMMDD_HHMMSS.csv`:
```
timestamp, tuber_id, session_id, disease, classifier, confidence, action, surface_pct, inference_ms, notifiable
```
