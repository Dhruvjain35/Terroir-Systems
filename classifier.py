"""
Terroir AI — Phase 2: Multi-Crop Disease & Quality Classifier
Primary:  Roboflow API (crop-specific pretrained models)
Fallback: SIFT-SVM + Otsu-CNN visual analysis (potato only)

Supported crops:
  potato — potato-detection-3et6q v11  (classes: Potato, Damaged potato, etc.)
  tomato — tomato-detection-wb9kx v2   (classes: Good, Bad, Unripe)
"""

import cv2
import numpy as np
import time
import os
import sys
import requests
import base64

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diseases import (
    get_db, get_class_map,
    POTATO_DISEASE_DB, TOMATO_QUALITY_DB,
)
from config import CROPS, ROBOFLOW_API_KEY, CONF_MIN, ACTIVE_CROP

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
CNN_PATH  = os.path.join(MODEL_DIR, "cnn_model.tflite")
SVM_PATH  = os.path.join(MODEL_DIR, "sift_svm.pkl")


# ─────────────────────────────────────────────────────────────────────────────
#  ACTIVE CROP STATE
#  Call set_active_crop("tomato") from app.py when user switches crops.
# ─────────────────────────────────────────────────────────────────────────────
_active_crop = ACTIVE_CROP


def set_active_crop(crop: str):
    """Switch the active crop globally. Call this when user changes the crop selector."""
    global _active_crop
    if crop in CROPS:
        _active_crop = crop
    else:
        raise ValueError(f"Unknown crop '{crop}'. Valid options: {list(CROPS.keys())}")


def get_active_crop() -> str:
    return _active_crop


# ─────────────────────────────────────────────────────────────────────────────
#  ROBOFLOW API CALL — crop-aware
# ─────────────────────────────────────────────────────────────────────────────
def _roboflow_url(crop: str) -> str:
    """Build the Roboflow serverless inference URL for the given crop."""
    cfg = CROPS[crop]
    return (
        f"https://serverless.roboflow.com/"
        f"{cfg['project']}/{cfg['version']}"
        f"?api_key={ROBOFLOW_API_KEY}"
    )


def _call_roboflow(crop_rgb: np.ndarray, crop: str, timeout: float = 10.0) -> list:
    """
    Send image to Roboflow and return list of prediction dicts.
    Raises on network error or empty predictions.
    """
    bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    # Ensure minimum 416×416 for reliable detection
    if h < 416 or w < 416:
        bgr = cv2.resize(bgr, (max(w, 416), max(h, 416)))
    _, buf    = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    encoded   = base64.b64encode(buf.tobytes()).decode("utf-8")
    response  = requests.post(
        _roboflow_url(crop),
        data=encoded,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=timeout,
    )
    response.raise_for_status()
    result = response.json()
    preds  = result.get("predictions", [])
    if not preds:
        raise ValueError("No predictions returned by Roboflow")
    return preds


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN CLASSIFIER CLASS
# ─────────────────────────────────────────────────────────────────────────────
class Classifier:
    """
    Crop-aware classifier.
    Usage:
        clf = Classifier()
        result = clf.classify(rgb_array)                  # uses active crop
        result = clf.classify(rgb_array, crop="tomato")   # explicit crop
    """

    def __init__(self):
        # Local fallback models (potato only — loaded lazily)
        self._cnn  = None
        self._svm  = None
        self._sift = None
        self._local_loaded = False

    def _ensure_local_models(self):
        if self._local_loaded:
            return
        self._local_loaded = True
        self._cnn  = self._load_cnn()
        self._svm, self._sift = self._load_svm()

    def _load_cnn(self):
        if not os.path.exists(CNN_PATH):
            return None
        try:
            import tflite_runtime.interpreter as tflite
            m = tflite.Interpreter(CNN_PATH)
            m.allocate_tensors()
            return m
        except ImportError:
            pass
        try:
            import tensorflow as tf
            m = tf.lite.Interpreter(CNN_PATH)
            m.allocate_tensors()
            return m
        except Exception:
            return None

    def _load_svm(self):
        sift = cv2.SIFT_create()
        if not os.path.exists(SVM_PATH):
            return None, sift
        try:
            import pickle
            with open(SVM_PATH, "rb") as f:
                return pickle.load(f), sift
        except Exception:
            return None, sift

    # ── Public entry point ────────────────────────────────────────────────────
    def classify(self, crop_rgb: np.ndarray, crop: str = None) -> dict:
        """
        Classify a crop image. Returns a result dict with keys:
            disease, confidence, action, classifier, grade, grade_label,
            description, treatment, market_value, spread_risk, notifiable,
            inference_ms, crop
        """
        t0   = time.time()
        crop = crop or _active_crop

        if crop_rgb is None or crop_rgb.size == 0:
            return self._fallback_result(crop, t0, reason="empty_image")

        # ── Primary: Roboflow API ─────────────────────────────────────────────
        try:
            return self._classify_roboflow(crop_rgb, crop, t0)
        except Exception as e:
            print(f"[Classifier] Roboflow failed for {crop}: {e} — using local fallback")

        # ── Fallback: local visual classifier (potato only) ───────────────────
        if crop == "potato":
            self._ensure_local_models()
            if self._is_structural(crop_rgb):
                r = self._sift_svm(crop_rgb, t0)
            else:
                r = self._otsu_cnn(crop_rgb, t0)
            r["inference_ms"] = (time.time() - t0) * 1000
            r["crop"] = crop
            return self._apply_action(r, crop_rgb, crop)

        # For non-potato crops with no local model, return a safe default
        return self._fallback_result(crop, t0, reason="no_local_model")

    # ── Roboflow classification ───────────────────────────────────────────────
    def _classify_roboflow(self, crop_rgb: np.ndarray, crop: str, t0: float) -> dict:
        preds     = _call_roboflow(crop_rgb, crop)
        best      = max(preds, key=lambda x: x["confidence"])
        raw_cls   = best["class"]
        class_map = get_class_map(crop)
        # Map raw class name → internal key; fall back to sanitized raw name
        disease   = class_map.get(raw_cls, raw_cls.replace(" ", "_"))
        conf      = float(best["confidence"])

        r = self._result(disease, conf, "Roboflow_API", t0, crop)
        r["inference_ms"] = (time.time() - t0) * 1000
        r["crop"] = crop
        return self._apply_action(r, crop_rgb, crop)

    # ── Structural detection heuristic ───────────────────────────────────────
    def _is_structural(self, rgb):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        sx   = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy   = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        ev   = np.sqrt(sx**2 + sy**2).var()
        dr   = (gray < 45).sum() / gray.size
        return ev > 1400 and dr > 0.15

    # ── SIFT-SVM path ─────────────────────────────────────────────────────────
    def _sift_svm(self, rgb, t0):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (384, 384))
        kp, des = self._sift.detectAndCompute(gray, None)
        if self._svm is not None and des is not None and len(des) >= 4:
            feat  = self._bovw(des)
            proba = self._svm.predict_proba([feat])[0]
            idx   = int(np.argmax(proba))
            cls   = self._svm.classes_[idx]
            return self._result(cls, float(proba[idx]), "SIFT_SVM", t0, "potato")
        return self._sift_visual(rgb, kp, t0)

    def _sift_visual(self, rgb, kp, t0):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        _, ot = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        dark    = ot.sum() / (255 * h * w)
        n_kp    = len(kp) if kp else 0
        kp_dens = n_kp / (h * w) * 1000
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        ev = np.sqrt(sx**2 + sy**2).var() / 1e6
        sc = {
            "Dry_Rot":  dark*3.5 + ev*2.0 + kp_dens*0.2,
            "Gangrene": dark*2.5 + ev*1.2 + kp_dens*0.4,
            "Healthy":  (1-dark)*3.0 + (1/(1+ev))*2.0,
        }
        total = sum(max(v, 0.01) for v in sc.values())
        pr    = {k: max(v, 0.01)/total for k, v in sc.items()}
        best  = max(pr, key=pr.get)
        vals  = sorted(pr.values(), reverse=True)
        conf  = min(0.82, 0.50 + (vals[0]-vals[1])*1.6)
        return self._result(best, conf, "SIFT_Visual", t0, "potato")

    def _bovw(self, des):
        vocab = getattr(self._svm, "vocabulary_", None)
        if vocab is None:
            return des.mean(axis=0)
        dists  = np.linalg.norm(des[:, None] - vocab[None, :], axis=2)
        assign = np.argmin(dists, axis=1)
        hist   = np.bincount(assign, minlength=len(vocab)).astype(float)
        hist  /= hist.sum() + 1e-8
        return hist

    # ── Otsu-CNN path ─────────────────────────────────────────────────────────
    def _otsu_cnn(self, rgb, t0):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        _, mask     = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, mask_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if self._cnn is not None:
            return self._tflite(rgb, t0)
        return self._cnn_visual(rgb, mask, mask_inv, t0)

    def _tflite(self, rgb, t0):
        inp  = self._cnn.get_input_details()
        out  = self._cnn.get_output_details()
        sz   = inp[0]["shape"][1:3]
        img  = cv2.resize(rgb, (sz[1], sz[0]))
        data = np.expand_dims(img, 0)
        data = data.astype(np.uint8 if inp[0]["dtype"] == np.uint8 else np.float32)
        if inp[0]["dtype"] == np.float32:
            data = data / 255.0
        self._cnn.set_tensor(inp[0]["index"], data)
        self._cnn.invoke()
        probs = self._softmax(self._cnn.get_tensor(out[0]["index"])[0].astype(np.float32))
        CNN_CLASSES = ["Healthy", "Early_Blight", "Late_Blight"]
        idx = int(np.argmax(probs))
        cls = CNN_CLASSES[idx] if idx < len(CNN_CLASSES) else "Healthy"
        return self._result(cls, float(probs[idx]), "CNN_TFLite", t0, "potato")

    def _cnn_visual(self, rgb, mask, mask_inv, t0):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        lesion_ratio = mask.sum() / (255 * h * w)
        kern  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern)
        cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        n_lesions  = len(cnts)
        areas      = [cv2.contourArea(c) for c in cnts]
        max_lesion = max(areas) / (h * w) if areas else 0.0
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_var = np.sqrt(sx**2 + sy**2).var() / (255**2)
        r_m = rgb[:,:,0].mean()/255; g_m = rgb[:,:,1].mean()/255; b_m = rgb[:,:,2].mean()/255
        brown      = r_m - g_m*0.8 - b_m*0.5
        dark_ratio = (gray < 70).sum() / (h * w)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        sat = hsv[:,:,1].mean()/255; val = hsv[:,:,2].mean()/255
        sc = {
            "Late_Blight":  dark_ratio*4.0 + lesion_ratio*2.5 + brown*2.0 + (1-val)*1.5,
            "Early_Blight": lesion_ratio*2.5 + brown*2.0 + edge_var*2.5 + n_lesions*0.12,
            "Soft_Rot":     dark_ratio*3.5 + (1-sat)*2.5 + lesion_ratio*1.5,
            "Common_Scab":  edge_var*3.0 + lesion_ratio*1.8 + n_lesions*0.10,
            "Silver_Scurf": (1-sat)*2.5 + (1-dark_ratio)*1.5 + lesion_ratio*1.0,
            "Healthy":      val*3.5 + (1-lesion_ratio)*3.0 + (1-dark_ratio)*2.5 + sat*0.5,
        }
        total = sum(max(v, 0.01) for v in sc.values())
        pr    = {k: max(v, 0.01)/total for k, v in sc.items()}
        best  = max(pr, key=pr.get)
        vals  = sorted(pr.values(), reverse=True)
        conf  = min(0.88, 0.50 + (vals[0]-vals[1])*1.8)
        r = self._result(best, conf, "CNN_Visual", t0, "potato")
        r["surface_pct"] = lesion_ratio
        r["n_lesions"]   = n_lesions
        r["max_lesion"]  = max_lesion
        return r

    # ── Action & metadata enrichment ─────────────────────────────────────────
    def _apply_action(self, r: dict, rgb: np.ndarray, crop: str) -> dict:
        disease = r["disease"]
        db      = get_db(crop)
        info    = db.get(disease, {})
        action  = info.get("action", "KEEP")

        # Potato-specific scab area check
        if crop == "potato" and disease == "Common_Scab":
            sa = r.get("surface_pct", 0.0)
            if sa == 0.0:
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                sa = mask.sum() / (255 * gray.size)
                r["surface_pct"] = sa
            thresh = info.get("scab_threshold", 0.10)
            action = "REJECT" if sa > thresh else "SECONDARY"
            r["scab_note"] = f"{sa*100:.1f}% surface (limit {thresh*100:.0f}%)"

        if r["confidence"] < CONF_MIN:
            action = "REVIEW"

        r["action"]       = action
        r["notifiable"]   = info.get("notifiable", False)
        r["spread_risk"]  = info.get("spread_risk", "")
        r["treatment"]    = info.get("treatment", "")
        r["market_value"] = info.get("market_value", "")
        r["description"]  = info.get("description", "")
        r["category"]     = info.get("category", "")
        r["grade"]        = info.get("grade", "")
        r["grade_label"]  = info.get("grade_label", "")
        return r

    # ── Result dict constructors ──────────────────────────────────────────────
    @staticmethod
    def _result(disease: str, confidence: float, classifier: str,
                t0: float, crop: str = "potato") -> dict:
        db   = get_db(crop)
        info = db.get(disease, {})
        return {
            "disease":      disease,
            "confidence":   confidence,
            "action":       info.get("action", "KEEP"),
            "classifier":   classifier,
            "notifiable":   info.get("notifiable", False),
            "surface_pct":  0.0,
            "grade":        info.get("grade", ""),
            "grade_label":  info.get("grade_label", ""),
            "description":  info.get("description", ""),
            "treatment":    info.get("treatment", ""),
            "market_value": info.get("market_value", ""),
            "spread_risk":  info.get("spread_risk", ""),
            "category":     info.get("category", ""),
            "crop":         crop,
            "inference_ms": (time.time() - t0) * 1000,
        }

    def _fallback_result(self, crop: str, t0: float, reason: str = "") -> dict:
        """Return a safe 'Healthy/Good' result when classification is impossible."""
        default_key = "Healthy" if crop == "potato" else "Good"
        r = self._result(default_key, 0.50, f"Fallback({reason})", t0, crop)
        r["action"] = "REVIEW"
        r["crop"]   = crop
        return r

    @staticmethod
    def _softmax(x):
        e = np.exp(x - x.max())
        return e / e.sum()
