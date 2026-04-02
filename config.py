"""
Terroir AI — Central Configuration
Multi-crop QA platform: Potato + Tomato
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── App ─────────────────────────────────────────────────────────────────────
APP_TITLE    = "Terroir AI — Intelligent Crop Sorting"
APP_GEOMETRY = "1360x820"
APP_MIN_SIZE = (1100, 680)

# ── UI Theme ───────────────────────────────────────────────────────────────
APPEARANCE_MODE = "dark"
COLOR_THEME     = "green"

COLORS = {
    "bg":      "#0A0C0F",
    "panel":   "#111418",
    "card":    "#161B22",
    "border":  "#21262D",
    "accent":  "#00E676",
    "text":    "#F0F6FC",
    "muted":   "#8B949E",
    "good":    "#00E676",
    "bad":     "#FF1744",
    "warn":    "#FF9800",
    "blue":    "#2196F3",
}

# ── Pipeline thresholds ─────────────────────────────────────────────────────
SCAB_THRESHOLD = 0.10
CONF_MIN       = 0.60

# ── Camera + Threading ─────────────────────────────────────────────────────
CAMERA_ID      = 0
CAMERA_FPS     = 30
UI_FPS         = 30
INFERENCE_FPS  = 10

FRAME_QUEUE_MAX  = 4
INFER_QUEUE_MAX  = 4
RESULT_QUEUE_MAX = 12

# ── Phase 1 detection ──────────────────────────────────────────────────────
SCAN_X_PCT      = 0.55
BGS_HISTORY     = 120
BGS_VAR_THRESHOLD = 40
CASCADE_PATH    = os.path.join(BASE_DIR, "models", "potato_cascade.xml")

# ── Demo assets ────────────────────────────────────────────────────────────
DEMO_ASSETS_DIR = os.path.join(BASE_DIR, "demo_assets")

# ── Debug crops ────────────────────────────────────────────────────────────
DEBUG_MODE = True
DEBUG_DIR  = os.path.join(BASE_DIR, "debug_crops")

# ─────────────────────────────────────────────────────────────────────────────
#  MULTI-CROP CONFIGURATION
#  Each crop entry defines its Roboflow model and display label.
#  workspace: Roboflow workspace slug
#  project:   Roboflow project slug
#  version:   Model version number (int)
#  label:     Human-readable crop name shown in the UI
# ─────────────────────────────────────────────────────────────────────────────
CROPS = {
    "potato": {
        "workspace": "terroir-ai",
        "project":   "potato-detection-3et6q",
        "version":   11,
        "label":     "Potato",
        "icon":      "🥔",
    },
    "tomato": {
        "workspace": "new-tomato-detection",
        "project":   "tomato-detection-wb9kx",
        "version":   2,
        "label":     "Tomato",
        "icon":      "🍅",
    },
}

# The crop that is active on startup — change to "tomato" to default to tomato
ACTIVE_CROP = "potato"

# ── Roboflow API key (shared across all crops) ────────────────────────────
ROBOFLOW_API_KEY = "Gw0wMoxKeLyaJ32Fo1nS"
