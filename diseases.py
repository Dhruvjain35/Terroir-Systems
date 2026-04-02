"""
Terroir AI — Disease & Quality Data Dictionary
Multi-crop: Potato (AHDB / Korchagin et al. 2021) + Tomato (USDA / FAO quality standards)

Structure per entry:
    action        — KEEP | SECONDARY | REJECT | REVIEW
    severity      — 0 (none) → 4 (critical)
    category      — HEALTHY | FUNGAL | BACTERIAL | VIRAL | DEFECT | QUALITY
    spread_risk   — human-readable risk level
    market_value  — commercial value descriptor
    treatment     — recommended handling / agronomic action
    description   — one-sentence diagnostic description
    notifiable    — bool, whether legally notifiable
"""

# ─────────────────────────────────────────────────────────────────────────────
#  POTATO DISEASE DATABASE
# ─────────────────────────────────────────────────────────────────────────────
POTATO_DISEASE_DB = {

    "Healthy": {
        "action":       "KEEP",
        "classifier":   "CNN",
        "severity":     0,
        "category":     "HEALTHY",
        "spread_risk":  "None",
        "market_value": "Full primary market value",
        "treatment":    "None. Store at 45-50°F / 7-10°C, 95% RH.",
        "description":  "No disease or defect. Tuber meets commercial grade.",
        "notifiable":   False,
    },
    "Late_Blight": {
        "action":       "REJECT",
        "classifier":   "CNN",
        "severity":     4,
        "category":     "FUNGAL",
        "spread_risk":  "CRITICAL — batch loss within 72h",
        "market_value": "Zero",
        "treatment":    "Destroy. Apply copper fungicide to neighbours. Notify extension.",
        "description":  "Phytophthora infestans. Brown-grey depressed spots extending inward. Rapid spreader.",
        "notifiable":   False,
    },
    "Early_Blight": {
        "action":       "REJECT",
        "classifier":   "CNN",
        "severity":     2,
        "category":     "FUNGAL",
        "spread_risk":  "MEDIUM",
        "market_value": "Animal feed — 20-30%",
        "treatment":    "Chlorothalonil or mancozeb fungicide.",
        "description":  "Alternaria solani. Concentric target-ring lesions.",
        "notifiable":   False,
    },
    "Dry_Rot": {
        "action":       "REJECT",
        "classifier":   "SIFT_SVM",
        "severity":     3,
        "category":     "FUNGAL",
        "spread_risk":  "MEDIUM",
        "market_value": "Zero",
        "treatment":    "Destroy. Disinfect storage. Use certified seed.",
        "description":  "Fusarium spp. Dry, shrunken, concentric wrinkles around entry wound.",
        "notifiable":   False,
    },
    "Gangrene": {
        "action":       "REJECT",
        "classifier":   "SIFT_SVM",
        "severity":     3,
        "category":     "FUNGAL",
        "spread_risk":  "MEDIUM",
        "market_value": "Zero",
        "treatment":    "Remove from storage. Improve ventilation.",
        "description":  "Phoma exigua. Thumb-impression pits. Wrinkles stretch ACROSS (not concentric).",
        "notifiable":   False,
    },
    "Soft_Rot": {
        "action":       "REJECT",
        "classifier":   "CNN",
        "severity":     4,
        "category":     "BACTERIAL",
        "spread_risk":  "CRITICAL — one tuber destroys batch",
        "market_value": "Zero",
        "treatment":    "REMOVE IMMEDIATELY. Bleach storage. Do not store with healthy.",
        "description":  "Pectobacterium/Dickeya. Wet cream decay. Sharp tissue boundary. Foul odour.",
        "notifiable":   False,
    },
    "Wart_Disease": {
        "action":       "REJECT",
        "classifier":   "CNN",
        "severity":     4,
        "category":     "FUNGAL",
        "spread_risk":  "CRITICAL — spores viable 40+ years",
        "market_value": "Zero — LEGAL obligation to report",
        "treatment":    "NOTIFIABLE. Contact APHA immediately. Full quarantine.",
        "description":  "Synchytrium endobioticum. Cauliflower tumours at eyes. NOTIFIABLE DISEASE.",
        "notifiable":   True,
    },
    "Damaged_Potato": {
        "action":       "REJECT",
        "classifier":   "CNN",
        "severity":     3,
        "category":     "DEFECT",
        "spread_risk":  "MEDIUM — entry point for disease",
        "market_value": "Animal feed only — 10-20%",
        "treatment":    "Remove from batch. Check storage conditions.",
        "description":  "Physical damage or bruising. Entry point for bacterial and fungal infection.",
        "notifiable":   False,
    },
    "Defected_Potato": {
        "action":       "REJECT",
        "classifier":   "CNN",
        "severity":     3,
        "category":     "DEFECT",
        "spread_risk":  "MEDIUM",
        "market_value": "Animal feed only — 10-20%",
        "treatment":    "Remove from batch. Do not store with healthy tubers.",
        "description":  "Structural defect or deformity affecting marketability.",
        "notifiable":   False,
    },
    "Diseased_Fungal_Potato": {
        "action":       "REJECT",
        "classifier":   "CNN",
        "severity":     4,
        "category":     "FUNGAL",
        "spread_risk":  "CRITICAL — spreads rapidly in storage",
        "market_value": "Zero",
        "treatment":    "Destroy immediately. Disinfect all contact surfaces.",
        "description":  "Fungal infection detected. Likely Late Blight or Early Blight.",
        "notifiable":   False,
    },
    "Sprouted_Potato": {
        "action":       "SECONDARY",
        "classifier":   "CNN",
        "severity":     1,
        "category":     "DEFECT",
        "spread_risk":  "LOW",
        "market_value": "Processing — 30-50%",
        "treatment":    "Store at lower temperature. Reduce light exposure.",
        "description":  "Sprouting detected. Reduced market value but not diseased.",
        "notifiable":   False,
    },
    "Common_Scab": {
        "action":          "SECONDARY",
        "classifier":      "CNN",
        "severity":        1,
        "scab_threshold":  0.10,
        "category":        "BACTERIAL",
        "spread_risk":     "LOW — soil-borne",
        "market_value":    "Processing if <10% affected",
        "treatment":       "Lower pH <5.2. Consistent irrigation during tuber initiation.",
        "description":     "Streptomyces scabiei. Varying-shaped corky lesions, angular edges.",
        "notifiable":      False,
    },
    "Silver_Scurf": {
        "action":       "KEEP",
        "classifier":   "CNN",
        "severity":     1,
        "category":     "FUNGAL",
        "spread_risk":  "LOW",
        "market_value": "Processing — 50-70%",
        "treatment":    "Thiabendazole post-harvest. Low storage humidity.",
        "description":  "Helminthosporium solani. Silvery sheen patches. Thread structures at 10x.",
        "notifiable":   False,
    },
    "Black_Dot": {
        "action":       "KEEP",
        "classifier":   "CNN",
        "severity":     1,
        "category":     "FUNGAL",
        "spread_risk":  "LOW",
        "market_value": "Processing — 40-60%",
        "treatment":    "No in-season treatment. Improve drainage.",
        "description":  "Colletotrichum coccodes. Tiny jet-black dots. NEVER removed by washing.",
        "notifiable":   False,
    },
    "Mechanical_Damage": {
        "action":       "SECONDARY",
        "classifier":   "SIFT_SVM",
        "severity":     1,
        "category":     "DEFECT",
        "spread_risk":  "LOW — pathogen entry point",
        "market_value": "Processing if minor",
        "treatment":    "Wound healing at 15-18°C for 2 weeks.",
        "description":  "Physical cuts or bruising from harvester. Wound sites visible.",
        "notifiable":   False,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
#  TOMATO QUALITY DATABASE
#  Classes from Roboflow model "tomato-detection-wb9kx" v2:
#    - Good    → Grade A, PASS
#    - Unripe  → Grade B, SECONDARY (hold for ripening)
#    - Bad     → Grade D, REJECT
#
#  References: USDA Tomato Grade Standards (52.1542), FAO Codex Stan 293-2008,
#              UC Davis Postharvest Technology Center (Kader 2002)
# ─────────────────────────────────────────────────────────────────────────────
TOMATO_QUALITY_DB = {

    "Good": {
        "action":       "KEEP",
        "classifier":   "Roboflow_API",
        "severity":     0,
        "category":     "HEALTHY",
        "spread_risk":  "None",
        "market_value": "Full primary market value — Grade A",
        "treatment":    "Store at 55-70°F / 13-21°C. Do not refrigerate below 50°F.",
        "description":  "Ripe, firm tomato meeting USDA Grade A standards. No visible defects or disease.",
        "notifiable":   False,
        # Grade info for UI display
        "grade":        "A",
        "grade_label":  "Grade A — Market Ready",
        "color_hint":   "green",
    },

    "Unripe": {
        "action":       "SECONDARY",
        "classifier":   "Roboflow_API",
        "severity":     1,
        "category":     "QUALITY",
        "spread_risk":  "LOW — ethylene exposure risk to ripe fruit",
        "market_value": "Hold — 40-60% value after ripening (5-7 days)",
        "treatment":    "Separate from ripe fruit. Ripen at 65-70°F with ethylene exposure if needed. "
                        "Do NOT refrigerate — chilling injury below 50°F is irreversible.",
        "description":  "Tomato has not reached commercial ripeness. Green or breaker-stage coloration. "
                        "Meets USDA Mature Green standard but requires additional ripening time.",
        "notifiable":   False,
        "grade":        "B",
        "grade_label":  "Grade B — Hold for Ripening",
        "color_hint":   "yellow",
    },

    "Bad": {
        "action":       "REJECT",
        "classifier":   "Roboflow_API",
        "severity":     3,
        "category":     "DEFECT",
        "spread_risk":  "HIGH — ethylene off-gassing accelerates decay in adjacent fruit",
        "market_value": "Zero — compost or waste stream only",
        "treatment":    "Remove IMMEDIATELY from batch. Ethylene produced by decaying fruit will "
                        "accelerate ripening and spoilage in surrounding tomatoes. Sanitize contact surfaces.",
        "description":  "Tomato shows visible defects: decay, mold, cracking, blossom-end rot, "
                        "catfacing, or mechanical damage exceeding USDA tolerance. Unfit for sale.",
        "notifiable":   False,
        "grade":        "D",
        "grade_label":  "Grade D — Reject",
        "color_hint":   "red",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
#  ROBOFLOW CLASS MAPS
#  Maps raw Roboflow prediction class strings → internal disease/quality keys
# ─────────────────────────────────────────────────────────────────────────────
POTATO_CLASS_MAP = {
    "Potato":                 "Healthy",
    "Bud":                    "Healthy",
    "Damaged potato":         "Damaged_Potato",
    "Defected potato":        "Defected_Potato",
    "Diseased-fungal potato": "Diseased_Fungal_Potato",
    "Diseased/fungal potato": "Diseased_Fungal_Potato",
    "Sprouted potato":        "Sprouted_Potato",
}

TOMATO_CLASS_MAP = {
    "Good":       "Good",
    "good":       "Good",
    "Fresh":      "Good",
    "fresh":      "Good",
    "Ripe":       "Good",
    "ripe":       "Good",
    "Healthy":    "Good",
    "Bad":        "Bad",
    "bad":        "Bad",
    "Defective":  "Bad",
    "defective":  "Bad",
    "Defect":     "Bad",
    "defect":     "Bad",
    "Rotten":     "Bad",
    "rotten":     "Bad",
    "Damaged":    "Bad",
    "damaged":    "Bad",
    "Diseased":   "Bad",
    "diseased":   "Bad",
    "Unripe":     "Unripe",
    "unripe":     "Unripe",
    "Under-ripe": "Unripe",
    "UnderRipe":  "Unripe",
}

# ─────────────────────────────────────────────────────────────────────────────
#  UNIFIED LOOKUP
#  Use get_db(crop) to get the right database for the active crop.
# ─────────────────────────────────────────────────────────────────────────────
CROP_DB = {
    "potato": POTATO_DISEASE_DB,
    "tomato": TOMATO_QUALITY_DB,
}

CROP_CLASS_MAP = {
    "potato": POTATO_CLASS_MAP,
    "tomato": TOMATO_CLASS_MAP,
}

# Legacy alias — keeps backward compatibility with existing code that imports DISEASE_DB
DISEASE_DB = POTATO_DISEASE_DB

# Also expose the old ROBOFLOW_CLASS_MAP alias for backward compatibility
ROBOFLOW_CLASS_MAP = POTATO_CLASS_MAP


def get_db(crop: str = "potato") -> dict:
    """Return the disease/quality database for the given crop."""
    return CROP_DB.get(crop, POTATO_DISEASE_DB)


def get_class_map(crop: str = "potato") -> dict:
    """Return the Roboflow class name → internal key map for the given crop."""
    return CROP_CLASS_MAP.get(crop, POTATO_CLASS_MAP)


def get_action(disease: str, crop: str = "potato") -> str:
    return get_db(crop).get(disease, {}).get("action", "REVIEW")


def is_notifiable(disease: str, crop: str = "potato") -> bool:
    return get_db(crop).get(disease, {}).get("notifiable", False)


ALL_POTATO_CLASSES = list(POTATO_DISEASE_DB.keys())
ALL_TOMATO_CLASSES = list(TOMATO_QUALITY_DB.keys())

# Legacy aliases
ALL_CLASSES  = ALL_POTATO_CLASSES
CNN_CLASSES  = [k for k, v in POTATO_DISEASE_DB.items() if v["classifier"] == "CNN"]
SIFT_CLASSES = [k for k, v in POTATO_DISEASE_DB.items() if v["classifier"] == "SIFT_SVM"]
