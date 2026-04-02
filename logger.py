"""
Terroir AI — Stateless CSV Audit Logger
Spec: log every classification locally (Timestamp, Tuber ID, Disease, Confidence, Action)
This is the agribusiness audit trail for the $299/mo SaaS model.
"""

import csv, os, time, threading
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

FIELDS = ["timestamp", "tuber_id", "session_id",
          "disease",   "classifier", "confidence",
          "action",    "surface_pct", "inference_ms", "notifiable"]


class Logger:
    def __init__(self):
        self.session_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path    = os.path.join(LOG_DIR, f"session_{self.session_id}.csv")
        self._lock       = threading.Lock()
        self._n          = 0
        self.start_time  = time.time()

        # In-memory stats
        self.total = 0; self.kept = 0; self.rejected = 0; self.secondary = 0
        self.value_saved = 0.0
        self.disease_counts = {}

        with open(self.log_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    # ── Public API ────────────────────────────────────────────────────────────

    def log(self, r: dict) -> int:
        """Log one result dict. Returns tuber_id."""
        with self._lock:
            self._n += 1
            tid = self._n

            row = {
                "timestamp":    datetime.now().isoformat(),
                "tuber_id":     tid,
                "session_id":   self.session_id,
                "disease":      r.get("disease", "Unknown"),
                "classifier":   r.get("classifier", "—"),
                "confidence":   round(r.get("confidence", 0), 4),
                "action":       r.get("action", "KEEP"),
                "surface_pct":  round(r.get("surface_pct", 0), 4),
                "inference_ms": round(r.get("inference_ms", 0), 2),
                "notifiable":   r.get("notifiable", False),
            }
            with open(self.log_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=FIELDS).writerow(row)

            # Stats
            self.total += 1
            a = row["action"]
            if a == "KEEP":
                self.kept     += 1
            elif a == "REJECT":
                self.rejected += 1;  self.value_saved += 0.32
            else:
                self.secondary += 1; self.value_saved += 0.08
            d = row["disease"]
            self.disease_counts[d] = self.disease_counts.get(d, 0) + 1

        return tid

    def stats(self) -> dict:
        with self._lock:
            elapsed = max(1, time.time() - self.start_time)
            rr = (self.rejected / max(1, self.total)) * 100
            return dict(
                total        = self.total,
                kept         = self.kept,
                rejected     = self.rejected,
                secondary    = self.secondary,
                reject_rate  = round(rr, 1),
                value_saved  = round(self.value_saved, 2),
                rate_per_min = round(self.total / elapsed * 60, 1),
                elapsed_s    = int(elapsed),
                disease_counts = dict(self.disease_counts),
                log_path     = self.log_path,
                session_id   = self.session_id,
            )
