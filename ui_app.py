"""
CustomTkinter UI for Terroir AI.
"""

import queue
import time

import customtkinter as ctk
from PIL import Image, ImageDraw
import cv2

from config import (
    APP_TITLE, APP_GEOMETRY, APP_MIN_SIZE,
    APPEARANCE_MODE, COLOR_THEME, COLORS,
    UI_FPS, INFERENCE_FPS,
    FRAME_QUEUE_MAX, RESULT_QUEUE_MAX,
)
from vision_engine import VisionEngine


class TerroirApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode(APPEARANCE_MODE)
        ctk.set_default_color_theme(COLOR_THEME)

        self.title(APP_TITLE)
        self.geometry(APP_GEOMETRY)
        self.minsize(*APP_MIN_SIZE)
        self.configure(fg_color=COLORS["bg"])

        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Queues for thread-safe communication
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_MAX)
        self.result_queue = queue.Queue(maxsize=RESULT_QUEUE_MAX)

        self.engine = VisionEngine(self.frame_queue, self.result_queue)
        self._running = False
        self._last_result = None
        self._last_infer_ts = 0.0

        self._build_ui()
        self._build_onboarding()
        self.after(int(1000 / max(1, UI_FPS)), self._poll_queues)

    # ── UI Layout ──────────────────────────────────────────────────────

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self, fg_color=COLORS["bg"], corner_radius=0)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=16, pady=(12, 8))
        header.grid_columnconfigure(0, weight=1)
        header.grid_columnconfigure(1, weight=0)

        title = ctk.CTkLabel(
            header,
            text="TERROIR AI",
            font=("Helvetica", 22, "bold"),
            text_color=COLORS["accent"],
        )
        title.grid(row=0, column=0, sticky="w")

        subtitle = ctk.CTkLabel(
            header,
            text="Intelligent Crop Sorting System",
            font=("Helvetica", 12),
            text_color=COLORS["muted"],
        )
        subtitle.grid(row=1, column=0, sticky="w", pady=(0, 4))

        self.demo_badge = ctk.CTkLabel(
            header,
            text="DEMO MODE",
            font=("Helvetica", 12, "bold"),
            text_color=COLORS["bg"],
            fg_color=COLORS["warn"],
            corner_radius=8,
            padx=10,
            pady=4,
        )
        self.demo_badge.grid(row=0, column=1, rowspan=2, sticky="e")

        # Camera panel
        self.camera_frame = ctk.CTkFrame(self, fg_color=COLORS["panel"], corner_radius=12)
        self.camera_frame.grid(row=1, column=0, sticky="nsew", padx=(16, 8), pady=12)
        self.camera_frame.grid_rowconfigure(1, weight=1)
        self.camera_frame.grid_columnconfigure(0, weight=1)

        cam_header = ctk.CTkLabel(
            self.camera_frame,
            text="CAMERA FEED",
            font=("Helvetica", 12, "bold"),
            text_color=COLORS["muted"],
        )
        cam_header.grid(row=0, column=0, sticky="w", padx=12, pady=(12, 6))

        self.camera_label = ctk.CTkLabel(
            self.camera_frame,
            text="Waiting for camera...",
            text_color=COLORS["muted"],
            fg_color=COLORS["card"],
            corner_radius=10,
        )
        self.camera_label.grid(row=1, column=0, sticky="nsew", padx=12, pady=12)

        # Right column
        right = ctk.CTkFrame(self, fg_color=COLORS["bg"], corner_radius=0)
        right.grid(row=1, column=1, sticky="nsew", padx=(8, 16), pady=12)
        right.grid_rowconfigure(2, weight=1)
        right.grid_columnconfigure(0, weight=1)

        # Controls
        controls = ctk.CTkFrame(right, fg_color=COLORS["panel"], corner_radius=12)
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        controls.grid_columnconfigure((0, 1), weight=1)

        self.start_btn = ctk.CTkButton(
            controls,
            text="START",
            height=48,
            font=("Helvetica", 16, "bold"),
            command=self._start,
        )
        self.start_btn.grid(row=0, column=0, padx=12, pady=12, sticky="ew")

        self.stop_btn = ctk.CTkButton(
            controls,
            text="STOP",
            height=48,
            font=("Helvetica", 16, "bold"),
            fg_color=COLORS["bad"],
            hover_color="#D31539",
            command=self._stop,
        )
        self.stop_btn.grid(row=0, column=1, padx=12, pady=12, sticky="ew")
        self.stop_btn.configure(state="disabled")

        self.status_label = ctk.CTkLabel(
            controls,
            text="Status: IDLE",
            font=("Helvetica", 12),
            text_color=COLORS["muted"],
        )
        self.status_label.grid(row=1, column=0, columnspan=2, sticky="w", padx=12, pady=(0, 12))

        # Stats panel
        stats = ctk.CTkFrame(right, fg_color=COLORS["panel"], corner_radius=12)
        stats.grid(row=1, column=0, sticky="ew")
        stats.grid_columnconfigure(0, weight=1)

        stats_title = ctk.CTkLabel(
            stats,
            text="LIVE STATISTICS",
            font=("Helvetica", 12, "bold"),
            text_color=COLORS["muted"],
        )
        stats_title.grid(row=0, column=0, sticky="w", padx=12, pady=(12, 6))

        self.tps_label = ctk.CTkLabel(stats, text="0.00 tubers/sec", font=("Helvetica", 18, "bold"))
        self.tps_label.grid(row=1, column=0, sticky="w", padx=12, pady=(0, 2))

        self.total_label = ctk.CTkLabel(stats, text="Total: 0", font=("Helvetica", 12), text_color=COLORS["text"])
        self.total_label.grid(row=2, column=0, sticky="w", padx=12)

        self.reject_label = ctk.CTkLabel(stats, text="Reject Rate: 0.0%", font=("Helvetica", 12), text_color=COLORS["warn"])
        self.reject_label.grid(row=3, column=0, sticky="w", padx=12, pady=(0, 6))

        self.last_pred_label = ctk.CTkLabel(
            stats,
            text="Last: —",
            font=("Helvetica", 12, "bold"),
            text_color=COLORS["accent"],
        )
        self.last_pred_label.grid(row=4, column=0, sticky="w", padx=12, pady=(0, 6))

        self.disease_counts_label = ctk.CTkLabel(
            stats,
            text="Disease Counts:\n—",
            justify="left",
            font=("Helvetica", 11),
            text_color=COLORS["muted"],
        )
        self.disease_counts_label.grid(row=5, column=0, sticky="w", padx=12, pady=(0, 12))

    def _build_onboarding(self):
        self._onboarding = ctk.CTkFrame(
            self,
            fg_color=COLORS["bg"],
            corner_radius=0,
        )
        self._onboarding.place(relx=0, rely=0, relwidth=1, relheight=1)

        card = ctk.CTkFrame(self._onboarding, fg_color=COLORS["panel"], corner_radius=16)
        card.place(relx=0.5, rely=0.5, anchor="center")

        title = ctk.CTkLabel(
            card,
            text="DEMO MODE",
            font=("Helvetica", 28, "bold"),
            text_color=COLORS["accent"],
        )
        title.pack(padx=24, pady=(24, 8))

        subtitle = ctk.CTkLabel(
            card,
            text="Investor Pitch Walkthrough",
            font=("Helvetica", 14),
            text_color=COLORS["muted"],
        )
        subtitle.pack(padx=24, pady=(0, 16))

        steps = (
            "1. Press START to activate the camera and inference.\n"
            "2. Place a tuber on the conveyor belt under the camera.\n"
            "3. Watch the live verdict and disease counts update.\n"
            "4. ROI crops are saved for debugging if enabled."
        )
        body = ctk.CTkLabel(
            card,
            text=steps,
            justify="left",
            font=("Helvetica", 12),
            text_color=COLORS["text"],
        )
        body.pack(padx=24, pady=(0, 20))

        btn_row = ctk.CTkFrame(card, fg_color=COLORS["panel"], corner_radius=0)
        btn_row.pack(fill="x", padx=24, pady=(0, 24))
        btn_row.grid_columnconfigure((0, 1), weight=1)

        enter_btn = ctk.CTkButton(
            btn_row,
            text="ENTER DEMO",
            height=40,
            font=("Helvetica", 14, "bold"),
            command=self._dismiss_onboarding,
        )
        enter_btn.grid(row=0, column=0, padx=(0, 8), sticky="ew")

        quit_btn = ctk.CTkButton(
            btn_row,
            text="QUIT",
            height=40,
            font=("Helvetica", 14, "bold"),
            fg_color=COLORS["bad"],
            hover_color="#D31539",
            command=self._on_close,
        )
        quit_btn.grid(row=0, column=1, padx=(8, 0), sticky="ew")

        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="disabled")

    def _dismiss_onboarding(self):
        if self._onboarding:
            self._onboarding.destroy()
            self._onboarding = None
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")

    # ── UI Actions ─────────────────────────────────────────────────────

    def _start(self):
        if self._running:
            return
        self._running = True
        self.engine.start()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_label.configure(text="Status: RUNNING", text_color=COLORS["good"])

    def _stop(self):
        if not self._running:
            return
        self._running = False
        self.engine.stop()
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_label.configure(text="Status: STOPPED", text_color=COLORS["muted"])

    def _on_close(self):
        self.engine.stop()
        self.destroy()

    # ── Poll Queues ────────────────────────────────────────────────────

    def _poll_queues(self):
        self._drain_frames()
        self._drain_results()
        self.after(int(1000 / max(1, UI_FPS)), self._poll_queues)

    def _drain_frames(self):
        frame = None
        while not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get_nowait()
            except queue.Empty:
                break

        if frame is not None:
            self._show_frame(frame)
            # Enqueue for inference AFTER display (sim-to-real fix)
            now = time.time()
            min_interval = 1.0 / max(1, INFERENCE_FPS)
            if self._running and (now - self._last_infer_ts) >= min_interval:
                self._last_infer_ts = now
                self.engine.enqueue_for_inference(frame)

    def _drain_results(self):
        packet = None
        while not self.result_queue.empty():
            try:
                packet = self.result_queue.get_nowait()
            except queue.Empty:
                break

        if not packet:
            return

        if packet.get("type") == "status":
            msg = packet.get("message", "")
            self.status_label.configure(text=f"Status: {msg}", text_color=COLORS["bad"])
            return

        if packet.get("type") == "result":
            result = packet.get("result", {})
            stats = packet.get("stats", {})
            self._last_result = result
            self._update_stats(result, stats)

    # ── Rendering ──────────────────────────────────────────────────────

    def _show_frame(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        w = max(1, self.camera_label.winfo_width())
        h = max(1, self.camera_label.winfo_height())
        img = img.resize((w, h), Image.LANCZOS)

        if self._last_result:
            action = self._last_result.get("action", "KEEP")
            col = COLORS["good"] if action == "KEEP" else COLORS["bad"] if action == "REJECT" else COLORS["warn"]
            draw = ImageDraw.Draw(img)
            for t in range(4):
                draw.rectangle([t, t, w - 1 - t, h - 1 - t], outline=col)

        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(w, h))
        self.camera_label.configure(image=ctk_img, text="")
        self.camera_label.image = ctk_img

    def _update_stats(self, result: dict, stats: dict):
        elapsed = max(1, stats.get("elapsed_s", 1))
        total = stats.get("total", 0)
        tps = total / elapsed

        self.tps_label.configure(text=f"{tps:.2f} tubers/sec")
        self.total_label.configure(text=f"Total: {total}")
        self.reject_label.configure(text=f"Reject Rate: {stats.get('reject_rate', 0.0):.1f}%")

        disease = result.get("disease", "—").replace("_", " ")
        conf = result.get("confidence", 0.0) * 100
        action = result.get("action", "KEEP")
        self.last_pred_label.configure(text=f"Last: {disease}  ({action}, {conf:.0f}%)")

        counts = stats.get("disease_counts", {})
        if counts:
            top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:6]
            lines = [f"{k.replace('_',' ')}: {v}" for k, v in top]
            text = "Disease Counts:\n" + "\n".join(lines)
        else:
            text = "Disease Counts:\n—"
        self.disease_counts_label.configure(text=text)
