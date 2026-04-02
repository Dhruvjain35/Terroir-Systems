"""
CustomTkinter dashboard with three modes:
1) Image Upload & Analytics
2) Live Simulation
3) Offline Demo
"""

import os
import time
import queue
import threading

import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageDraw
import cv2

from config import (
    APP_TITLE, APP_GEOMETRY, APP_MIN_SIZE,
    APPEARANCE_MODE, COLOR_THEME, COLORS,
    UI_FPS, INFERENCE_FPS, DEMO_ASSETS_DIR,
    FRAME_QUEUE_MAX, RESULT_QUEUE_MAX,
)
from vision_pipeline import VisionPipeline


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
        self.image_result_queue = queue.Queue(maxsize=4)

        self.pipeline = VisionPipeline(self.frame_queue, self.result_queue)

        self._active_mode = "live"
        self._stream_mode = None
        self._last_result = None
        self._last_infer_ts = 0.0
        self._fps_count = 0
        self._fps_last_ts = time.time()
        self._fps_value = 0.0

        self._build_ui()
        self._switch_mode("live")
        self.after(int(1000 / max(1, UI_FPS)), self._poll_queues)

    # ── UI Layout ──────────────────────────────────────────────────────

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar
        sidebar = ctk.CTkFrame(self, width=220, fg_color=COLORS["panel"], corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsw")
        sidebar.grid_rowconfigure(6, weight=1)

        logo = ctk.CTkLabel(
            sidebar,
            text="TERROIR AI",
            font=("Helvetica", 18, "bold"),
            text_color=COLORS["accent"],
        )
        logo.grid(row=0, column=0, padx=16, pady=(16, 6), sticky="w")

        subtitle = ctk.CTkLabel(
            sidebar,
            text="Dashboard",
            font=("Helvetica", 12),
            text_color=COLORS["muted"],
        )
        subtitle.grid(row=1, column=0, padx=16, pady=(0, 16), sticky="w")

        self.btn_image = ctk.CTkButton(
            sidebar,
            text="Image Analytics",
            height=40,
            command=lambda: self._switch_mode("image"),
        )
        self.btn_image.grid(row=2, column=0, padx=16, pady=(4, 8), sticky="ew")

        self.btn_live = ctk.CTkButton(
            sidebar,
            text="Live Simulation",
            height=40,
            command=lambda: self._switch_mode("live"),
        )
        self.btn_live.grid(row=3, column=0, padx=16, pady=8, sticky="ew")

        self.btn_demo = ctk.CTkButton(
            sidebar,
            text="Offline Demo",
            height=40,
            command=lambda: self._switch_mode("demo"),
        )
        self.btn_demo.grid(row=4, column=0, padx=16, pady=8, sticky="ew")

        self.sidebar_status = ctk.CTkLabel(
            sidebar,
            text="Status: IDLE",
            font=("Helvetica", 11),
            text_color=COLORS["muted"],
        )
        self.sidebar_status.grid(row=7, column=0, padx=16, pady=16, sticky="w")

        # Main content container
        self.container = ctk.CTkFrame(self, fg_color=COLORS["bg"], corner_radius=0)
        self.container.grid(row=0, column=1, sticky="nsew")
        self.container.grid_columnconfigure(0, weight=1)
        self.container.grid_rowconfigure(0, weight=1)

        # Pages
        self.page_image = ctk.CTkFrame(self.container, fg_color=COLORS["bg"], corner_radius=0)
        self.page_live = ctk.CTkFrame(self.container, fg_color=COLORS["bg"], corner_radius=0)
        self.page_demo = ctk.CTkFrame(self.container, fg_color=COLORS["bg"], corner_radius=0)

        for p in (self.page_image, self.page_live, self.page_demo):
            p.grid(row=0, column=0, sticky="nsew")

        self._build_image_page()
        self._build_stream_page(self.page_live, mode="live")
        self._build_stream_page(self.page_demo, mode="demo")

    def _build_image_page(self):
        self.page_image.grid_columnconfigure(0, weight=1)
        self.page_image.grid_columnconfigure(1, weight=1)
        self.page_image.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self.page_image, fg_color=COLORS["bg"], corner_radius=0)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=16, pady=(16, 8))
        header.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            header,
            text="Image Upload & Analytics",
            font=("Helvetica", 18, "bold"),
            text_color=COLORS["text"],
        )
        title.grid(row=0, column=0, sticky="w")

        upload_btn = ctk.CTkButton(
            header,
            text="Upload Image",
            height=36,
            command=self._select_image,
        )
        upload_btn.grid(row=0, column=1, sticky="e")

        self.image_path_label = ctk.CTkLabel(
            self.page_image,
            text="No image selected",
            font=("Helvetica", 11),
            text_color=COLORS["muted"],
        )
        self.image_path_label.grid(row=2, column=0, columnspan=2, sticky="w", padx=16, pady=(4, 12))

        # Image panels
        self.img_panel_left = ctk.CTkFrame(self.page_image, fg_color=COLORS["panel"], corner_radius=12)
        self.img_panel_left.grid(row=1, column=0, sticky="nsew", padx=(16, 8), pady=8)
        self.img_panel_left.grid_rowconfigure(1, weight=1)
        self.img_panel_left.grid_columnconfigure(0, weight=1)

        self.img_panel_right = ctk.CTkFrame(self.page_image, fg_color=COLORS["panel"], corner_radius=12)
        self.img_panel_right.grid(row=1, column=1, sticky="nsew", padx=(8, 16), pady=8)
        self.img_panel_right.grid_rowconfigure(1, weight=1)
        self.img_panel_right.grid_columnconfigure(0, weight=1)

        left_title = ctk.CTkLabel(self.img_panel_left, text="Original / ROI", font=("Helvetica", 12, "bold"), text_color=COLORS["muted"])
        left_title.grid(row=0, column=0, sticky="w", padx=12, pady=(12, 6))
        self.img_left_label = ctk.CTkLabel(self.img_panel_left, text="", fg_color=COLORS["card"], corner_radius=10)
        self.img_left_label.grid(row=1, column=0, sticky="nsew", padx=12, pady=12)

        right_title = ctk.CTkLabel(self.img_panel_right, text="Heatmap / Overlay", font=("Helvetica", 12, "bold"), text_color=COLORS["muted"])
        right_title.grid(row=0, column=0, sticky="w", padx=12, pady=(12, 6))
        self.img_right_label = ctk.CTkLabel(self.img_panel_right, text="", fg_color=COLORS["card"], corner_radius=10)
        self.img_right_label.grid(row=1, column=0, sticky="nsew", padx=12, pady=12)

        # Result card
        self.result_card = ctk.CTkFrame(self.page_image, fg_color=COLORS["panel"], corner_radius=12)
        self.result_card.grid(row=3, column=0, columnspan=2, sticky="ew", padx=16, pady=(0, 16))
        self.result_card.grid_columnconfigure(0, weight=1)

        self.result_label = ctk.CTkLabel(
            self.result_card,
            text="Prediction: —",
            font=("Helvetica", 14, "bold"),
            text_color=COLORS["accent"],
        )
        self.result_label.grid(row=0, column=0, sticky="w", padx=12, pady=(12, 2))

        self.result_meta = ctk.CTkLabel(
            self.result_card,
            text="Confidence: —",
            font=("Helvetica", 12),
            text_color=COLORS["muted"],
        )
        self.result_meta.grid(row=1, column=0, sticky="w", padx=12, pady=(0, 12))

    def _build_stream_page(self, page, mode: str):
        page.grid_columnconfigure(0, weight=3)
        page.grid_columnconfigure(1, weight=2)
        page.grid_rowconfigure(1, weight=1)

        title = "Live Simulation" if mode == "live" else "Offline Demo Mode"
        header = ctk.CTkFrame(page, fg_color=COLORS["bg"], corner_radius=0)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=16, pady=(16, 8))
        header.grid_columnconfigure(0, weight=1)

        label = ctk.CTkLabel(header, text=title, font=("Helvetica", 18, "bold"), text_color=COLORS["text"])
        label.grid(row=0, column=0, sticky="w")

        if mode == "demo":
            badge = ctk.CTkLabel(
                header,
                text="DEMO MODE",
                font=("Helvetica", 12, "bold"),
                text_color=COLORS["bg"],
                fg_color=COLORS["warn"],
                corner_radius=8,
                padx=10,
                pady=4,
            )
            badge.grid(row=0, column=1, sticky="e")

        # Camera panel
        cam_panel = ctk.CTkFrame(page, fg_color=COLORS["panel"], corner_radius=12)
        cam_panel.grid(row=1, column=0, sticky="nsew", padx=(16, 8), pady=8)
        cam_panel.grid_rowconfigure(1, weight=1)
        cam_panel.grid_columnconfigure(0, weight=1)

        cam_title = ctk.CTkLabel(cam_panel, text="Camera Feed", font=("Helvetica", 12, "bold"), text_color=COLORS["muted"])
        cam_title.grid(row=0, column=0, sticky="w", padx=12, pady=(12, 6))

        cam_label = ctk.CTkLabel(cam_panel, text="Waiting for stream...", fg_color=COLORS["card"], corner_radius=10)
        cam_label.grid(row=1, column=0, sticky="nsew", padx=12, pady=12)

        # Right panel: controls + stats + log
        right = ctk.CTkFrame(page, fg_color=COLORS["panel"], corner_radius=12)
        right.grid(row=1, column=1, sticky="nsew", padx=(8, 16), pady=8)
        right.grid_rowconfigure(3, weight=1)
        right.grid_columnconfigure(0, weight=1)

        btn_row = ctk.CTkFrame(right, fg_color=COLORS["panel"], corner_radius=0)
        btn_row.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 8))
        btn_row.grid_columnconfigure((0, 1), weight=1)

        start_btn = ctk.CTkButton(
            btn_row,
            text="START",
            height=44,
            font=("Helvetica", 14, "bold"),
            command=self._start_live if mode == "live" else self._start_demo,
        )
        start_btn.grid(row=0, column=0, padx=(0, 8), sticky="ew")

        stop_btn = ctk.CTkButton(
            btn_row,
            text="STOP",
            height=44,
            font=("Helvetica", 14, "bold"),
            fg_color=COLORS["bad"],
            hover_color="#D31539",
            command=self._stop_stream,
        )
        stop_btn.grid(row=0, column=1, padx=(8, 0), sticky="ew")
        stop_btn.configure(state="disabled")

        status = ctk.CTkLabel(right, text="Status: IDLE", font=("Helvetica", 11), text_color=COLORS["muted"])
        status.grid(row=1, column=0, sticky="w", padx=12, pady=(0, 6))

        fps_label = ctk.CTkLabel(right, text="FPS: 0", font=("Helvetica", 12, "bold"), text_color=COLORS["blue"])
        fps_label.grid(row=2, column=0, sticky="w", padx=12, pady=(0, 8))

        log_box = ctk.CTkTextbox(right, height=220, corner_radius=8)
        log_box.grid(row=3, column=0, sticky="nsew", padx=12, pady=(0, 12))
        log_box.insert("end", "Event Log:\n")
        log_box.configure(state="disabled")

        # Stats labels
        stats = ctk.CTkLabel(right, text="Tubers/sec: 0.00 | Reject: 0.0%", font=("Helvetica", 11), text_color=COLORS["muted"])
        stats.grid(row=4, column=0, sticky="w", padx=12, pady=(0, 12))

        # Store references
        if mode == "live":
            self.live_cam_label = cam_label
            self.live_start_btn = start_btn
            self.live_stop_btn = stop_btn
            self.live_status = status
            self.live_fps = fps_label
            self.live_log = log_box
            self.live_stats = stats
        else:
            self.demo_cam_label = cam_label
            self.demo_start_btn = start_btn
            self.demo_stop_btn = stop_btn
            self.demo_status = status
            self.demo_fps = fps_label
            self.demo_log = log_box
            self.demo_stats = stats

    # ── Mode switching ──────────────────────────────────────────────────

    def _switch_mode(self, mode: str):
        if mode not in ("image", "live", "demo"):
            return

        self.pipeline.stop()
        self._stream_mode = None
        self.sidebar_status.configure(text="Status: IDLE", text_color=COLORS["muted"])
        self._reset_stream_controls()

        for p in (self.page_image, self.page_live, self.page_demo):
            p.grid_remove()

        if mode == "image":
            self.page_image.grid()
        elif mode == "live":
            self.page_live.grid()
        else:
            self.page_demo.grid()

        self._active_mode = mode
        self._highlight_nav(mode)

    def _reset_stream_controls(self):
        if hasattr(self, "live_start_btn") and hasattr(self, "live_stop_btn"):
            self.live_start_btn.configure(state="normal")
            self.live_stop_btn.configure(state="disabled")
            self.live_status.configure(text="Status: IDLE", text_color=COLORS["muted"])
        if hasattr(self, "demo_start_btn") and hasattr(self, "demo_stop_btn"):
            self.demo_start_btn.configure(state="normal")
            self.demo_stop_btn.configure(state="disabled")
            self.demo_status.configure(text="Status: IDLE", text_color=COLORS["muted"])

    def _highlight_nav(self, mode: str):
        for btn in (self.btn_image, self.btn_live, self.btn_demo):
            btn.configure(fg_color=COLORS["card"], text_color=COLORS["text"])
        if mode == "image":
            self.btn_image.configure(fg_color=COLORS["accent"], text_color=COLORS["bg"])
        elif mode == "live":
            self.btn_live.configure(fg_color=COLORS["accent"], text_color=COLORS["bg"])
        else:
            self.btn_demo.configure(fg_color=COLORS["accent"], text_color=COLORS["bg"])

    # ── Image analytics ────────────────────────────────────────────────

    def _select_image(self):
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All", "*.*")],
        )
        if not path:
            return
        self.image_path_label.configure(text=os.path.basename(path))

        # Display original first (non-blocking)
        bgr = cv2.imread(path)
        if bgr is not None:
            self._render_static_image(self.img_left_label, bgr)

        def _worker():
            res = self.pipeline.analyze_image(path)
            self.image_result_queue.put(res)

        threading.Thread(target=_worker, daemon=True).start()

    # ── Streaming controls ─────────────────────────────────────────────

    def _start_live(self):
        self.pipeline.start_live_camera()
        self._stream_mode = "live"
        self.live_start_btn.configure(state="disabled")
        self.live_stop_btn.configure(state="normal")
        self.live_status.configure(text="Status: RUNNING", text_color=COLORS["good"])
        self.sidebar_status.configure(text="Status: LIVE", text_color=COLORS["good"])

    def _start_demo(self):
        self.pipeline.start_demo(DEMO_ASSETS_DIR)
        self._stream_mode = "demo"
        self.demo_start_btn.configure(state="disabled")
        self.demo_stop_btn.configure(state="normal")
        self.demo_status.configure(text="Status: RUNNING", text_color=COLORS["good"])
        self.sidebar_status.configure(text="Status: DEMO", text_color=COLORS["warn"])

    def _stop_stream(self):
        self.pipeline.stop()
        self._stream_mode = None
        if hasattr(self, "live_start_btn"):
            self.live_start_btn.configure(state="normal")
            self.live_stop_btn.configure(state="disabled")
            self.live_status.configure(text="Status: STOPPED", text_color=COLORS["muted"])
        if hasattr(self, "demo_start_btn"):
            self.demo_start_btn.configure(state="normal")
            self.demo_stop_btn.configure(state="disabled")
            self.demo_status.configure(text="Status: STOPPED", text_color=COLORS["muted"])
        self.sidebar_status.configure(text="Status: IDLE", text_color=COLORS["muted"])

    def _on_close(self):
        self.pipeline.stop()
        self.destroy()

    # ── Polling ────────────────────────────────────────────────────────

    def _poll_queues(self):
        self._drain_frames()
        self._drain_results()
        self._drain_image_results()
        self.after(int(1000 / max(1, UI_FPS)), self._poll_queues)

    def _drain_frames(self):
        frame = None
        while not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get_nowait()
            except queue.Empty:
                break

        if frame is not None:
            self._fps_count += 1
            now = time.time()
            if now - self._fps_last_ts >= 1.0:
                self._fps_value = self._fps_count / (now - self._fps_last_ts)
                self._fps_count = 0
                self._fps_last_ts = now

            if self._stream_mode == "live":
                self._render_stream_frame(self.live_cam_label, frame)
                self.live_fps.configure(text=f"FPS: {self._fps_value:.0f}")
            elif self._stream_mode == "demo":
                self._render_stream_frame(self.demo_cam_label, frame)
                self.demo_fps.configure(text=f"FPS: {self._fps_value:.0f}")

            # Enqueue for inference AFTER display (sim-to-real fix)
            min_interval = 1.0 / max(1, INFERENCE_FPS)
            if self._stream_mode and (now - self._last_infer_ts) >= min_interval:
                self._last_infer_ts = now
                self.pipeline.enqueue_for_inference(frame)

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
            if self._stream_mode == "live" and hasattr(self, "live_status"):
                self.live_status.configure(text=f"Status: {msg}", text_color=COLORS["bad"])
            if self._stream_mode == "demo" and hasattr(self, "demo_status"):
                self.demo_status.configure(text=f"Status: {msg}", text_color=COLORS["bad"])
            self.sidebar_status.configure(text=f"Status: {msg}", text_color=COLORS["bad"])
            return

        if packet.get("type") == "result":
            result = packet.get("result", {})
            stats = packet.get("stats", {})
            tuber_id = packet.get("tuber_id", 0)
            self._last_result = result
            self._append_log(result, tuber_id)
            self._update_stats(stats)

    def _drain_image_results(self):
        res = None
        while not self.image_result_queue.empty():
            try:
                res = self.image_result_queue.get_nowait()
            except queue.Empty:
                break

        if not res:
            return

        if res.get("error"):
            self.result_label.configure(text=f"Prediction: {res['error']}")
            self.result_meta.configure(text="Confidence: —")
            return

        annotated = res.get("annotated_bgr")
        heatmap = res.get("heatmap_bgr")
        result = res.get("result", {})

        if annotated is not None:
            self._render_static_image(self.img_left_label, annotated)
        if heatmap is not None:
            self._render_static_image(self.img_right_label, heatmap)

        disease = result.get("disease", "—").replace("_", " ")
        conf = result.get("confidence", 0.0) * 100
        self.result_label.configure(text=f"Prediction: {disease}")
        self.result_meta.configure(text=f"Confidence: {conf:.1f}%")

    # ── Rendering helpers ──────────────────────────────────────────────

    def _render_stream_frame(self, label, frame_bgr):
        self._render_image(label, frame_bgr, draw_border=True)

    def _render_static_image(self, label, frame_bgr):
        self._render_image(label, frame_bgr, draw_border=False)

    def _render_image(self, label, frame_bgr, draw_border: bool):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        w = max(1, label.winfo_width())
        h = max(1, label.winfo_height())
        img = img.resize((w, h), Image.LANCZOS)

        if draw_border and self._last_result:
            action = self._last_result.get("action", "KEEP")
            col = COLORS["good"] if action == "KEEP" else COLORS["bad"] if action == "REJECT" else COLORS["warn"]
            draw = ImageDraw.Draw(img)
            for t in range(4):
                draw.rectangle([t, t, w - 1 - t, h - 1 - t], outline=col)

        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(w, h))
        label.configure(image=ctk_img, text="")
        label.image = ctk_img

    # ── Logs & stats ───────────────────────────────────────────────────

    def _append_log(self, result: dict, tuber_id: int):
        line = f"[{tuber_id:04d}] {result.get('action','KEEP'):<8} {result.get('disease','?').replace('_',' ')} {result.get('confidence',0)*100:4.0f}%\n"
        if self._stream_mode == "live" and hasattr(self, "live_log"):
            self._append_text(self.live_log, line)
        if self._stream_mode == "demo" and hasattr(self, "demo_log"):
            self._append_text(self.demo_log, line)

    def _append_text(self, textbox, text: str):
        textbox.configure(state="normal")
        textbox.insert("end", text)
        textbox.see("end")
        textbox.configure(state="disabled")

    def _update_stats(self, stats: dict):
        elapsed = max(1, stats.get("elapsed_s", 1))
        total = stats.get("total", 0)
        tps = total / elapsed
        reject = stats.get("reject_rate", 0.0)
        label_text = f"Tubers/sec: {tps:.2f} | Reject: {reject:.1f}%"

        if self._stream_mode == "live" and hasattr(self, "live_stats"):
            self.live_stats.configure(text=label_text)
        if self._stream_mode == "demo" and hasattr(self, "demo_stats"):
            self.demo_stats.configure(text=label_text)
