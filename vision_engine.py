"""
Vision engine: Phase 1 detection + Phase 2 classification on background threads.
"""

import os
import time
import threading
import queue

import cv2
import numpy as np

from classifier import Classifier
from logger import Logger
from config import (
    CAMERA_ID, CAMERA_FPS, INFERENCE_FPS,
    FRAME_QUEUE_MAX, INFER_QUEUE_MAX,
    SCAN_X_PCT, BGS_HISTORY, BGS_VAR_THRESHOLD,
    CASCADE_PATH, DEBUG_MODE, DEBUG_DIR,
)


def _safe_sleep(start_ts: float, interval_s: float) -> None:
    """Sleep for the remaining interval, never negative."""
    remaining = interval_s - (time.time() - start_ts)
    if remaining > 0:
        time.sleep(remaining)


class VisionEngine:
    """
    Runs camera I/O and inference on separate daemon threads.

    Queues:
      - frame_queue: camera frames -> GUI
      - infer_queue: GUI frames -> inference
      - result_queue: inference results -> GUI
    """

    def __init__(self, frame_queue: queue.Queue, result_queue: queue.Queue):
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.infer_queue = queue.Queue(maxsize=INFER_QUEUE_MAX)

        self._cam_thread = None
        self._infer_thread = None
        self._running = False
        self._cap = None

        self.clf = Classifier()
        self.logger = Logger()

        # Phase 1 components
        self.cascade = None
        if os.path.exists(CASCADE_PATH):
            self.cascade = cv2.CascadeClassifier(CASCADE_PATH)
        self.bgs = cv2.createBackgroundSubtractorMOG2(
            history=BGS_HISTORY,
            varThreshold=BGS_VAR_THRESHOLD,
            detectShadows=False,
        )
        self._scan_x_pct = SCAN_X_PCT
        self._tracked = {}
        self._next_id = 0
        self._processed_ids = set()

        # Debug crops
        self._debug_idx = 0
        if DEBUG_MODE:
            os.makedirs(DEBUG_DIR, exist_ok=True)

    # ── Lifecycle ───────────────────────────────────────────────────────

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._cam_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._infer_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._cam_thread.start()
        self._infer_thread.start()

    def stop(self) -> None:
        self._running = False
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def enqueue_for_inference(self, frame_bgr) -> None:
        """Called by GUI after display to feed the inference queue."""
        if not self._running or frame_bgr is None:
            return
        if self.infer_queue.full():
            try:
                self.infer_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.infer_queue.put_nowait(frame_bgr)
        except queue.Full:
            pass

    # ── Threads ────────────────────────────────────────────────────────

    def _camera_loop(self) -> None:
        self._cap = cv2.VideoCapture(CAMERA_ID)
        if not self._cap.isOpened():
            self._push_status("Camera error: unable to open device")
            self._running = False
            return

        interval = 1.0 / max(1, CAMERA_FPS)
        while self._running:
            t0 = time.time()
            ret, frame = self._cap.read()
            if ret:
                self._push_frame(frame)
            _safe_sleep(t0, interval)

    def _inference_loop(self) -> None:
        interval = 1.0 / max(1, INFERENCE_FPS)
        while self._running:
            t0 = time.time()
            try:
                frame = self.infer_queue.get(timeout=0.5)
            except queue.Empty:
                _safe_sleep(t0, interval)
                continue

            if frame is not None:
                self._process_frame(frame)

            _safe_sleep(t0, interval)

    # ── Phase 1 + Phase 2 ──────────────────────────────────────────────

    def _process_frame(self, frame_bgr) -> None:
        H, W = frame_bgr.shape[:2]
        scan_x = int(W * self._scan_x_pct)

        boxes = self._detect(frame_bgr, W, H)

        for box in boxes:
            tid = self._match_or_create(box)
            x, y, bw, bh = box
            cx = x + bw // 2

            if cx >= scan_x and tid not in self._processed_ids:
                self._processed_ids.add(tid)
                x1 = max(0, x - 10); y1 = max(0, y - 10)
                x2 = min(W, x + bw + 10); y2 = min(H, y + bh + 10)
                crop = frame_bgr[y1:y2, x1:x2]

                if crop.size > 0:
                    if DEBUG_MODE:
                        self._save_debug_crop(crop, tid)
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    result = self.clf.classify(crop_rgb)
                    tuber_id = self.logger.log(result)
                    stats = self.logger.stats()
                    self._push_result(result, stats, tuber_id)

        if len(self._processed_ids) > 500:
            self._processed_ids = set(list(self._processed_ids)[-200:])

    def _detect(self, frame_bgr, W, H):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sob = np.uint8(np.clip(np.sqrt(sx**2 + sy**2), 0, 255))

        if self.cascade is not None and not self.cascade.empty():
            small = cv2.resize(sob, (max(1, W // 10), max(1, H // 10)))
            dets = self.cascade.detectMultiScale(
                small, scaleFactor=1.1, minNeighbors=4, minSize=(25, 25)
            )
            if len(dets):
                return [(x * 10, y * 10, w * 10, h * 10) for (x, y, w, h) in dets]

        mask = self.bgs.apply(gray)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for c in cnts:
            area = cv2.contourArea(c)
            if 600 < area < 90000:
                x, y, bw, bh = cv2.boundingRect(c)
                if 0.3 < (bw / max(bh, 1)) < 3.0:
                    boxes.append((x, y, bw, bh))
        return boxes

    def _match_or_create(self, box) -> int:
        best_iou = 0.0
        best_tid = -1
        for tid, tb in self._tracked.items():
            iou = self._iou(box, tb)
            if iou > best_iou:
                best_iou = iou
                best_tid = tid
        if best_iou >= 0.40:
            self._tracked[best_tid] = box
            return best_tid
        self._next_id += 1
        self._tracked[self._next_id] = box
        if len(self._tracked) > 80:
            oldest = min(self._tracked.keys())
            del self._tracked[oldest]
        return self._next_id

    @staticmethod
    def _iou(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix1 = max(ax, bx); iy1 = max(ay, by)
        ix2 = min(ax + aw, bx + bw); iy2 = min(ay + ah, by + bh)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    # ── Queue helpers ──────────────────────────────────────────────────

    def _push_frame(self, frame_bgr) -> None:
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.frame_queue.put_nowait(frame_bgr)
        except queue.Full:
            pass

    def _push_result(self, result: dict, stats: dict, tuber_id: int) -> None:
        packet = {
            "type": "result",
            "result": result,
            "stats": stats,
            "tuber_id": tuber_id,
        }
        if self.result_queue.full():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.result_queue.put_nowait(packet)
        except queue.Full:
            pass

    def _push_status(self, message: str) -> None:
        packet = {"type": "status", "message": message}
        if self.result_queue.full():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.result_queue.put_nowait(packet)
        except queue.Full:
            pass

    def _save_debug_crop(self, crop_bgr, tid: int) -> None:
        self._debug_idx += 1
        ts = int(time.time() * 1000)
        name = f"tuber_{tid:04d}_{ts}_{self._debug_idx:06d}.jpg"
        path = os.path.join(DEBUG_DIR, name)
        cv2.imwrite(path, crop_bgr)
