#!/usr/bin/env python3
"""Terroir AI — QA System Dashboard v6"""

import tkinter as tk
from tkinter import filedialog, messagebox
import threading, time, queue, os, sys, random
import cv2, numpy as np
from PIL import Image, ImageTk, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classifier import Classifier
from logger     import Logger
from diseases   import DISEASE_DB

C = {
    "bg":        "#09090B",
    "surface":   "#18181B",
    "surface2":  "#27272A",
    "surface3":  "#3F3F46",
    "border":    "#27272A",
    "border2":   "#3F3F46",
    "green":     "#22C55E",
    "green_dk":  "#15803D",
    "green_bg":  "#052E16",
    "yellow":    "#EAB308",
    "yellow_bg": "#422006",
    "red":       "#EF4444",
    "red_bg":    "#450A0A",
    "blue":      "#3B82F6",
    "blue_bg":   "#0F1C3F",
    "text":      "#FAFAFA",
    "text2":     "#A1A1AA",
    "text3":     "#71717A",
    "text4":     "#52525B",
    "slate":     "#334155",
    "slate2":    "#475569",
    "slate_text":"#E2E8F0",
}

GRADE_MAP = {
    "Healthy":                ("A", C["green"],  C["green_bg"]),
    "Sprouted_Potato":        ("C", C["yellow"], C["yellow_bg"]),
    "Damaged_Potato":         ("D", C["red"],    C["red_bg"]),
    "Defected_Potato":        ("D", C["red"],    C["red_bg"]),
    "Diseased_Fungal_Potato": ("F", C["red"],    C["red_bg"]),
    "Late_Blight":            ("F", C["red"],    C["red_bg"]),
    "Early_Blight":           ("D", C["red"],    C["red_bg"]),
    "Dry_Rot":                ("F", C["red"],    C["red_bg"]),
    "Common_Scab":            ("C", C["yellow"], C["yellow_bg"]),
    "Silver_Scurf":           ("B", C["yellow"], C["yellow_bg"]),
    "Black_Dot":              ("B", C["yellow"], C["yellow_bg"]),
    "Soft_Rot":               ("F", C["red"],    C["red_bg"]),
    "Wart_Disease":           ("F", C["red"],    C["red_bg"]),
    "Mechanical_Damage":      ("C", C["yellow"], C["yellow_bg"]),
    "Gangrene":               ("F", C["red"],    C["red_bg"]),
}

ACTION_COLOR = {
    "KEEP":      C["green"],
    "REJECT":    C["red"],
    "SECONDARY": C["yellow"],
    "REVIEW":    C["blue"],
}

F = {
    "logo":    ("Helvetica", 15, "bold"),
    "h1":      ("Helvetica", 20, "bold"),
    "h2":      ("Helvetica", 14, "bold"),
    "h3":      ("Helvetica", 11, "bold"),
    "body":    ("Helvetica", 10),
    "sm":      ("Helvetica", 9),
    "mono":    ("Courier", 10),
    "mono_sm": ("Courier", 9),
    "stat_xl": ("Helvetica", 36, "bold"),
    "stat_lg": ("Helvetica", 24, "bold"),
    "stat_md": ("Helvetica", 16, "bold"),
    "badge":   ("Helvetica", 9, "bold"),
    "grade":   ("Helvetica", 72, "bold"),
    "btn":     ("Helvetica", 11, "bold"),
}


def make_btn(parent, text, command, primary=False):
    """Mac-compatible button using Frame+Label trick."""
    bg   = C["green"]   if primary else C["slate"]
    fg   = "#000000"    if primary else C["slate_text"]
    hbg  = C["green_dk"] if primary else C["slate2"]

    frame = tk.Frame(parent, bg=bg, cursor="hand2")
    lbl   = tk.Label(frame, text=text, font=F["btn"],
                     bg=bg, fg=fg, padx=20, pady=10)
    lbl.pack()

    def on_enter(e):
        frame.config(bg=hbg); lbl.config(bg=hbg)
    def on_leave(e):
        frame.config(bg=frame._bg); lbl.config(bg=frame._bg)
    def on_click(e):
        command()

    frame._bg = bg
    frame.bind("<Enter>", on_enter)
    frame.bind("<Leave>", on_leave)
    frame.bind("<Button-1>", on_click)
    lbl.bind("<Enter>", on_enter)
    lbl.bind("<Leave>", on_leave)
    lbl.bind("<Button-1>", on_click)

    return frame


class ScanOverlay:
    def __init__(self, canvas):
        self.canvas   = canvas
        self._y       = 0
        self._dir     = 1
        self._running = False

    def start(self):
        if self._running: return
        self._running = True
        self._animate()

    def stop(self):
        self._running = False
        try: self.canvas.delete("scanline")
        except: pass

    def _animate(self):
        if not self._running: return
        try: self.canvas.delete("scanline")
        except: pass
        W = self.canvas.winfo_width()
        H = self.canvas.winfo_height()
        if W > 10:
            for i in range(4, 0, -1):
                self.canvas.create_line(0, self._y-i, W, self._y-i,
                                         fill="#22C55E22", width=1, tags="scanline")
                self.canvas.create_line(0, self._y+i, W, self._y+i,
                                         fill="#22C55E22", width=1, tags="scanline")
            self.canvas.create_line(0, self._y, W, self._y,
                                     fill=C["green"], width=2, tags="scanline")
            self._y += self._dir * 4
            if self._y >= H: self._dir = -1
            if self._y <= 0:  self._dir = 1
        self.canvas.after(18, self._animate)


class ConveyorCanvas(tk.Canvas):
    def __init__(self, parent, on_scan_cb, **kw):
        kw.pop("bg", None)
        super().__init__(parent, bg=C["surface"], highlightthickness=0, **kw)
        self.on_scan_cb  = on_scan_cb
        self._potatoes   = []
        self._next_spawn = 50
        self._frame      = 0
        self._running    = False
        self._scan_x_pct = 0.55
        self._processed  = set()
        self._results    = {}

    def start(self):
        self._running = True
        self._animate()

    def stop(self):
        self._running = False

    def mark_result(self, pid, action):
        self._results[pid] = ACTION_COLOR.get(action, C["text3"])

    def _animate(self):
        if not self._running: return
        self._frame += 1
        self.delete("potato")
        self.delete("belt")
        W = self.winfo_width(); H = self.winfo_height()
        if W < 10: self.after(33, self._animate); return
        BY1, BY2 = int(H*.12), int(H*.88)
        sx = int(W * self._scan_x_pct)
        self.create_rectangle(0, BY1, W, BY2, fill="#1C1C1F", outline="", tags="belt")
        off = (self._frame * 3) % 40
        for i in range(-1, W//40+2):
            x = i*40+off
            self.create_rectangle(x, BY1+3, x+20, BY2-3,
                                   fill="#1F1F23", outline="", tags="belt")
        self.create_rectangle(0, BY1, W, BY1+3, fill=C["border2"], outline="", tags="belt")
        self.create_rectangle(0, BY2-3, W, BY2, fill=C["border2"], outline="", tags="belt")
        for i in range(5, 0, -1):
            self.create_line(sx, BY1, sx, BY2, fill=C["green"],
                              width=i, tags="belt")
        self.create_line(sx, 0, sx, H, fill=C["green"], width=2, tags="belt")
        self.create_rectangle(sx+2, 2, sx+40, 16,
                               fill=C["green_bg"], outline=C["green"], tags="belt")
        self.create_text(sx+21, 9, text="SCAN", fill=C["green"],
                          font=("Helvetica", 7, "bold"), tags="belt")
        self._next_spawn -= 1
        if self._next_spawn <= 0:
            self._next_spawn = max(14, int(random.gauss(46, 8)))
            mid = (BY1+BY2)//2
            sz = random.randint(28, 50)
            sh = random.randint(110, 180)
            pid = self._frame*100 + len(self._potatoes)
            self._potatoes.append({
                "id": pid, "x": -sz,
                "y":  mid + random.randint(-14, 14),
                "sz": sz, "spd": random.uniform(2.2, 3.8),
                "col": (int(sh*.46), int(sh*.66), sh),
            })
        bm = (BY1+BY2)//2
        dead = []
        for p in self._potatoes:
            p["x"] += p["spd"]
            if p["x"] > W+80: dead.append(p); continue
            x, y, s = int(p["x"]), int(p["y"]), p["sz"]
            r, g, b  = p["col"]
            fill = f"#{r:02x}{g:02x}{b:02x}"
            hi   = f"#{min(255,r+38):02x}{min(255,g+38):02x}{min(255,b+38):02x}"
            rc   = self._results.get(p["id"])
            oc   = rc if rc else C["border"]
            self.create_oval(x-s+2, y-int(s*.73)+3, x+s+2, y+int(s*.73)+3,
                              fill="#060608", outline="", tags="potato")
            self.create_oval(x-s, y-int(s*.75), x+s, y+int(s*.75),
                              fill=fill, outline=oc, width=2 if rc else 1, tags="potato")
            self.create_oval(x-s//2-2, y-s//2, x+s//8, y-s//8,
                              fill=hi, outline="", tags="potato")
            if p["x"] >= sx and p["id"] not in self._processed:
                self._processed.add(p["id"])
                self.on_scan_cb(p)
        for p in dead: self._potatoes.remove(p)
        if len(self._processed) > 400:
            self._processed = set(list(self._processed)[-150:])
        self.after(33, self._animate)


class TerriorApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Terroir AI — QA System")
        self.root.configure(bg=C["bg"])
        self.root.geometry("1500x900")
        self.root.minsize(1200, 750)

        self.clf       = Classifier()
        self.logger    = Logger()
        self._running  = False
        self._result_q = queue.Queue(maxsize=8)
        self._cam = None
        self._cam_running = False
        self._img_ref = None
        self._last_result = None
        self._hist_rows = []

        self._build_ui()
        self._tick_clock()
        self._poll_results()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── BUILD ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self._build_header()
        tk.Frame(self.root, bg=C["border"], height=1).pack(fill="x")
        body = tk.Frame(self.root, bg=C["bg"])
        body.pack(fill="both", expand=True, padx=14, pady=10)
        left = tk.Frame(body, bg=C["bg"])
        left.pack(side="left", fill="both", expand=True, padx=(0,8))
        right = tk.Frame(body, bg=C["bg"], width=420)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)
        self._build_camera_panel(left)
        self._build_conveyor_panel(left)
        self._build_right_panel(right)
        tk.Frame(self.root, bg=C["border"], height=1).pack(fill="x")
        foot = tk.Frame(self.root, bg=C["surface"], height=28)
        foot.pack(fill="x")
        foot.pack_propagate(False)
        self.footer_lbl = tk.Label(foot, text="Terroir AI QA System  ·  Ready",
                                    font=F["mono_sm"], bg=C["surface"], fg=C["text4"])
        self.footer_lbl.pack(side="left", padx=14, pady=5)
        tk.Label(foot,
                 text="Dhruv Jain  ·  Emerson High School, McKinney TX  ·  Powered by YOLOv8",
                 font=F["sm"], bg=C["surface"], fg=C["text4"]).pack(side="right", padx=14)

    def _build_header(self):
        hdr = tk.Frame(self.root, bg=C["bg"], height=56)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        ll = tk.Frame(hdr, bg=C["bg"])
        ll.pack(side="left", padx=18, pady=10)

        # Plant icon drawn on canvas
        icon_c = tk.Canvas(ll, width=28, height=28, bg=C["bg"],
                            highlightthickness=0)
        icon_c.pack(side="left", padx=(0,8))
        icon_c.create_oval(4, 4, 24, 24, fill=C["green_bg"], outline=C["green"], width=2)
        icon_c.create_line(14, 20, 14, 8,  fill=C["green"], width=2)
        icon_c.create_line(14, 14, 9,  10, fill=C["green"], width=2)
        icon_c.create_line(14, 11, 19, 7,  fill=C["green"], width=2)

        tk.Label(ll, text="TERROIR AI", font=F["logo"],
                 bg=C["bg"], fg=C["green"]).pack(side="left")
        tk.Label(ll, text="  QA SYSTEM", font=("Helvetica", 10),
                 bg=C["bg"], fg=C["text3"]).pack(side="left", pady=2)

        rr = tk.Frame(hdr, bg=C["bg"])
        rr.pack(side="right", padx=18, pady=10)

        self.clock_lbl = tk.Label(rr, text="", font=F["mono"],
                                   bg=C["surface2"], fg=C["text2"], padx=12, pady=4)
        self.clock_lbl.pack(side="right", padx=(6,0))

        status = tk.Frame(rr, bg=C["green_bg"], padx=10, pady=4)
        status.pack(side="right", padx=(6,0))
        self._pulse_dot = tk.Label(status, text="●", font=("Helvetica", 9),
                                    bg=C["green_bg"], fg=C["green"])
        self._pulse_dot.pack(side="left")
        tk.Label(status, text=" SYSTEM ONLINE", font=F["badge"],
                 bg=C["green_bg"], fg=C["green"]).pack(side="left")

        self.inf_badge = tk.Label(rr, text="— ms", font=F["badge"],
                                   bg=C["blue_bg"], fg=C["blue"], padx=8, pady=4)
        self.inf_badge.pack(side="right", padx=(6,0))

        self.clf_badge = tk.Label(rr, text="—", font=F["badge"],
                                   bg=C["surface2"], fg=C["text2"], padx=8, pady=4)
        self.clf_badge.pack(side="right", padx=(6,0))

    def _build_camera_panel(self, parent):
        panel = self._card(parent)
        panel.pack(fill="both", expand=True, pady=(0,8))

        ph = tk.Frame(panel, bg=C["surface"])
        ph.pack(fill="x", padx=14, pady=(10,6))
        tk.Label(ph, text="LIVE CAMERA FEED", font=F["h3"],
                 bg=C["surface"], fg=C["text2"]).pack(side="left")
        self.scan_status = tk.Label(ph, text="● STANDBY", font=F["badge"],
                                     bg=C["surface2"], fg=C["text3"], padx=8, pady=3)
        self.scan_status.pack(side="right")
        self.ts_lbl = tk.Label(ph, text="", font=F["mono_sm"],
                                bg=C["surface"], fg=C["text4"])
        self.ts_lbl.pack(side="right", padx=10)

        cam_wrap = tk.Frame(panel, bg=C["border2"], padx=1, pady=1)
        cam_wrap.pack(fill="both", expand=True, padx=14)
        self.cam_canvas = tk.Canvas(cam_wrap, bg="#0A0A0D", highlightthickness=0)
        self.cam_canvas.pack(fill="both", expand=True)
        self._scan_overlay = ScanOverlay(self.cam_canvas)
        self.cam_canvas.after(200, self._draw_placeholder)

        # Buttons
        btn_row = tk.Frame(panel, bg=C["surface"])
        btn_row.pack(fill="x", padx=14, pady=12)

        self.start_btn = make_btn(btn_row, "START SCAN",
                                   self._toggle_sim, primary=True)
        self.start_btn.pack(side="left", padx=(0,8))

        make_btn(btn_row, "LOAD IMAGE",
                 self._load_image).pack(side="left", padx=(0,8))
        make_btn(btn_row, "LIVE CAMERA",
                 self._toggle_camera).pack(side="left", padx=(0,8))
        make_btn(btn_row, "EXPORT LOG",
                 self._export_log).pack(side="right")

    def _draw_placeholder(self):
        self.cam_canvas.delete("all")
        W = self.cam_canvas.winfo_width() or 600
        H = self.cam_canvas.winfo_height() or 300
        for i in range(0, W, 50):
            self.cam_canvas.create_line(i, 0, i, H, fill="#111115", width=1)
        for i in range(0, H, 50):
            self.cam_canvas.create_line(0, i, W, i, fill="#111115", width=1)
        cx, cy = W//2, H//2
        for r in [55, 85]:
            self.cam_canvas.create_oval(cx-r, cy-r, cx+r, cy+r,
                                         outline=C["border2"], width=1)
        self.cam_canvas.create_line(cx-100, cy, cx+100, cy, fill=C["border2"], width=1)
        self.cam_canvas.create_line(cx, cy-100, cx, cy+100, fill=C["border2"], width=1)
        self.cam_canvas.create_text(cx, cy+115,
                                     text="Press START SCAN to begin  or  LOAD IMAGE",
                                     fill=C["text4"], font=F["sm"])
        sz = 18
        for bx, by in [(22,22),(W-22,22),(22,H-22),(W-22,H-22)]:
            dx = 1 if bx < W//2 else -1
            dy = 1 if by < H//2 else -1
            self.cam_canvas.create_line(bx, by, bx+dx*sz, by, fill=C["text3"], width=2)
            self.cam_canvas.create_line(bx, by, bx, by+dy*sz, fill=C["text3"], width=2)

    def _build_conveyor_panel(self, parent):
        panel = self._card(parent)
        panel.pack(fill="x")
        ph = tk.Frame(panel, bg=C["surface"])
        ph.pack(fill="x", padx=14, pady=(8,4))
        tk.Label(ph, text="CONVEYOR SIMULATION", font=F["h3"],
                 bg=C["surface"], fg=C["text2"]).pack(side="left")
        tk.Label(ph, text="Real-time belt tracking", font=F["sm"],
                 bg=C["surface"], fg=C["text4"]).pack(side="left", padx=8)
        wrap = tk.Frame(panel, bg=C["border2"], padx=1, pady=1)
        wrap.pack(fill="x", padx=14, pady=(0,10))
        self.conveyor = ConveyorCanvas(wrap, on_scan_cb=self._on_conveyor_scan, height=88)
        self.conveyor.pack(fill="x")

    def _build_right_panel(self, parent):
        self._build_grade_card(parent)
        self._build_stats_row(parent)
        self._build_details_card(parent)
        self._build_history_card(parent)

    def _build_grade_card(self, parent):
        card = self._card(parent)
        card.pack(fill="x", pady=(0,8))

        hdr = tk.Frame(card, bg=C["surface"])
        hdr.pack(fill="x", padx=14, pady=(10,0))
        tk.Label(hdr, text="CURRENT ITEM STATUS", font=F["h3"],
                 bg=C["surface"], fg=C["text2"]).pack(side="left")
        self.action_badge = tk.Label(hdr, text="STANDBY", font=F["badge"],
                                      bg=C["surface2"], fg=C["text3"], padx=8, pady=3)
        self.action_badge.pack(side="right")

        grade_row = tk.Frame(card, bg=C["surface"])
        grade_row.pack(fill="x", padx=14, pady=6)

        self.grade_lbl = tk.Label(grade_row, text="—", font=F["grade"],
                                   bg=C["surface"], fg=C["text4"], width=2)
        self.grade_lbl.pack(side="left")

        info_col = tk.Frame(grade_row, bg=C["surface"])
        info_col.pack(side="left", padx=12, fill="y", expand=True)
        self.disease_lbl = tk.Label(info_col, text="Awaiting scan...",
                                     font=F["h2"], bg=C["surface"],
                                     fg=C["text"], anchor="w")
        self.disease_lbl.pack(anchor="w")
        self.grade_desc = tk.Label(info_col, text="No item detected",
                                    font=F["sm"], bg=C["surface"],
                                    fg=C["text3"], anchor="w", wraplength=220)
        self.grade_desc.pack(anchor="w", pady=(4,0))

        conf_frame = tk.Frame(card, bg=C["surface"])
        conf_frame.pack(fill="x", padx=14, pady=(0,6))
        conf_hdr = tk.Frame(conf_frame, bg=C["surface"])
        conf_hdr.pack(fill="x")
        tk.Label(conf_hdr, text="CONFIDENCE", font=F["badge"],
                 bg=C["surface"], fg=C["text4"]).pack(side="left")
        self.conf_pct = tk.Label(conf_hdr, text="—", font=F["badge"],
                                  bg=C["surface"], fg=C["text3"])
        self.conf_pct.pack(side="right")
        track = tk.Frame(conf_frame, bg=C["surface2"], height=6)
        track.pack(fill="x", pady=(3,0))
        self.conf_bar = tk.Frame(track, bg=C["text4"], height=6)
        self.conf_bar.place(relwidth=0.0, relheight=1.0)

        tk.Frame(card, bg=C["surface"], height=8).pack()

    def _build_stats_row(self, parent):
        row = tk.Frame(parent, bg=C["bg"])
        row.pack(fill="x", pady=(0,8))
        self._stats = {}
        for i, (key, label, color) in enumerate([
            ("total", "TOTAL SCANNED", C["text"]),
            ("defect","DEFECT RATE",   C["red"]),
            ("rate",  "ITEMS / MIN",   C["blue"]),
        ]):
            card = self._card(row)
            card.pack(side="left", fill="both", expand=True,
                      padx=(0,6) if i < 2 else 0)
            tk.Label(card, text=label, font=F["badge"],
                     bg=C["surface"], fg=C["text3"], pady=8).pack()
            v = tk.Label(card, text="0", font=F["stat_lg"],
                         bg=C["surface"], fg=color)
            v.pack(pady=(0,10))
            self._stats[key] = v

    def _build_details_card(self, parent):
        card = self._card(parent)
        card.pack(fill="x", pady=(0,8))
        tk.Label(card, text="DISEASE DETAILS", font=F["h3"],
                 bg=C["surface"], fg=C["text2"],
                 padx=14, pady=10).pack(anchor="w")
        tk.Frame(card, bg=C["border"], height=1).pack(fill="x", padx=14)
        self._detail_fields = {}
        for label, key in [
            ("PATHOGEN",    "path_lbl"),
            ("DESCRIPTION", "desc_lbl"),
            ("TREATMENT",   "tx_lbl"),
            ("MARKET",      "mkt_lbl"),
            ("SPREAD RISK", "risk_lbl"),
        ]:
            r = tk.Frame(card, bg=C["surface"])
            r.pack(fill="x", padx=14, pady=3)
            tk.Label(r, text=label, font=F["badge"],
                     bg=C["surface"], fg=C["text4"],
                     width=11, anchor="w").pack(side="left")
            lbl = tk.Label(r, text="—", font=F["sm"],
                           bg=C["surface"], fg=C["text3"],
                           anchor="w", wraplength=240, justify="left")
            lbl.pack(side="left", fill="x", expand=True)
            self._detail_fields[key] = lbl
        tk.Frame(card, bg=C["surface"], height=8).pack()

    def _build_history_card(self, parent):
        card = self._card(parent)
        card.pack(fill="both", expand=True)
        hdr = tk.Frame(card, bg=C["surface"])
        hdr.pack(fill="x", padx=14, pady=(10,4))
        tk.Label(hdr, text="RECENT SCANS", font=F["h3"],
                 bg=C["surface"], fg=C["text2"]).pack(side="left")
        self.hist_count = tk.Label(hdr, text="Last 8 items", font=F["sm"],
                                    bg=C["surface"], fg=C["text4"])
        self.hist_count.pack(side="right")
        col_hdr = tk.Frame(card, bg=C["surface2"])
        col_hdr.pack(fill="x", padx=14)
        for txt, w, side in [
            ("#",4,"left"),("TIME",7,"left"),("GR",3,"left"),
            ("DISEASE",0,"left"),("CONF",4,"right")
        ]:
            tk.Label(col_hdr, text=txt, font=F["badge"],
                     bg=C["surface2"], fg=C["text4"],
                     width=w, anchor="w", pady=4).pack(
                         side="left" if side=="left" else "right",
                         padx=(8 if txt=="#" else 2, 0),
                         fill="x" if txt=="DISEASE" else None,
                         expand=(txt=="DISEASE"))
        self.hist_frame = tk.Frame(card, bg=C["surface"])
        self.hist_frame.pack(fill="both", expand=True, padx=14, pady=(2,10))

    def _add_history(self, result, tid):
        disease = result.get("disease","?")
        conf    = result.get("confidence", 0.0)
        grade, col, gbg = GRADE_MAP.get(disease, ("?", C["text3"], C["surface2"]))
        ts = time.strftime("%H:%M:%S")
        row = tk.Frame(self.hist_frame, bg=C["surface2"], pady=3)
        row.pack(fill="x", pady=1)
        tk.Frame(row, bg=col, width=3).pack(side="left", fill="y")
        tk.Label(row, text=f"{tid:03d}", font=F["mono_sm"],
                 bg=C["surface2"], fg=C["text4"], width=4).pack(side="left", padx=4)
        tk.Label(row, text=ts, font=F["mono_sm"],
                 bg=C["surface2"], fg=C["text4"], width=8).pack(side="left")
        tk.Label(row, text=f" {grade} ", font=F["badge"],
                 bg=gbg, fg=col, width=3).pack(side="left", padx=4)
        tk.Label(row, text=disease.replace("_"," ")[:18], font=F["sm"],
                 bg=C["surface2"], fg=C["text"],
                 anchor="w").pack(side="left", fill="x", expand=True)
        tk.Label(row, text=f"{conf*100:.0f}%", font=F["mono_sm"],
                 bg=C["surface2"], fg=col, width=4).pack(side="right", padx=6)
        self._hist_rows.append(row)
        if len(self._hist_rows) > 8:
            self._hist_rows.pop(0).destroy()
        self.hist_count.config(text=f"{tid} total")

    def _card(self, parent):
        return tk.Frame(parent, bg=C["surface"],
                        highlightbackground=C["border"],
                        highlightthickness=1)

    # ── CLOCK ─────────────────────────────────────────────────────────────────

    def _tick_clock(self):
        self.clock_lbl.config(text=f"  {time.strftime('%a %b %d  %H:%M:%S')}  ")
        self._pulse_dot.config(fg=C["green"] if int(time.time()*2)%2 else C["green_dk"])
        self.root.after(500, self._tick_clock)

    # ── SIMULATION ────────────────────────────────────────────────────────────

    def _toggle_sim(self):
        if not self._running:
            self._running = True
            lbl = self.start_btn.winfo_children()[0]
            self.start_btn.config(bg=C["red"])
            lbl.config(text="STOP SCAN", bg=C["red"], fg=C["text"])
            self.start_btn._bg = C["red"]
            self.scan_status.config(text="● SCANNING",
                                     bg=C["green_bg"], fg=C["green"])
            self.conveyor.start()
            self._scan_overlay.start()
        else:
            self._running = False
            lbl = self.start_btn.winfo_children()[0]
            self.start_btn.config(bg=C["green"])
            lbl.config(text="START SCAN", bg=C["green"], fg="#000000")
            self.start_btn._bg = C["green"]
            self.scan_status.config(text="● STANDBY",
                                     bg=C["surface2"], fg=C["text3"])
            self.conveyor.stop()
            self._scan_overlay.stop()
            self._draw_placeholder()

    def _on_conveyor_scan(self, potato):
        roll = random.random()
        if roll < 0.62:
            disease, conf = "Healthy",                random.uniform(0.80, 0.97)
        elif roll < 0.75:
            disease, conf = "Sprouted_Potato",        random.uniform(0.70, 0.88)
        elif roll < 0.84:
            disease, conf = "Damaged_Potato",         random.uniform(0.72, 0.90)
        elif roll < 0.92:
            disease, conf = "Diseased_Fungal_Potato", random.uniform(0.74, 0.92)
        else:
            disease, conf = "Defected_Potato",        random.uniform(0.68, 0.86)

        info   = DISEASE_DB.get(disease, {})
        action = info.get("action", "KEEP")
        result = {
            "disease": disease, "confidence": conf,
            "action": action, "classifier": "Simulation",
            "notifiable": info.get("notifiable", False),
            "spread_risk": info.get("spread_risk",""),
            "treatment": info.get("treatment",""),
            "market_value": info.get("market_value",""),
            "description": info.get("description",""),
            "category": info.get("category",""),
            "surface_pct": 0.0,
            "inference_ms": random.uniform(18, 55),
        }
        self._result_q.put(("sim", result, None, potato["id"]))

    # ── LOAD IMAGE ────────────────────────────────────────────────────────────

    def _load_image(self):
        path = filedialog.askopenfilename(
            title="Select Potato Image",
            filetypes=[("Images","*.jpg *.jpeg *.png *.bmp *.tiff"),("All","*.*")]
        )
        if not path: return
        bgr = cv2.imread(path)
        if bgr is None:
            messagebox.showerror("Error", f"Cannot read:\n{path}"); return

        self.footer_lbl.config(
            text=f"Analysing: {os.path.basename(path)}...")
        self.scan_status.config(text="● ANALYSING",
                                 bg=C["blue_bg"], fg=C["blue"])
        self._scan_overlay.start()

        bgr_copy = bgr.copy()
        rgb_copy = cv2.cvtColor(bgr_copy, cv2.COLOR_BGR2RGB)

        def _run():
            try:
                result = self.clf.classify(rgb_copy)
            except Exception as e:
                info   = DISEASE_DB.get("Healthy", {})
                result = {
                    "disease": "Healthy", "confidence": 0.60,
                    "action": "REVIEW", "classifier": "Fallback",
                    "notifiable": False, "spread_risk": "",
                    "treatment": "", "market_value": "",
                    "description": str(e), "category": "",
                    "surface_pct": 0.0, "inference_ms": 0,
                }
            self._result_q.put(("image", result, bgr_copy, None))

        threading.Thread(target=_run, daemon=True).start()

    # ── CAMERA ────────────────────────────────────────────────────────────────

    def _toggle_camera(self):
        if self._cam_running:
            self._cam_running = False
            if self._cam: self._cam.release()
            self._scan_overlay.stop()
            self.scan_status.config(text="● STANDBY",
                                     bg=C["surface2"], fg=C["text3"])
        else:
            self._cam = cv2.VideoCapture(0)
            if not self._cam.isOpened():
                messagebox.showerror("Camera Error","Cannot open camera."); return
            self._cam_running = True
            self._scan_overlay.start()
            self.scan_status.config(text="● LIVE",
                                     bg=C["green_bg"], fg=C["green"])
            threading.Thread(target=self._cam_reader, daemon=True).start()
            self._cam_display_loop()

    def _cam_reader(self):
        self._latest_frame = None
        while self._cam_running:
            ret, frame = self._cam.read()
            if ret: self._latest_frame = frame
            time.sleep(0.033)

    def _cam_display_loop(self):
        if not self._cam_running: return
        f = getattr(self, "_latest_frame", None)
        if f is not None:
            self._show_frame(f, self._last_result)
            if not hasattr(self,"_last_clf") or time.time()-self._last_clf > 1.5:
                self._last_clf = time.time()
                rgb = cv2.cvtColor(f.copy(), cv2.COLOR_BGR2RGB)
                def _run(r=rgb, b=f.copy()):
                    try:
                        res = self.clf.classify(r)
                        self._result_q.put(("cam", res, b, None))
                    except: pass
                threading.Thread(target=_run, daemon=True).start()
        self.root.after(40, self._cam_display_loop)

    # ── SHOW FRAME ────────────────────────────────────────────────────────────

    def _show_frame(self, bgr, result=None):
        self.cam_canvas.update_idletasks()
        W = self.cam_canvas.winfo_width()  or 600
        H = self.cam_canvas.winfo_height() or 300
        if W < 10: return
        rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img  = Image.fromarray(rgb).resize((W, H), Image.LANCZOS)
        draw = ImageDraw.Draw(img)
        if result:
            action  = result.get("action","KEEP")
            disease = result.get("disease","")
            col_hex = ACTION_COLOR.get(action, "#666666")
            cr = int(col_hex[1:3],16)
            cg = int(col_hex[3:5],16)
            cb_v = int(col_hex[5:7],16)
            for t in range(3):
                draw.rectangle([t,t,W-1-t,H-1-t], outline=(cr,cg,cb_v))
            grade, gc, gbg = GRADE_MAP.get(disease, ("?",cr,cg))
            txt = f"  Grade {grade} — {action}  "
            draw.rectangle([6,6,6+len(txt)*7,26], fill=(cr//6,cg//6,cb_v//6))
            draw.text((8,8), txt, fill=(cr,cg,cb_v))
        tk_img = ImageTk.PhotoImage(img)
        self.cam_canvas.delete("all")
        self.cam_canvas.create_image(0, 0, anchor="nw", image=tk_img)
        self._img_ref = tk_img
        self.ts_lbl.config(text=time.strftime("%H:%M:%S"))

    # ── RESULT POLLING ────────────────────────────────────────────────────────

    def _poll_results(self):
        try:
            while not self._result_q.empty():
                item   = self._result_q.get_nowait()
                kind   = item[0]; result = item[1]
                bgr    = item[2]; pid    = item[3] if len(item) > 3 else None

                tid = self.logger.log(result)
                s   = self.logger.stats()
                self._last_result = result

                self._update_grade_card(result)
                self._update_details(result)
                self._update_stats(s)
                self._add_history(result, tid)

                if kind == "image" and bgr is not None:
                    self._show_frame(bgr, result)
                    self._scan_overlay.stop()
                    self.scan_status.config(text="● STANDBY",
                                             bg=C["surface2"], fg=C["text3"])
                elif kind == "sim":
                    dummy = self._make_dummy(result)
                    self._show_frame(dummy, result)
                    if pid: self.conveyor.mark_result(pid, result.get("action","KEEP"))

                inf_ms = result.get("inference_ms", 0)
                self.inf_badge.config(text=f"{inf_ms:.0f} ms")
                self.clf_badge.config(text=result.get("classifier","—"))
                self.footer_lbl.config(
                    text=f"Terroir AI  ·  #{tid}  "
                         f"{result.get('action','?')}  ·  "
                         f"{result.get('disease','?').replace('_',' ')}  "
                         f"{result.get('confidence',0)*100:.0f}%  ·  {inf_ms:.0f}ms"
                )
        except Exception:
            pass
        self.root.after(80, self._poll_results)

    def _update_grade_card(self, result):
        disease = result.get("disease","?")
        conf    = result.get("confidence", 0.0)
        action  = result.get("action","KEEP")
        grade, col, gbg = GRADE_MAP.get(disease, ("?", C["text3"], C["surface2"]))
        info    = DISEASE_DB.get(disease, {})

        self.grade_lbl.config(text=grade, fg=col)
        self.disease_lbl.config(text=disease.replace("_"," "), fg=col)
        self.grade_desc.config(text=info.get("description","—")[:60], fg=C["text3"])
        self.conf_bar.place(relwidth=conf, relheight=1.0)
        self.conf_bar.config(bg=col)
        self.conf_pct.config(text=f"{conf*100:.1f}%", fg=col)
        ac = {"KEEP": C["green_bg"], "REJECT": C["red_bg"],
              "SECONDARY": C["yellow_bg"]}.get(action, C["surface2"])
        self.action_badge.config(text=action, bg=ac, fg=col)

    def _update_details(self, result):
        disease = result.get("disease","—")
        info    = DISEASE_DB.get(disease, {})
        action  = result.get("action","KEEP")
        col     = ACTION_COLOR.get(action, C["text3"])
        self._detail_fields["path_lbl"].config(
            text=disease.replace("_"," "), fg=col)
        self._detail_fields["desc_lbl"].config(
            text=info.get("description","—"), fg=C["text3"])
        self._detail_fields["tx_lbl"].config(
            text=info.get("treatment","—"),
            fg=C["yellow"] if action=="REJECT" else C["text3"])
        self._detail_fields["mkt_lbl"].config(
            text=info.get("market_value","—"), fg=C["text3"])
        risk = info.get("spread_risk","—")
        self._detail_fields["risk_lbl"].config(
            text=risk, fg=C["red"] if "CRITICAL" in risk else C["text3"])

    def _update_stats(self, s):
        total    = s.get("total", 0)
        rejected = s.get("rejected", 0)
        rate     = s.get("rate_per_min", 0.0)
        dr = f"{rejected/total*100:.1f}%" if total > 0 else "0.0%"
        self._stats["total"].config(text=str(total))
        self._stats["defect"].config(text=dr)
        self._stats["rate"].config(text=f"{rate:.0f}")

    def _make_dummy(self, result):
        W, H = 600, 320
        img  = np.zeros((H,W,3), dtype=np.uint8)
        img[:] = (9,9,12)
        cv2.rectangle(img,(0,H//4),(W,3*H//4),(18,20,26),-1)
        cx, cy = W//2, H//2
        cv2.ellipse(img,(cx,cy),(105,82),0,0,360,(70,108,148),-1)
        cv2.ellipse(img,(cx-28,cy-22),(35,27),20,0,360,(100,138,180),-1)
        if result.get("action") == "REJECT":
            for _ in range(random.randint(2,5)):
                lx = cx+random.randint(-60,60)
                ly = cy+random.randint(-40,40)
                cv2.ellipse(img,(lx,ly),(random.randint(8,28),
                            random.randint(6,18)),
                            random.randint(0,180),0,360,(8,12,5),-1)
        return img

    def _export_log(self):
        s = self.logger.stats()
        path = s.get("log_path","")
        if path and os.path.exists(path):
            messagebox.showinfo("Session Log",
                f"Log saved:\n{path}\n\n"
                f"Scanned: {s['total']}  ·  Rejected: {s['rejected']}\n"
                f"Value Protected: ${s['value_saved']:.2f}")
        else:
            messagebox.showinfo("Log","No session data yet.")

    def _on_close(self):
        self._running = False
        self._cam_running = False
        self.conveyor.stop()
        self._scan_overlay.stop()
        if self._cam: self._cam.release()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = TerriorApp()
    app.run()
