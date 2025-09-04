# color_gap_test_gui.py
# Single-file Tkinter app for the ring-gap color blindness test
# Requires: pip install pillow

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from math import atan2, degrees
from pathlib import Path
from PIL import Image, ImageTk
from datetime import datetime

# ───────────────────────────────
# CONFIG: put your 14 images here
# ───────────────────────────────
# Map: filename -> correct sector (1..8)
# Sectors are 8 equal slices, numbered CLOCKWISE with 1 at TOP (12 o'clock).
ANSWERS = {
    "szintevesztes teszt RG300_710_1.jpg": 7,
    "szintevesztes teszt RG280_710_2.jpg": 5,
    "szintevesztes teszt RG260_710_2.jpg": 7,
    "szintevesztes teszt RG240_710_2.jpg": 5,
    "szintevesztes teszt RG220_710_2.jpg": 8,
    "szintevesztes teszt RG200_710_2.jpg": 2,
    "szintevesztes teszt RG180_710_2.jpg": 6,
    "szintevesztes teszt RG160_710_1.jpg": 1,
    "szintevesztes teszt RG140_710_1.jpg": 4,
    "szintevesztes teszt RG120_710_1.jpg": 1,
    "szintevesztes teszt RG100_710_1.jpg": 5,
    "szintevesztes teszt RG080_710_1.jpg": 3,
    "szintevesztes teszt RG060_710_1.jpg": 2,
    "szintevesztes teszt RG040_710_1.jpg": 6,
}


# Optional: limit maximum display size (in pixels) to keep things tidy
MAX_DISPLAY_W = 900
MAX_DISPLAY_H = 900

# Ignore clicks too close to center (uncomment to enforce ring-only clicks)
IGNORE_CENTER = True
CENTER_MIN_RADIUS = 0.18  # relative (0..1 of half-diagonal-ish); tweak for your ring geometry

# ───────────────────────────────
# Helpers: angle/sector & images
# ───────────────────────────────
def angle_to_sector(angle_deg: float) -> int:
    """
    Convert a standard math angle (0°=right, CCW positive) into sectors:
    Sector 1 starts at 12 o'clock and increases CLOCKWISE in 45° steps.
    """
    # Convert to "clock" angle: 0°=up, clockwise positive
    a = (90.0 - angle_deg) % 360.0
    # Fix the sector calculation - sectors should be 1-8 based on 45° divisions
    sector = int((a + 22.5) // 45) + 1
    if sector < 1: sector = 1
    if sector > 8: sector = 8
    return sector

def rel_click_to_sector(rx: float, ry: float) -> int:
    """
    rx, ry are click coords relative to image box [0..1]x[0..1],
    with (0,0)=top-left. We compute vector from center & get angle.
    """
    dx = rx - 0.5
    dy = 0.5 - ry  # invert y so up is positive
    ang = degrees(atan2(dy, dx)) % 360.0  # 0°=right, CCW
    return angle_to_sector(ang)

def list_images_from_answers() -> list[tuple[Path, int]]:
    """
    Return [(path, sector), ...] only for files that exist in the ./img folder.
    """
    base = Path(__file__).parent
    img_dir = base / "img"
    seq = []
    for fname, sec in ANSWERS.items():
        p = img_dir / fname
        if p.exists():
            seq.append((p, int(sec)))
    return seq

# ───────────────────────────────
# Main App
# ───────────────────────────────
class GapTestApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Color Gap Test")
        self.geometry("1000x800")
        self.minsize(640, 540)

        # Data
        self.items = list_images_from_answers()
        if not self.items:
            messagebox.showerror(
                "Setup needed",
                "No images found.\n\n"
                "Create an 'img' folder next to this .py file and put your 14 images there, "
                "then fill ANSWERS = { 'filename.jpg': sector, ... } at the top."
            )
            self.destroy()
            return
        self.idx = 0
        self.responses = []  # dicts: file, expected, sector, correct, method

        # UI
        self._build_ui()

        # Load first
        self._load_current()

        # Resize handling
        self.bind("<Configure>", self._on_resize)

    # ───────────── UI LAYOUT ─────────────
    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=12, pady=10)

        self.title_lbl = ttk.Label(top, text="Click the gap in the ring", font=("Segoe UI", 13, "bold"))
        self.title_lbl.pack(side=tk.LEFT)

        self.progress_lbl = ttk.Label(top, text="")
        self.progress_lbl.pack(side=tk.RIGHT)

        # Canvas area (image container)
        self.canvas = tk.Canvas(self, bg="#111", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)

        # Click handlers
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Button-3>", self._on_click)  # allow right-click too

        # Bottom controls
        bottom = ttk.Frame(self)
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=12, pady=12)

        self.cant_btn = ttk.Button(bottom, text="I can't see the gap", command=self._cant_see)
        self.cant_btn.pack(side=tk.LEFT)

        self.repeat_btn = ttk.Button(bottom, text="Repeat image", command=self._reload_image)
        self.repeat_btn.pack(side=tk.LEFT, padx=(8, 0))

        self.restart_btn = ttk.Button(bottom, text="Restart test", command=self._restart)
        self.restart_btn.pack(side=tk.RIGHT)

        # Hint button
        self.hint_btn = ttk.Button(bottom, text="Show hint", command=self._show_hint_overlay)
        self.hint_btn.pack(side=tk.RIGHT, padx=(0, 8))

        # State for hint overlay
        self._hint_items = []
        self._hint_after_id = None

        # Styles
        s = ttk.Style(self)
        try:
            s.theme_use("clam")
        except tk.TclError:
            pass

    # ───────────── LOAD / SHOW IMAGE ─────────────
    def _load_current(self):
        path, expected = self.items[self.idx]
        self.current_img_path = path
        self.current_expected = expected
        self._load_image(path)
        self._update_progress()

    def _load_image(self, path: Path):
        # Load & store original PIL image
        pil = Image.open(path).convert("RGB")
        self.orig_w, self.orig_h = pil.size
        self.pil_original = pil
        self._render_image_to_canvas()

    def _render_image_to_canvas(self):
        if not hasattr(self, "pil_original"):
            return
        # Fit image inside canvas while preserving aspect
        c_w = max(100, self.canvas.winfo_width())
        c_h = max(100, self.canvas.winfo_height())
        # Also clamp to MAX_DISPLAY_*
        c_w = min(c_w, MAX_DISPLAY_W)
        c_h = min(c_h, MAX_DISPLAY_H)

        img_w, img_h = self.orig_w, self.orig_h
        scale = min(c_w / img_w, c_h / img_h)
        disp_w = max(1, int(img_w * scale))
        disp_h = max(1, int(img_h * scale))

        # Resize
        disp = self.pil_original.resize((disp_w, disp_h), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(disp)

        # Center it
        self.canvas.delete("all")
        self.img_x = (self.canvas.winfo_width() - disp_w) // 2
        self.img_y = (self.canvas.winfo_height() - disp_h) // 2
        self.img_w = disp_w
        self.img_h = disp_h

        self.canvas_img = self.canvas.create_image(
            self.img_x, self.img_y, anchor=tk.NW, image=self.tk_img
        )
        # Clear any lingering hint overlay after re-render
        self._clear_hint_overlay()

    def _on_resize(self, _evt):
        # Re-render current image on resize
        self.after_idle(self._render_image_to_canvas)

    def _reload_image(self):
        # Just rerender; useful if window resized or to “try again”
        self._render_image_to_canvas()

    def _update_progress(self):
        self.progress_lbl.config(text=f"{self.idx+1} / {len(self.items)}")

    # ───────────── CLICK HANDLING ─────────────
    def _on_click(self, event):
        # Compute click relative to the displayed image (0..1)
        if not self._click_on_image(event.x, event.y):
            return  # ignore clicks outside the image rectangle

        rx = (event.x - self.img_x) / self.img_w
        ry = (event.y - self.img_y) / self.img_h

        # Optionally ignore clicks too close to center (not on ring)
        if IGNORE_CENTER:
            dx = rx - 0.5
            dy = ry - 0.5
            dist = (dx*dx + dy*dy) ** 0.5
            if dist < CENTER_MIN_RADIUS:
                messagebox.showinfo("Hint", "Please click on the ring (near the circle).")
                return

        sector = rel_click_to_sector(rx, ry)
        self._record_and_advance(sector, method="click")

    def _click_on_image(self, x, y) -> bool:
        return (self.img_x <= x <= self.img_x + self.img_w) and (self.img_y <= y <= self.img_y + self.img_h)

    def _cant_see(self):
        self._record_and_advance(None, method="cantsee")

    def _record_and_advance(self, sector: int | None, method: str):
        correct = (sector == self.current_expected) if sector is not None else False
        self.responses.append({
            "file": self.current_img_path.name,
            "expected": self.current_expected,
            "sector": sector,
            "correct": bool(correct),
            "method": method
        })

        # Next image or results
        if self.idx + 1 >= len(self.items):
            self._auto_save_results()
            self._show_results()
        else:
            self.idx += 1
            self._clear_hint_overlay()
            self._load_current()

    # ───────────── HINT OVERLAY ─────────────
    def _clear_hint_overlay(self):
        if self._hint_after_id is not None:
            try:
                self.after_cancel(self._hint_after_id)
            except Exception:
                pass
            self._hint_after_id = None
        if self._hint_items:
            for item_id in self._hint_items:
                try:
                    self.canvas.delete(item_id)
                except Exception:
                    pass
            self._hint_items.clear()

    def _show_hint_overlay(self):
        # Remove previous overlay if any
        self._clear_hint_overlay()

        # Need current geometry of displayed image
        if not hasattr(self, "img_x"):
            return

        # Compute center and radii
        cx = self.img_x + self.img_w / 2.0
        cy = self.img_y + self.img_h / 2.0
        radius = min(self.img_w, self.img_h) * 0.48
        inner_radius = radius * 0.70

        # Sector geometry - use same calculation as click detection
        s = int(self.current_expected)
        # Convert sector to math angle (0°=right, CCW positive)
        # Sector 1 is at 12 o'clock = 90° math angle
        math_angle = 90.0 - (s - 1) * 45.0
        start_angle = math_angle + 22.5
        end_angle = math_angle - 22.5

        # Bounding boxes for arcs
        outer_bbox = (
            cx - radius,
            cy - radius,
            cx + radius,
            cy + radius,
        )
        inner_bbox = (
            cx - inner_radius,
            cy - inner_radius,
            cx + inner_radius,
            cy + inner_radius,
        )

        # Draw bounding radial lines
        def polar_to_canvas(r, ang_deg):
            from math import cos, sin, radians
            a = radians(ang_deg)
            return (cx + r * cos(a), cy - r * sin(a))

        line_color = "#ff0"
        arc_color = "#ff0"

        p1 = polar_to_canvas(inner_radius, start_angle)
        p2 = polar_to_canvas(radius, start_angle)
        p3 = polar_to_canvas(inner_radius, end_angle)
        p4 = polar_to_canvas(radius, end_angle)

        self._hint_items.append(self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill=line_color, width=3))
        self._hint_items.append(self.canvas.create_line(p3[0], p3[1], p4[0], p4[1], fill=line_color, width=3))

        # Draw arc along the ring (outer arc)
        extent = -45.0  # clockwise from start_angle
        self._hint_items.append(self.canvas.create_arc(outer_bbox, start=start_angle, extent=extent, style=tk.ARC, outline=arc_color, width=4))
        # Draw arc along the ring (inner arc)
        self._hint_items.append(self.canvas.create_arc(inner_bbox, start=start_angle, extent=extent, style=tk.ARC, outline=arc_color, width=2))

        # Auto-hide after 1.2s
        self._hint_after_id = self.after(1200, self._clear_hint_overlay)

    # ───────────── RESULTS VIEW ─────────────
    def _show_results(self):
        # Clear canvas & controls, show a table
        for w in self.winfo_children():
            w.destroy()

        wrap = ttk.Frame(self)
        wrap.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        correct = sum(1 for r in self.responses if r["correct"])
        total = len(self.responses)

        # Determine color vision assessment
        if correct == 14:
            assessment = "Normal color vision"
            color = "black"
        elif correct >= 11:
            assessment = "Mild color vision deficiency"
            color = "black"
        elif correct >= 6:
            assessment = "Moderate color vision deficiency"
            color = "#8B0000"  # Dark red
        else:
            assessment = "Severe color vision deficiency"
            color = "#8B0000"  # Dark red

        ttk.Label(
            wrap, text=f"Score: {correct} / {total}",
            font=("Segoe UI", 14, "bold")
        ).pack(anchor="w", pady=(0, 4))
        
        assessment_label = ttk.Label(
            wrap, text=f"Assessment: {assessment}",
            font=("Segoe UI", 18, "bold"),
            foreground=color
        )
        assessment_label.pack(anchor="w", pady=(0, 8))
        
        # Show where results were saved
        if hasattr(self, 'saved_file_path'):
            save_info = ttk.Label(
                wrap, 
                text=f"Results automatically saved to: {Path(self.saved_file_path).name}",
                font=("Segoe UI", 10),
                foreground="gray"
            )
            save_info.pack(anchor="w", pady=(0, 8))

        cols = ("#", "Image", "Expected", "Your sector", "Result", "Method")
        tree = ttk.Treeview(wrap, columns=cols, show="headings", height=18)
        for c, w in zip(cols, (40, 320, 100, 110, 100, 110)):
            tree.heading(c, text=c)
            tree.column(c, width=w, stretch=(c == "Image"))
        tree.pack(fill=tk.BOTH, expand=True)

        for i, r in enumerate(self.responses, start=1):
            res = "Correct" if r["correct"] else "Incorrect"
            your = "—" if r["sector"] is None else str(r["sector"])
            meth = "Could not see" if r["method"] == "cantsee" else "Clicked"
            tree.insert("", tk.END, values=(i, r["file"], r["expected"], your, res, meth))

        btns = ttk.Frame(wrap)
        btns.pack(fill=tk.X, pady=10)
        ttk.Button(btns, text="Restart test", command=self._restart).pack(side=tk.RIGHT)

    def _auto_save_results(self):
        """Automatically save test results to a text file with timestamp"""
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"color_vision_test_results_{timestamp}.txt"
        
        # Save to the same folder as the script
        script_dir = Path(__file__).parent
        file_path = script_dir / filename
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                # Write header
                f.write("Color Vision Test Results\n")
                f.write("=" * 50 + "\n")
                f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Write summary
                correct = sum(1 for r in self.responses if r["correct"])
                total = len(self.responses)
                f.write(f"Score: {correct} / {total}\n")
                
                # Write assessment
                if correct == 14:
                    assessment = "Normal color vision"
                elif correct >= 11:
                    assessment = "Mild color vision deficiency"
                elif correct >= 6:
                    assessment = "Moderate color vision deficiency"
                else:
                    assessment = "Severe color vision deficiency"
                
                f.write(f"Assessment: {assessment}\n\n")
                
                # Write detailed results
                f.write("Detailed Results:\n")
                f.write("-" * 50 + "\n")
                f.write(f"{'#':<3} {'Image':<35} {'Expected':<8} {'Your':<6} {'Result':<10} {'Method':<12}\n")
                f.write("-" * 50 + "\n")
                
                for i, r in enumerate(self.responses, start=1):
                    res = "Correct" if r["correct"] else "Incorrect"
                    your = "—" if r["sector"] is None else str(r["sector"])
                    meth = "Could not see" if r["method"] == "cantsee" else "Clicked"
                    f.write(f"{i:<3} {r['file']:<35} {r['expected']:<8} {your:<6} {res:<10} {meth:<12}\n")
            
            # Store the saved file path for display
            self.saved_file_path = str(file_path)
            
        except Exception as e:
            print(f"Failed to auto-save results: {str(e)}")

    def _save_results(self):
        """Save test results to a text file with timestamp"""
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"color_vision_test_results_{timestamp}.txt"
        
        # Save to the same folder as the script
        script_dir = Path(__file__).parent
        file_path = script_dir / filename
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                # Write header
                f.write("Color Vision Test Results\n")
                f.write("=" * 50 + "\n")
                f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Write summary
                correct = sum(1 for r in self.responses if r["correct"])
                total = len(self.responses)
                f.write(f"Score: {correct} / {total}\n")
                
                # Write assessment
                if correct == 14:
                    assessment = "Normal color vision"
                elif correct >= 11:
                    assessment = "Mild color vision deficiency"
                elif correct >= 6:
                    assessment = "Moderate color vision deficiency"
                else:
                    assessment = "Severe color vision deficiency"
                
                f.write(f"Assessment: {assessment}\n\n")
                
                # Write detailed results
                f.write("Detailed Results:\n")
                f.write("-" * 50 + "\n")
                f.write(f"{'#':<3} {'Image':<35} {'Expected':<8} {'Your':<6} {'Result':<10} {'Method':<12}\n")
                f.write("-" * 50 + "\n")
                
                for i, r in enumerate(self.responses, start=1):
                    res = "Correct" if r["correct"] else "Incorrect"
                    your = "—" if r["sector"] is None else str(r["sector"])
                    meth = "Could not see" if r["method"] == "cantsee" else "Clicked"
                    f.write(f"{i:<3} {r['file']:<35} {r['expected']:<8} {your:<6} {res:<10} {meth:<12}\n")
            
            messagebox.showinfo("Success", f"Results saved to:\n{file_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results:\n{str(e)}")

    def _restart(self):
        self.idx = 0
        self.responses.clear()
        # Rebuild UI
        for w in self.winfo_children():
            w.destroy()
        self._build_ui()
        self._load_current()


if __name__ == "__main__":
    app = GapTestApp()
    if app.winfo_exists():
        app.mainloop()
