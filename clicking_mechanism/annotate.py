"""
Region Annotation Tool
======================
Annotate images by clicking to mark checkboxes, lines, and form boxes.

CONTROLS:
  Left Click       → Add point / draw annotation
  C                → Switch to CHECKBOX mode
  L                → Switch to LINE mode
  B                → Switch to BOX mode (click two corners)
  Z                → Undo last point
  D                → Delete a specific annotation (click near it after pressing D)
  S / Enter        → Save & move to next image
  Q / Escape       → Quit (saves current progress first)

LINE MODE:
  First click sets start point, second click sets end point → line is saved.

BOX MODE:
  First click sets one corner, second click sets opposite corner → box is saved.
"""

#completed till 329.
import os
import sys
import json
import math
import glob
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), 'annotations')
IMG_EXTS    = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif', '*.gif', '*.webp')

CHECKBOX_COLOR  = '#00FF00'   # green
LINE_COLOR      = '#FF4444'   # red
BOX_COLOR       = '#44AAFF'   # blue
PENDING_COLOR   = '#FFFF00'   # yellow
DELETE_COLOR    = '#FF8800'   # orange
POINT_RADIUS    = 6
# ────────────────────────────────────────────────────────────────────────────


def gather_images(data_dir):
    images = []
    for ext in IMG_EXTS:
        images.extend(glob.glob(os.path.join(data_dir, ext)))
        images.extend(glob.glob(os.path.join(data_dir, ext.upper())))
    return sorted(set(images))


def load_existing(json_path):
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
            data.setdefault('checkboxes', [])
            data.setdefault('lines', [])
            data.setdefault('boxes', [])
            return data
    return {'checkboxes': [], 'lines': [], 'boxes': []}


def save_annotations(json_path, data):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)


class AnnotationApp:
    def __init__(self, root, images):
        self.root   = root
        self.images = images
        self.idx    = 0

        self.mode        = 'checkbox'
        self.delete_mode = False
        self.first_point = None

        self.data         = {}
        self.canvas_items = []

        self.root.title('Region Annotation Tool')
        self.root.configure(bg='#1e1e1e')
        self._build_ui()
        self._bind_keys()
        self._load_image()

    # ── UI ───────────────────────────────────────────────────────────────────
    def _build_ui(self):
        top = tk.Frame(self.root, bg='#1e1e1e')
        top.pack(fill=tk.X, padx=6, pady=4)

        self.lbl_file = tk.Label(top, text='', bg='#1e1e1e', fg='white',
                                 font=('Courier', 11, 'bold'))
        self.lbl_file.pack(side=tk.LEFT)

        self.lbl_mode = tk.Label(top, text='', bg='#1e1e1e', fg=CHECKBOX_COLOR,
                                 font=('Courier', 12, 'bold'), width=36)
        self.lbl_mode.pack(side=tk.RIGHT)

        self.canvas = tk.Canvas(self.root, cursor='crosshair', bg='#111')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        hint = ("[C] Checkbox  [L] Line  [B] Box  "
                "[Z] Undo  [D] Delete  [S/Enter] Save & Next  [Q/Esc] Quit")
        tk.Label(self.root, text=hint, bg='#2a2a2a', fg='#aaa',
                 font=('Courier', 9)).pack(fill=tk.X)

    def _bind_keys(self):
        for key in ('<c>', '<C>'):
            self.root.bind(key, lambda e: self._set_mode('checkbox'))
        for key in ('<l>', '<L>'):
            self.root.bind(key, lambda e: self._set_mode('line'))
        for key in ('<b>', '<B>'):
            self.root.bind(key, lambda e: self._set_mode('box'))
        for key in ('<z>', '<Z>'):
            self.root.bind(key, lambda e: self._undo())
        for key in ('<d>', '<D>'):
            self.root.bind(key, lambda e: self._toggle_delete_mode())
        for key in ('<s>', '<S>', '<Return>'):
            self.root.bind(key, lambda e: self._save_and_next())
        for key in ('<q>', '<Q>', '<Escape>'):
            self.root.bind(key, lambda e: self._quit())
        self.canvas.bind('<Button-1>', self._on_click)

    # ── Image loading ────────────────────────────────────────────────────────
    def _load_image(self):
        if self.idx >= len(self.images):
            messagebox.showinfo('Done', 'All images annotated!')
            self.root.quit()
            return

        img_path  = self.images[self.idx]
        base      = os.path.splitext(os.path.basename(img_path))[0]
        json_path = os.path.join(OUTPUT_DIR, base + '.json')

        self.current_img_path  = img_path
        self.current_json_path = json_path
        self.data              = load_existing(json_path)
        self.first_point       = None
        self.delete_mode       = False

        pil_img = Image.open(img_path)
        self.orig_w, self.orig_h = pil_img.size

        screen_w = self.root.winfo_screenwidth()  - 40
        screen_h = self.root.winfo_screenheight() - 160
        scale    = min(screen_w / self.orig_w, screen_h / self.orig_h, 1.0)
        self.scale = scale

        disp_w = int(self.orig_w * scale)
        disp_h = int(self.orig_h * scale)
        pil_img = pil_img.resize((disp_w, disp_h), Image.LANCZOS)

        self.tk_img = ImageTk.PhotoImage(pil_img)
        self.canvas.config(width=disp_w, height=disp_h)
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

        self.lbl_file.config(
            text=f'[{self.idx+1}/{len(self.images)}]  {os.path.basename(img_path)}'
        )
        self._refresh_mode_label()
        self._redraw_all()

    # ── Mode helpers ─────────────────────────────────────────────────────────
    def _set_mode(self, mode):
        self.mode        = mode
        self.delete_mode = False
        self.first_point = None
        self._refresh_mode_label()

    def _toggle_delete_mode(self):
        self.delete_mode = not self.delete_mode
        self.first_point = None
        self._refresh_mode_label()

    def _refresh_mode_label(self):
        if self.delete_mode:
            self.lbl_mode.config(text='MODE: DELETE (click near item)', fg=DELETE_COLOR)
        elif self.mode == 'checkbox':
            self.lbl_mode.config(text='MODE: CHECKBOX', fg=CHECKBOX_COLOR)
        elif self.mode == 'line':
            if self.first_point:
                self.lbl_mode.config(text='MODE: LINE  (click end point)', fg=PENDING_COLOR)
            else:
                self.lbl_mode.config(text='MODE: LINE  (click start point)', fg=LINE_COLOR)
        elif self.mode == 'box':
            if self.first_point:
                self.lbl_mode.config(text='MODE: BOX  (click opposite corner)', fg=PENDING_COLOR)
            else:
                self.lbl_mode.config(text='MODE: BOX  (click first corner)', fg=BOX_COLOR)

    # ── Canvas helpers ───────────────────────────────────────────────────────
    def _to_orig(self, cx, cy):
        return cx / self.scale, cy / self.scale

    def _to_canvas(self, ox, oy):
        return ox * self.scale, oy * self.scale

    def _draw_checkbox(self, ox, oy):
        cx, cy = self._to_canvas(ox, oy)
        r = POINT_RADIUS
        self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r,
                                fill=CHECKBOX_COLOR, outline='white', width=1,
                                tags='annotation')

    def _draw_line(self, ox1, oy1, ox2, oy2):
        cx1, cy1 = self._to_canvas(ox1, oy1)
        cx2, cy2 = self._to_canvas(ox2, oy2)
        r = POINT_RADIUS // 2
        self.canvas.create_line(cx1, cy1, cx2, cy2,
                                fill=LINE_COLOR, width=2, tags='annotation')
        for cx, cy in [(cx1, cy1), (cx2, cy2)]:
            self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r,
                                    fill=LINE_COLOR, outline='white', tags='annotation')

    def _draw_box(self, ox1, oy1, ox2, oy2):
        cx1, cy1 = self._to_canvas(ox1, oy1)
        cx2, cy2 = self._to_canvas(ox2, oy2)
        self.canvas.create_rectangle(cx1, cy1, cx2, cy2,
                                     outline=BOX_COLOR, width=2,
                                     fill='', tags='annotation')
        r = POINT_RADIUS // 2
        for cx, cy in [(cx1, cy1), (cx2, cy2)]:
            self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r,
                                    fill=BOX_COLOR, outline='white', tags='annotation')

    def _draw_pending(self, ox, oy):
        cx, cy = self._to_canvas(ox, oy)
        r = POINT_RADIUS
        self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r,
                                fill=PENDING_COLOR, outline='white', width=1,
                                tags='annotation')

    def _redraw_all(self):
        self.canvas.delete('annotation')
        for cb in self.data['checkboxes']:
            self._draw_checkbox(*cb)
        for ln in self.data['lines']:
            self._draw_line(*ln)
        for bx in self.data['boxes']:
            self._draw_box(*bx)
        if self.first_point:
            self._draw_pending(*self.first_point)

    # ── Click handler ────────────────────────────────────────────────────────
    def _on_click(self, event):
        ox, oy = self._to_orig(event.x, event.y)

        if self.delete_mode:
            self._delete_nearest(ox, oy)
            return

        if self.mode == 'checkbox':
            self.data['checkboxes'].append([round(ox, 2), round(oy, 2)])
            self._redraw_all()

        elif self.mode in ('line', 'box'):
            if self.first_point is None:
                self.first_point = (round(ox, 2), round(oy, 2))
                self._refresh_mode_label()
                self._redraw_all()
            else:
                x1, y1 = self.first_point
                x2, y2 = round(ox, 2), round(oy, 2)
                key = 'lines' if self.mode == 'line' else 'boxes'
                self.data[key].append([x1, y1, x2, y2])
                self.first_point = None
                self._refresh_mode_label()
                self._redraw_all()

    # ── Undo ─────────────────────────────────────────────────────────────────
    def _undo(self):
        if self.first_point:
            self.first_point = None
            self._refresh_mode_label()
            self._redraw_all()
            return
        if self.mode == 'line' and self.data['lines']:
            self.data['lines'].pop()
        elif self.mode == 'checkbox' and self.data['checkboxes']:
            self.data['checkboxes'].pop()
        elif self.mode == 'box' and self.data['boxes']:
            self.data['boxes'].pop()
        self._redraw_all()

    # ── Delete nearest ───────────────────────────────────────────────────────
    def _delete_nearest(self, ox, oy):
        best_dist  = float('inf')
        best_type  = None
        best_index = None

        for i, cb in enumerate(self.data['checkboxes']):
            d = math.hypot(ox - cb[0], oy - cb[1])
            if d < best_dist:
                best_dist, best_type, best_index = d, 'checkboxes', i

        for i, ln in enumerate(self.data['lines']):
            for px, py in [(ln[0], ln[1]), (ln[2], ln[3])]:
                d = math.hypot(ox - px, oy - py)
                if d < best_dist:
                    best_dist, best_type, best_index = d, 'lines', i

        for i, bx in enumerate(self.data['boxes']):
            for px, py in [(bx[0], bx[1]), (bx[2], bx[3])]:
                d = math.hypot(ox - px, oy - py)
                if d < best_dist:
                    best_dist, best_type, best_index = d, 'boxes', i

        threshold = 20 / self.scale
        if best_dist < threshold and best_index is not None:
            self.data[best_type].pop(best_index)
            self._redraw_all()
        else:
            messagebox.showwarning('Delete', 'No annotation found close enough to click.')

        self.delete_mode = False
        self._refresh_mode_label()

    # ── Save / navigation ────────────────────────────────────────────────────
    def _save_current(self):
        save_annotations(self.current_json_path, self.data)
        print(f'Saved → {self.current_json_path}  '
              f'({len(self.data["checkboxes"])} checkboxes, '
              f'{len(self.data["lines"])} lines, '
              f'{len(self.data["boxes"])} boxes)')

    def _save_and_next(self):
        self._save_current()
        self.idx += 1
        self._load_image()

    def _quit(self):
        if messagebox.askyesno('Quit', 'Save current image and quit?'):
            self._save_current()
            self.root.quit()


# ── Entry point ──────────────────────────────────────────────────────────────
def main():
    images = gather_images(DATA_DIR)
    if not images:
        print(f'No images found in: {os.path.abspath(DATA_DIR)}')
        sys.exit(1)

    print(f'Found {len(images)} image(s) in {os.path.abspath(DATA_DIR)}')
    print(f'Annotations will be saved to: {os.path.abspath(OUTPUT_DIR)}')

    root = tk.Tk()
    app  = AnnotationApp(root, images)
    root.mainloop()


if __name__ == '__main__':
    main()