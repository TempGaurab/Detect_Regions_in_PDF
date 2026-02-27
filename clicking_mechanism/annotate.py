import os
import json
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(SCRIPT_DIR, '..', 'data')
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, '..', 'clicking_mechanism', 'annotations', 'model_output')

IMAGE_NAME  = "form2.png"   # ← change as needed
IMAGE_PATH  = os.path.join(DATA_FOLDER, IMAGE_NAME)

# ─────────────────────────────────────────────
# IMAGE LOADING & PREPROCESSING
# ─────────────────────────────────────────────

def load_cv_image(path):
    img_pil = Image.open(path).convert("L")
    return np.array(img_pil)

def preprocess(gray):
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def rect_contains(outer, inner, tol=3):
    """Returns True if inner rect is fully inside outer rect (with tolerance)."""
    ox, oy, ow, oh = outer
    ix, iy, iw, ih = inner
    return (ix >= ox - tol and iy >= oy - tol and
            ix + iw <= ox + ow + tol and iy + ih <= oy + oh + tol)

def rect_overlaps(a, b, tol=3):
    """Returns True if two rects overlap."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw < bx - tol or bx + bw < ax - tol or
                ay + ah < by - tol or by + bh < ay - tol)

def rect_area(r):
    return r[2] * r[3]

# ─────────────────────────────────────────────
# BOX DETECTION (all rectangular contours)
# ─────────────────────────────────────────────

def detect_all_boxes(binary, img_w, img_h):
    """
    Detect all 4-sided rectangular contours.
    Returns list of (x, y, w, h).
    """
    BOX_MIN   = 8          # minimum side length in px
    BOX_MAX_W = img_w * 0.90   # ignore full-page-width frames
    BOX_MAX_H = img_h * 0.90

    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    raw = []
    for i, cnt in enumerate(contours):
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) not in (4, 5):   # allow slight rounding
            continue
        x, y, w, h = cv2.boundingRect(approx)
        if w < BOX_MIN or h < BOX_MIN:
            continue
        if w > BOX_MAX_W or h > BOX_MAX_H:
            continue
        raw.append((x, y, w, h))

    # Deduplicate near-identical boxes (same position within 5 px)
    raw.sort(key=lambda b: (b[0], b[1]))
    kept = []
    for box in raw:
        x, y, w, h = box
        dup = False
        for kb in kept:
            if abs(x - kb[0]) <= 5 and abs(y - kb[1]) <= 5 and \
               abs(w - kb[2]) <= 5 and abs(h - kb[3]) <= 5:
                dup = True
                break
        if not dup:
            kept.append(box)

    return kept

# ─────────────────────────────────────────────
# PDF-FILLER BOX RULES
# ─────────────────────────────────────────────

# Rule constants
PARAGRAPH_MIN_AREA   = 0    # set dynamically below
CHECKBOX_MAX_SIDE    = 50   # boxes ≤ this on both sides → checkbox candidate
DATE_BOX_ASPECT_MAX  = 6.0  # width/height — wide thin boxes = date/text inputs
MAX_FILLABLE_H       = 80   # boxes taller than this are likely paragraph areas
                             # (user can't "click into" them like a PDF field)

def classify_boxes(all_boxes, img_w, img_h):
    """
    Apply PDF-filler logic:
      1. Ignore giant paragraph/section boxes.
      2. For nested boxes, keep only the INNERMOST ones.
      3. If a box contains a horizontal line, prefer the box (ignore the line).
      4. Classify survivors as 'checkbox' or 'input_box'.
    Returns (checkboxes, input_boxes) — both as lists of (x, y, w, h).
    """
    # --- Step 1: Remove obviously non-fillable large boxes ---
    # A box is too big if it's taller than MAX_FILLABLE_H AND wider than 40 % of page
    # (paragraph / section containers that a user would never "type into")
    fillable = []
    for box in all_boxes:
        x, y, w, h = box
        if h > MAX_FILLABLE_H and w > img_w * 0.40:
            continue   # big paragraph container — skip
        fillable.append(box)

    # --- Step 2: Keep only innermost boxes (remove any box that contains another) ---
    # Sort smallest-first so we can check containment efficiently
    fillable.sort(key=rect_area)
    innermost = []
    for i, box in enumerate(fillable):
        is_parent = False
        for other in fillable:
            if other is box:
                continue
            if rect_contains(box, other, tol=4) and rect_area(other) < rect_area(box) * 0.85:
                # box contains a smaller box → box is a container, not a field
                is_parent = True
                break
        if not is_parent:
            innermost.append(box)

    # --- Step 3: Classify as checkbox vs input box ---
    checkboxes  = []
    input_boxes = []

    for box in innermost:
        x, y, w, h = box
        aspect = w / h if h > 0 else 999

        # Checkbox: roughly square, small
        if (w <= CHECKBOX_MAX_SIDE and h <= CHECKBOX_MAX_SIDE and
                abs(1 - aspect) <= 0.35):
            checkboxes.append(box)
        else:
            # Input box (date cell, text field, etc.)
            input_boxes.append(box)

    return checkboxes, input_boxes

# ─────────────────────────────────────────────
# LINE DETECTION
# ─────────────────────────────────────────────

def check_above_is_free(binary, x, y, w, search_height=None, max_ink_ratio=0.03):
    if search_height is None:
        search_height = min(60, max(20, w // 8))
    ax1, ax2 = x, x + w
    ay2 = max(0, y - 2)
    ay1 = max(0, y - 2 - search_height)
    if ay2 <= ay1:
        return True
    region = binary[ay1:ay2, ax1:ax2]
    if region.size == 0:
        return True
    return np.count_nonzero(region) / region.size <= max_ink_ratio


def line_inside_any_box(lx, ly, lw, lh, input_boxes, tol=6):
    """Return True if this line lives inside one of the input boxes."""
    for (bx, by, bw, bh) in input_boxes:
        if (lx >= bx - tol and lx + lw <= bx + bw + tol and
                ly >= by - tol and ly + lh <= by + bh + tol):
            return True
    return False


def detect_input_lines(binary, img_w, img_h, checkboxes, input_boxes):
    """
    Detect bare underline-style input fields (_______).
    Skips lines that are:
      - inside a detected box (box wins)
      - near a checkbox
      - spanning the full page (table borders)
      - not preceded by whitespace above (table rules / text underlines)
      - missing a label to the left (and not long enough to be a signature line)
    """
    LINE_MIN_W   = 60
    LINE_MAX_H   = 6
    LABEL_SEARCH = 400
    LABEL_RATIO  = 0.008
    VICINITY     = 15
    NMS_Y        = 8

    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(h_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    raw = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < LINE_MIN_W or h > LINE_MAX_H:
            continue
        if w > img_w * 0.80:
            continue

        # Skip if near a checkbox
        near_cb = False
        for (cx, cy, cw, ch) in checkboxes:
            if (x >= cx - VICINITY and x + w <= cx + cw + VICINITY and
                    y >= cy - VICINITY and y + h <= cy + ch + VICINITY):
                near_cb = True
                break
        if near_cb:
            continue

        # Skip if inside a detected input box (box takes priority)
        if line_inside_any_box(x, y, w, h, input_boxes):
            continue

        # Above must be whitespace (real input line, not a table border)
        if not check_above_is_free(binary, x, y, w):
            continue

        # Label check (text to the left)
        lx1 = max(0, x - LABEL_SEARCH)
        lx2 = max(0, x - 5)
        ly1 = max(0, y - 12)
        ly2 = min(binary.shape[0], y + h + 12)
        has_label = False
        if lx2 > lx1:
            region = binary[ly1:ly2, lx1:lx2]
            if region.size > 0 and np.count_nonzero(region) / region.size > LABEL_RATIO:
                has_label = True

        long_line = w > img_w * 0.15
        if not has_label and not long_line:
            continue

        raw.append((x, y, w, h))

    # Deduplicate
    raw.sort(key=lambda b: (b[1], -b[2]))
    kept = []
    for box in raw:
        x, y, w, h = box
        dup = False
        for kx, ky, kw, kh in kept:
            if abs(y - ky) <= NMS_Y and abs(x - kx) <= LINE_MIN_W:
                dup = True
                break
        if not dup:
            kept.append(box)

    return kept

# ─────────────────────────────────────────────
# COORDINATE CONVERSION
# ─────────────────────────────────────────────

def checkboxes_to_coords(checkboxes):
    return [
        [round(x + w / 2, 2), round(y + h / 2, 2)]
        for (x, y, w, h) in checkboxes
    ]

def input_boxes_to_coords(input_boxes):
    """
    Returns [x1, y1, x2, y2] for each input box
    so the clicker knows the full clickable region.
    """
    return [
        [round(float(x), 2), round(float(y), 2),
         round(float(x + w), 2), round(float(y + h), 2)]
        for (x, y, w, h) in input_boxes
    ]

def lines_to_coords(input_lines):
    return [
        [round(float(x),     2),
         round(y + h / 2,    2),
         round(float(x + w), 2),
         round(y + h / 2,    2)]
        for (x, y, w, h) in input_lines
    ]

# ─────────────────────────────────────────────
# SAVE JSON
# ─────────────────────────────────────────────

def save_coordinates(image_name, checkboxes, input_boxes, input_lines, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    base     = os.path.splitext(image_name)[0]
    out_path = os.path.join(output_dir, base + ".json")

    payload = {
        "checkboxes":   checkboxes_to_coords(checkboxes),
        "input_boxes":  input_boxes_to_coords(input_boxes),
        "lines":        lines_to_coords(input_lines),
        "meta": {
            "source":           "opencv",
            "image":            image_name,
            "checkbox_count":   len(checkboxes),
            "input_box_count":  len(input_boxes),
            "line_count":       len(input_lines),
        }
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved → {out_path}")
    print(f"  checkboxes  : {len(payload['checkboxes'])}")
    print(f"  input_boxes : {len(payload['input_boxes'])}")
    print(f"  lines       : {len(payload['lines'])}")
    return out_path

# ─────────────────────────────────────────────
# VISUALISE
# ─────────────────────────────────────────────

def draw_results(gray, checkboxes, input_boxes, input_lines):
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # Checkboxes — blue
    for (x, y, w, h) in checkboxes:
        cv2.rectangle(vis, (x, y), (x+w, y+h), (255, 80, 0), 2)
    # Input boxes — green
    for (x, y, w, h) in input_boxes:
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 200, 80), 2)
    # Underline lines — orange
    for (x, y, w, h) in input_lines:
        cv2.rectangle(vis, (x, y-3), (x+w, y+h+3), (0, 165, 255), 2)
    return vis

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    gray         = load_cv_image(IMAGE_PATH)
    h_img, w_img = gray.shape
    binary       = preprocess(gray)

    # 1. Detect every rectangle on the page
    all_boxes = detect_all_boxes(binary, w_img, h_img)

    # 2. Apply PDF-filler rules → innermost fillable boxes only
    checkboxes, input_boxes = classify_boxes(all_boxes, w_img, h_img)

    # 3. Detect bare underline fields (box takes priority over line)
    input_lines = detect_input_lines(binary, w_img, h_img, checkboxes, input_boxes)

    print(f"=== Detection Results for {IMAGE_NAME} ===")
    print(f"  All boxes detected : {len(all_boxes)}")
    print(f"  Checkboxes         : {len(checkboxes)}")
    print(f"  Input boxes        : {len(input_boxes)}")
    print(f"  Underline lines    : {len(input_lines)}")

    save_coordinates(IMAGE_NAME, checkboxes, input_boxes, input_lines, OUTPUT_DIR)

    vis = draw_results(gray, checkboxes, input_boxes, input_lines)

    legend = [
        mpatches.Patch(color="dodgerblue", label=f"Checkboxes ({len(checkboxes)})"),
        mpatches.Patch(color="limegreen",  label=f"Input boxes ({len(input_boxes)})"),
        mpatches.Patch(color="orange",     label=f"Underline lines ({len(input_lines)})"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 14))
    ax1.imshow(gray, cmap="gray")
    ax1.set_title("Original Form", fontsize=14)
    ax1.axis("off")

    ax2.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    ax2.set_title("Detected Fillable Elements", fontsize=14)
    ax2.legend(handles=legend, loc="upper right", fontsize=11)
    ax2.axis("off")

    plt.suptitle(f"OpenCV Detection — {IMAGE_NAME}", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()

    annotated_path = os.path.join(OUTPUT_DIR, os.path.splitext(IMAGE_NAME)[0] + "_annotated.png")
    cv2.imwrite(annotated_path, vis)
    print(f"Annotated image saved → {annotated_path}")


if __name__ == "__main__":
    main()