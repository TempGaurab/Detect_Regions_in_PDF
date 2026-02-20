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
# CHECKBOX DETECTION  (deduplication fixed)
# ─────────────────────────────────────────────

def detect_checkboxes(binary, img_w, img_h):
    CHECKBOX_MIN = 10
    CHECKBOX_MAX = 50
    SQUARE_TOL   = 0.30
    NMS_DIST     = 5        # suppress duplicates closer than this many px

    contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    raw = []
    for cnt in contours:
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        x, y, w, h = cv2.boundingRect(approx)
        if w > img_w * 0.95 or h > img_h * 0.95:
            continue
        if (CHECKBOX_MIN <= w <= CHECKBOX_MAX and
                CHECKBOX_MIN <= h <= CHECKBOX_MAX and
                abs(1 - w / h) <= SQUARE_TOL):
            raw.append((x, y, w, h))

    # Deduplicate: remove boxes whose top-left is within NMS_DIST px of a kept box
    raw.sort(key=lambda b: (b[0], b[1]))
    kept = []
    for box in raw:
        x, y, w, h = box
        duplicate = False
        for kx, ky, kw, kh in kept:
            if abs(x - kx) <= NMS_DIST and abs(y - ky) <= NMS_DIST:
                duplicate = True
                break
        if not duplicate:
            kept.append(box)

    return kept

# ─────────────────────────────────────────────
# LINE DETECTION  (improved)
# ─────────────────────────────────────────────

def check_above_is_free(binary, x, y, w, search_height=None, max_ink_ratio=0.03):
    """
    Returns True if the region directly above the line is mostly whitespace.
    This confirms the line is an input field (user writes above it), not a
    table border or decorative rule (which would have content on both sides).

    x, y, w   — bounding rect of the detected line
    search_height — how many px above the line to inspect (default: 1× line width capped at 60px)
    max_ink_ratio — maximum allowed ink density above the line (default 3 %)
    """
    if search_height is None:
        search_height = min(60, max(20, w // 8))

    ax1 = x
    ax2 = x + w
    ay2 = max(0, y - 2)                          # just above the line
    ay1 = max(0, y - 2 - search_height)

    if ay2 <= ay1:
        return True   # too close to top of image → assume free

    region = binary[ay1:ay2, ax1:ax2]
    if region.size == 0:
        return True

    ink_ratio = np.count_nonzero(region) / region.size
    return ink_ratio <= max_ink_ratio


def detect_input_lines(binary, img_w, img_h, checkboxes):
    LINE_MIN_W   = 60       # minimum line width in px
    LINE_MAX_H   = 6        # maximum line thickness
    LABEL_SEARCH = 400      # how far left to look for a label
    LABEL_RATIO  = 0.008    # min ink density in label area
    VICINITY     = 15       # px proximity to checkbox → skip
    NMS_Y        = 8        # suppress duplicate lines within this many px vertically

    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(h_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    raw = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < LINE_MIN_W or h > LINE_MAX_H:
            continue

        # Skip page-spanning lines (table borders / ruling lines)
        if w > img_w * 0.80:
            continue

        # Skip if inside / adjacent to a checkbox
        near_checkbox = False
        for (cx, cy, cw, ch) in checkboxes:
            if (x >= cx - VICINITY and x + w <= cx + cw + VICINITY and
                    y >= cy - VICINITY and y + h <= cy + ch + VICINITY):
                near_checkbox = True
                break
        if near_checkbox:
            continue

        # ── NEW: above-line whitespace check ──────────────────────────────
        # Real input lines have free space above them (where the user writes).
        # Table borders / text underlines / decorative rules have ink above.
        if not check_above_is_free(binary, x, y, w):
            continue
        # ──────────────────────────────────────────────────────────────────

        # Label check — look to the left for ink (inline label e.g. "Name: ___")
        lx1 = max(0, x - LABEL_SEARCH)
        lx2 = max(0, x - 5)
        ly1 = max(0, y - 12)
        ly2 = min(binary.shape[0], y + h + 12)
        has_label = False
        if lx2 > lx1:
            region = binary[ly1:ly2, lx1:lx2]
            if region.size > 0 and np.count_nonzero(region) / region.size > LABEL_RATIO:
                has_label = True

        # Also accept lines that are fairly long (signature / notary lines).
        # These may have their label centred below rather than to the left.
        long_line = w > img_w * 0.15

        if not has_label and not long_line:
            continue

        raw.append((x, y, w, h))

    # Deduplicate: keep widest line when multiple share same y
    raw.sort(key=lambda b: (b[1], -b[2]))   # sort by y, then widest first
    kept = []
    for box in raw:
        x, y, w, h = box
        duplicate = False
        for kx, ky, kw, kh in kept:
            if abs(y - ky) <= NMS_Y and abs(x - kx) <= LINE_MIN_W:
                duplicate = True
                break
        if not duplicate:
            kept.append(box)

    return kept

# ─────────────────────────────────────────────
# COORDINATE CONVERSION  (matches annotation tool format)
# checkboxes: [[cx, cy], ...]
# lines:      [[x1, y1, x2, y2], ...]
# ─────────────────────────────────────────────

def checkboxes_to_coords(checkboxes):
    return [
        [round(x + w / 2, 2), round(y + h / 2, 2)]
        for (x, y, w, h) in checkboxes
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

def save_coordinates(image_name, checkboxes, input_lines, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    base     = os.path.splitext(image_name)[0]
    out_path = os.path.join(output_dir, base + ".json")

    payload = {
        "checkboxes": checkboxes_to_coords(checkboxes),
        "lines":      lines_to_coords(input_lines),
        "meta": {
            "source":         "opencv",
            "image":          image_name,
            "checkbox_count": len(checkboxes),
            "line_count":     len(input_lines),
        }
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved → {out_path}")
    print(f"  checkboxes : {len(payload['checkboxes'])}")
    print(f"  lines      : {len(payload['lines'])}")
    return out_path

# ─────────────────────────────────────────────
# VISUALISE
# ─────────────────────────────────────────────

def draw_results(gray, checkboxes, input_lines):
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h) in checkboxes:
        cv2.rectangle(vis, (x, y), (x+w, y+h), (255, 80, 0), 2)
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

    checkboxes  = detect_checkboxes(binary, w_img, h_img)
    input_lines = detect_input_lines(binary, w_img, h_img, checkboxes)

    print(f"=== Detection Results for {IMAGE_NAME} ===")
    print(f"  Checkboxes  : {len(checkboxes)}")
    print(f"  Input lines : {len(input_lines)}")

    save_coordinates(IMAGE_NAME, checkboxes, input_lines, OUTPUT_DIR)

    vis = draw_results(gray, checkboxes, input_lines)

    legend = [
        mpatches.Patch(color="dodgerblue", label=f"Checkboxes ({len(checkboxes)})"),
        mpatches.Patch(color="orange",     label=f"Input lines ({len(input_lines)})"),
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