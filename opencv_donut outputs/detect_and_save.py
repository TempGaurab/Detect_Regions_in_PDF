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
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FOLDER       = os.path.join(SCRIPT_DIR, '..', 'data')
ANNOTATION_FOLDER = os.path.join(SCRIPT_DIR, '..', 'clicking_mechanism', 'annotations')

OUTPUT_BASE       = os.path.join(SCRIPT_DIR, '..', 'annotated_output')
OUTPUT_JSON_DIR   = os.path.join(OUTPUT_BASE, 'json')
OUTPUT_IMAGE_DIR  = os.path.join(OUTPUT_BASE, 'images')

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
# CHECKBOX DETECTION
# ─────────────────────────────────────────────

def detect_checkboxes(binary, img_w, img_h):
    CHECKBOX_MIN = 10
    CHECKBOX_MAX = 50
    SQUARE_TOL   = 0.30
    NMS_DIST     = 5

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
# LINE DETECTION
# ─────────────────────────────────────────────

def check_above_is_free(binary, x, y, w, search_height=None, max_ink_ratio=0.03):
    if search_height is None:
        search_height = min(60, max(20, w // 8))

    ax1 = x
    ax2 = x + w
    ay2 = max(0, y - 2)
    ay1 = max(0, y - 2 - search_height)

    if ay2 <= ay1:
        return True

    region = binary[ay1:ay2, ax1:ax2]
    if region.size == 0:
        return True

    ink_ratio = np.count_nonzero(region) / region.size
    return ink_ratio <= max_ink_ratio


def detect_input_lines(binary, img_w, img_h, checkboxes):

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

        near_checkbox = False
        for (cx, cy, cw, ch) in checkboxes:
            if (x >= cx - VICINITY and x + w <= cx + cw + VICINITY and
                y >= cy - VICINITY and y + h <= cy + ch + VICINITY):
                near_checkbox = True
                break

        if near_checkbox:
            continue

        if not check_above_is_free(binary, x, y, w):
            continue

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

    raw.sort(key=lambda b: (b[1], -b[2]))

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
# COORDINATE CONVERSION
# ─────────────────────────────────────────────

def checkboxes_to_coords(checkboxes):
    return [
        [round(x + w / 2, 2), round(y + h / 2, 2)]
        for (x, y, w, h) in checkboxes
    ]

def lines_to_coords(input_lines):
    return [
        [
            round(float(x), 2),
            round(y + h / 2, 2),
            round(float(x + w), 2),
            round(y + h / 2, 2)
        ]
        for (x, y, w, h) in input_lines
    ]

# ─────────────────────────────────────────────
# SAVE JSON
# ─────────────────────────────────────────────

def save_coordinates(image_name, checkboxes, input_lines):
    os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)

    base = os.path.splitext(image_name)[0]
    out_path = os.path.join(OUTPUT_JSON_DIR, base + ".json")

    payload = {
        "checkboxes": checkboxes_to_coords(checkboxes),
        "lines": lines_to_coords(input_lines),
        "meta": {
            "source": "opencv",
            "image": image_name,
            "checkbox_count": len(checkboxes),
            "line_count": len(input_lines),
        }
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"  JSON saved → {out_path}")

# ─────────────────────────────────────────────
# DRAW RESULTS
# ─────────────────────────────────────────────

def draw_results(gray, checkboxes, input_lines):
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for (x, y, w, h) in checkboxes:
        cv2.rectangle(vis, (x, y), (x+w, y+h), (255, 80, 0), 2)

    for (x, y, w, h) in input_lines:
        cv2.rectangle(vis, (x, y-3), (x+w, y+h+3), (0, 165, 255), 2)

    return vis

# ─────────────────────────────────────────────
# MAIN (PROCESS ALL FILES)
# ─────────────────────────────────────────────

def main():

    os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)
    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

    json_files = [f for f in os.listdir(ANNOTATION_FOLDER) if f.endswith(".json")]

    if not json_files:
        print("No annotation JSON files found.")
        return

    print(f"Found {len(json_files)} files.\n")

    for json_file in json_files:

        base_name = os.path.splitext(json_file)[0]
        image_name = base_name + ".png"
        image_path = os.path.join(DATA_FOLDER, image_name)

        if not os.path.exists(image_path):
            print(f"Image missing for {json_file}")
            continue

        print(f"Processing → {image_name}")

        gray = load_cv_image(image_path)
        h_img, w_img = gray.shape
        binary = preprocess(gray)

        checkboxes = detect_checkboxes(binary, w_img, h_img)
        input_lines = detect_input_lines(binary, w_img, h_img, checkboxes)

        print(f"  Checkboxes : {len(checkboxes)}")
        print(f"  Lines      : {len(input_lines)}")

        save_coordinates(image_name, checkboxes, input_lines)

        vis = draw_results(gray, checkboxes, input_lines)

        annotated_path = os.path.join(OUTPUT_IMAGE_DIR, base_name + "_annotated.png")
        cv2.imwrite(annotated_path, vis)

        print(f"  Image saved → {annotated_path}\n")

    print("All files processed successfully.")

if __name__ == "__main__":
    main()