import os
import json
import cv2
import numpy as np
from PIL import Image

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER      = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'data'))

BASE_OUTPUT_DIR  = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'annotated_output'))
IMAGE_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, 'image')
JSON_OUTPUT_DIR  = os.path.join(BASE_OUTPUT_DIR, 'json')

os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)

VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp')


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def rect_contains(outer, inner, tol=5):
    ox, oy, ow, oh = outer
    ix, iy, iw, ih = inner
    return (ix >= ox - tol and iy >= oy - tol and
            ix + iw <= ox + ow + tol and iy + ih <= oy + oh + tol)

def get_inner_ink_ratio(binary_img, x, y, w, h, margin=4):
    """Calculates the percentage of black pixels inside a bounding box."""
    ix, iy = max(0, x + margin), max(0, y + margin)
    iw, ih = max(1, w - 2*margin), max(1, h - 2*margin)
    region = binary_img[iy:iy+ih, ix:ix+iw]
    if region.size == 0:
        return 1.0
    return np.count_nonzero(region) / region.size

def is_near_page_edge(x, y, w, h, img_w, img_h, margin_frac=0.03):
    """
    Returns True if this element's centre sits within the top or bottom
    edge zone of the page (fraction of page height).
    """
    margin_px = int(img_h * margin_frac)
    cy = y + h / 2
    if cy < margin_px:
        return True
    if cy > img_h - margin_px:
        return True
    return False

def is_page_border_line(x, y, w, h, img_w, img_h,
                        span_thresh=0.85, edge_frac=0.05):
    """
    Returns True when a line looks like a page border/rule rather than
    an input field:
      - Spans nearly the full page width, OR
      - Sits in the top/bottom edge zone
    """
    if w >= img_w * span_thresh:
        return True
    if is_near_page_edge(x, y, w, h, img_w, img_h, margin_frac=edge_frac):
        return True
    return False

def is_inside_box(x, y, w, h, boxes, tol=4):
    """Returns True if this element sits fully inside one of *boxes*."""
    for (bx, by, bw, bh) in boxes:
        if rect_contains((bx, by, bw, bh), (x, y, w, h), tol=tol):
            return True
    return False

def deduplicate(items, xy_tol=6, wh_tol=8):
    """Remove near-duplicate bounding boxes."""
    kept = []
    for item in items:
        duplicate = False
        for k in kept:
            if (abs(item[0] - k[0]) < xy_tol and
                    abs(item[1] - k[1]) < xy_tol and
                    abs(item[2] - k[2]) < wh_tol and
                    abs(item[3] - k[3]) < wh_tol):
                duplicate = True
                break
        if not duplicate:
            kept.append(item)
    return kept


# ─────────────────────────────────────────────
# DETECTION LOGIC
# ─────────────────────────────────────────────

def detect_checkboxes(binary, img_w, img_h):
    """
    Checkbox Detection: small square-like hollow contours.
    Rejects page-edge artefacts and shapes that are too filled.
    """
    checkboxes = []
    contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Must be small and square-ish
        if not (12 <= w <= 55 and 12 <= h <= 55):
            continue
        aspect = w / float(h)
        if not (0.75 <= aspect <= 1.33):
            continue

        # Must approximate to a rectangle (4–5 vertices)
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) not in [4, 5]:
            continue

        # Mostly empty interior
        ink_ratio = get_inner_ink_ratio(binary, x, y, w, h, margin=2)
        if ink_ratio >= 0.40:
            continue

        # Not at the very edge of the page
        if is_near_page_edge(x, y, w, h, img_w, img_h, margin_frac=0.02):
            continue

        # The drawn border itself must be inky (hollow-box test)
        border_mask = np.zeros_like(binary)
        cv2.drawContours(border_mask, [cnt], -1, 255, 2)
        roi_bin    = binary[y:y+h, x:x+w]
        roi_border = border_mask[y:y+h, x:x+w]
        denom = max(1, np.count_nonzero(roi_border))
        border_ink = np.count_nonzero(cv2.bitwise_and(roi_bin, roi_border)) / denom
        if border_ink < 0.30:
            continue

        checkboxes.append((x, y, w, h))

    return deduplicate(checkboxes, xy_tol=5, wh_tol=6)


def detect_structural_boxes_mask(binary, img_w, img_h):
    """
    Builds the structural grid (horizontal + vertical rules) and returns:
      - a list of all structural bounding boxes (used to filter lines)
      - the closed grid image (used to detect fillable input boxes)
    """
    h_struct = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1)))
    v_struct = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40)))
    grid = cv2.add(h_struct, v_struct)

    # Seal 3-sided open boxes
    grid_closed = cv2.morphologyEx(
        grid, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20)))

    cnts, _ = cv2.findContours(grid_closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        if w >= 30 and h >= 12:
            boxes.append((x, y, w, h))
    return boxes, grid_closed


def detect_input_lines(binary, img_w, img_h, structural_boxes):
    """
    Input Line Detection — strict multi-stage filtering:

    1. Only thin horizontal strokes (contour height ≤ 10 px)
    2. Reject page-border / full-width rules (>= 85 % of page width,
       or centre-y in the top/bottom 4 % of the page)
    3. Reject lines that live fully inside a structural table box
       (those are cell separators, not writable fields)
    4. Require blank writing space directly above
    5. Width gate: 35 px minimum, 88 % of page width maximum
    """
    # ── Step 1: keep only thin items ─────────────────────────────────────────
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    thin_mask = np.zeros_like(binary)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h <= 10:
            cv2.drawContours(thin_mask, [cnt], -1, 255, -1)

    # ── Step 2: connect dashes, then enforce minimum line length ─────────────
    connected = cv2.morphologyEx(
        thin_mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (18, 1)))
    h_lines = cv2.morphologyEx(
        connected, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1)))

    # ── Step 3: cut where vertical rules cross ───────────────────────────────
    v_lines = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30)))
    v_thick = cv2.dilate(
        v_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (6, 1)))
    h_lines_split = cv2.bitwise_and(h_lines, cv2.bitwise_not(v_thick))

    # ── Step 4: validate each candidate ──────────────────────────────────────
    line_cnts, _ = cv2.findContours(
        h_lines_split, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_lines = []

    for cnt in line_cnts:
        x, y, w, h = cv2.boundingRect(cnt)

        # Width gate
        if w < 35 or w > img_w * 0.88:
            continue

        # ── REJECT page-border / header / footer rules ────────────────────
        if is_page_border_line(x, y, w, h, img_w, img_h,
                               span_thresh=0.85, edge_frac=0.04):
            continue

        # ── REJECT cell-separator lines inside structural table boxes ─────
        if is_inside_box(x, y, w, h, structural_boxes, tol=6):
            continue

        # ── Require empty writing space above ────────────────────────────
        look_up = max(8, h * 2)
        y_top   = max(0, y - look_up)
        space   = binary[y_top:y, x:x + w]
        if space.size > 0:
            ink_above = np.count_nonzero(space) / space.size
            if ink_above > 0.12:
                continue

        valid_lines.append((x, y, w, h))

    return deduplicate(valid_lines, xy_tol=6, wh_tol=10)


def detect_input_boxes(binary, img_w, img_h,
                       checkboxes, lines,
                       structural_boxes, grid_closed):
    """
    Fillable box detection from the sealed structural grid.

    Filters out:
      - Full-page-width spans (page frame / section banners)
      - Elements at page edges
      - Boxes whose interior is too inky (contain content / are headers)
      - Containers (boxes that hold checkboxes, lines, or other boxes)
    """
    cnts, _ = cv2.findContours(grid_closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    raw_boxes = []
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)

        # Size gate
        if not (35 <= w <= img_w * 0.92 and 14 <= h <= img_h * 0.85):
            continue

        # Reject full-page-width elements
        if w >= img_w * 0.90:
            continue

        # Reject page-edge elements
        if is_near_page_edge(x, y, w, h, img_w, img_h, margin_frac=0.03):
            continue

        # Interior must be mostly empty
        ink_inside = get_inner_ink_ratio(binary, x, y, w, h, margin=6)
        if ink_inside >= 0.22:
            continue

        raw_boxes.append((x, y, w, h))

    # Sort smallest area first; deduplicate
    raw_boxes.sort(key=lambda b: b[2] * b[3])
    unique_boxes = deduplicate(raw_boxes, xy_tol=10, wh_tol=10)

    # Remove containers
    final_boxes = []
    for box in unique_boxes:
        is_container = False

        for cb in checkboxes:
            if rect_contains(box, cb, tol=5):
                is_container = True
                break

        if not is_container:
            for line in lines:
                if rect_contains(box, line, tol=5):
                    is_container = True
                    break

        if not is_container:
            for other in unique_boxes:
                if other != box and rect_contains(box, other, tol=3):
                    is_container = True
                    break

        if not is_container:
            final_boxes.append(box)

    return final_boxes


# ─────────────────────────────────────────────
# BATCH PROCESSOR
# ─────────────────────────────────────────────

def process_file(file_path):
    file_name = os.path.basename(file_path)
    base_name = os.path.splitext(file_name)[0]

    img_pil = Image.open(file_path).convert("L")
    gray    = np.array(img_pil)
    h_img, w_img = gray.shape

    # Preprocessing
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary  = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 10)

    # 1. Build structural grid
    structural_boxes, grid_closed = detect_structural_boxes_mask(binary, w_img, h_img)

    # 2. Detect Checkboxes
    checkboxes = detect_checkboxes(binary, w_img, h_img)

    # 3. Detect Input Lines (filtered against structural boxes)
    input_lines = detect_input_lines(binary, w_img, h_img, structural_boxes)

    # 4. Detect Fillable Boxes
    input_boxes = detect_input_boxes(
        binary, w_img, h_img,
        checkboxes, input_lines,
        structural_boxes, grid_closed)

    # Save JSON
    json_path = os.path.join(JSON_OUTPUT_DIR, f"{base_name}.json")
    payload = {
        "checkboxes": [
            [round(float(x + w / 2), 2), round(float(y + h / 2), 2)]
            for (x, y, w, h) in checkboxes
        ],
        "input_boxes": [
            [float(x), float(y), float(x + w), float(y + h)]
            for (x, y, w, h) in input_boxes
        ],
        "lines": [
            [float(x), round(float(y + h / 2), 2),
             float(x + w), round(float(y + h / 2), 2)]
            for (x, y, w, h) in input_lines
        ],
        "meta": {
            "source_file": file_name,
            "image_size": [w_img, h_img]
        }
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    # Save Annotated Image
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h) in checkboxes:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 80,  0), 2)   # Blue
    for (x, y, w, h) in input_boxes:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200,  80), 2)   # Green
    for (x, y, w, h) in input_lines:
        cv2.rectangle(vis, (x, y - 3), (x + w, y + h + 3), (0, 165, 255), 2)  # Orange

    cv2.imwrite(os.path.join(IMAGE_OUTPUT_DIR, f"{base_name}_annotated.png"), vis)


def main():
    if not os.path.exists(DATA_FOLDER):
        print(f"Error: Data folder not found at {DATA_FOLDER}")
        return

    files = [f for f in os.listdir(DATA_FOLDER)
             if f.lower().endswith(VALID_EXTENSIONS)]

    if not files:
        print(f"No valid images found in {DATA_FOLDER}")
        return

    print(f"Processing {len(files)} files...")
    for filename in files:
        try:
            process_file(os.path.join(DATA_FOLDER, filename))
            print(f"  [DONE] {filename}")
        except Exception as e:
            print(f"  [FAIL] {filename}: {e}")

    print(f"\nBatch processing complete.")


if __name__ == "__main__":
    main()