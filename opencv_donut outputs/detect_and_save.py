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
    if region.size == 0: return 1.0
    return np.count_nonzero(region) / region.size

# ─────────────────────────────────────────────
# DETECTION LOGIC
# ─────────────────────────────────────────────

def detect_checkboxes(binary):
    """
    Independent Checkbox Detection: Looks for small square-like contours.
    Does not rely on the table grid.
    """
    checkboxes = []
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Rule: Must be small and square-ish
        if 12 <= w <= 50 and 12 <= h <= 50:
            aspect = w / float(h)
            if 0.8 <= aspect <= 1.2:
                # Rule: Must be a rectangle shape (4-5 points when approximated)
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                
                if len(approx) in [4, 5]:
                    # Rule: Checkboxes are mostly empty (allows an X or Check inside)
                    ink_ratio = get_inner_ink_ratio(binary, x, y, w, h, margin=2)
                    if ink_ratio < 0.35:
                        checkboxes.append((x, y, w, h))
                        
    # Deduplicate perfectly overlapping checkboxes
    kept = []
    for cb in checkboxes:
        if not any(abs(cb[0]-k[0])<5 and abs(cb[1]-k[1])<5 for k in kept):
            kept.append(cb)
    return kept

def detect_input_lines(binary, img_w):
    """
    Strict Line Detection: Splits by vertical lines, ignores text, ensures space above.
    """
    # 1. KILL TEXT: Create mask of ONLY thin items (height <= 12)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    thin_mask = np.zeros_like(binary)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h <= 12: # Only draw thin dashes, dots, and underlines
            cv2.drawContours(thin_mask, [cnt], -1, 255, -1)

    # 2. Connect dashed/dotted lines horizontally
    connected_thin = cv2.morphologyEx(thin_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1)))
    h_lines = cv2.morphologyEx(connected_thin, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1)))

    # 3. Find vertical lines (for slicing)
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30)))
    
    # 4. SLICE horizontal lines where vertical lines intersect
    v_thick = cv2.dilate(v_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)))
    h_lines_split = cv2.bitwise_and(h_lines, cv2.bitwise_not(v_thick))

    # 5. Extract and Validate the lines
    line_cnts, _ = cv2.findContours(h_lines_split, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_lines = []
    for cnt in line_cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 40 or w > img_w * 0.90: 
            continue
            
        # RULE: Must have empty space on top to write
        search_h = 5
        y_top = max(0, y - search_h)
        space_above = binary[y_top:y, x:x+w]
        
        if space_above.size > 0:
            ink_above = np.count_nonzero(space_above) / space_above.size
            if ink_above < 0.15:  # Low ink = empty space to write
                valid_lines.append((x, y, w, h))

    return valid_lines

def detect_input_boxes(binary, img_w, img_h, checkboxes, lines):
    """
    Detects table cells and fillable boxes. Fills 3-sided open boxes.
    """
    # Extract robust structural grid
    h_struct = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1)))
    v_struct = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30)))
    grid = cv2.add(h_struct, v_struct)
    
    # RULE: Form 4th wall on 3-sided boxes (Close gaps heavily)
    grid_closed = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)))
    
    box_cnts, _ = cv2.findContours(grid_closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    raw_boxes = []
    for cnt in box_cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        # Size limits (Bigger than checkbox, smaller than full page)
        if 40 <= w <= img_w * 0.95 and 20 <= h <= img_h * 0.90:
            
            # RULE: Box must be mostly empty cell
            ink_inside = get_inner_ink_ratio(binary, x, y, w, h, margin=6)
            if ink_inside < 0.25: # Strict: 75%+ empty space
                raw_boxes.append((x, y, w, h))

    # Remove duplicates
    raw_boxes.sort(key=lambda b: b[2]*b[3]) # Smallest area first
    unique_boxes = []
    for box in raw_boxes:
        if not any(abs(box[0]-u[0])<10 and abs(box[1]-u[1])<10 and abs(box[2]-u[2])<10 for u in unique_boxes):
            unique_boxes.append(box)

    # RULE: If a box contains a checkbox or line, it is NOT a fillable box (it's a container)
    final_boxes = []
    for box in unique_boxes:
        is_container = False
        
        for cb in checkboxes:
            if rect_contains(box, cb, tol=5): 
                is_container = True; break
                
        for line in lines:
            if rect_contains(box, line, tol=5): 
                is_container = True; break
                
        # Also ensure it isn't completely swallowing another valid smaller box
        for other_box in unique_boxes:
            if box != other_box and rect_contains(box, other_box, tol=2):
                is_container = True; break
                
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
    gray = np.array(img_pil)
    h_img, w_img = gray.shape
    
    # Preprocessing
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    
    # 1. Detect Checkboxes
    checkboxes = detect_checkboxes(binary)
    
    # 2. Detect Lines (Sliced, Space Above)
    input_lines = detect_input_lines(binary, w_img)
    
    # 3. Detect Boxes (Mostly Empty, Closed Walls, Not Containers)
    input_boxes = detect_input_boxes(binary, w_img, h_img, checkboxes, input_lines)

    # Save JSON
    json_path = os.path.join(JSON_OUTPUT_DIR, f"{base_name}.json")
    payload = {
        "checkboxes": [[round(float(x+w/2), 2), round(float(y+h/2), 2)] for (x,y,w,h) in checkboxes],
        "input_boxes": [[float(x), float(y), float(x+w), float(y+h)] for (x,y,w,h) in input_boxes],
        "lines": [[float(x), round(float(y+h/2), 2), float(x+w), round(float(y+h/2), 2)] for (x,y,w,h) in input_lines],
        "meta": {"source_file": file_name, "image_size": [w_img, h_img]}
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    # Save Annotated Image
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h) in checkboxes: cv2.rectangle(vis, (x, y), (x+w, y+h), (255, 80, 0), 2)  # Blue
    for (x, y, w, h) in input_boxes: cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 200, 80), 2) # Green
    for (x, y, w, h) in input_lines: cv2.rectangle(vis, (x, y-3), (x+w, y+h+3), (0, 165, 255), 2) # Orange
    
    cv2.imwrite(os.path.join(IMAGE_OUTPUT_DIR, f"{base_name}_annotated.png"), vis)

def main():
    if not os.path.exists(DATA_FOLDER):
        print(f"Error: Data folder not found at {DATA_FOLDER}")
        return

    files = [f for f in os.listdir(DATA_FOLDER) if f.lower().endswith(VALID_EXTENSIONS)]
    
    if not files:
        print(f"No valid images found in {DATA_FOLDER}")
        return

    print(f"Processing {len(files)} files with new strict rules...")
    for filename in files:
        try:
            process_file(os.path.join(DATA_FOLDER, filename))
            print(f"  [DONE] {filename}")
        except Exception as e:
            print(f"  [FAIL] {filename}: {e}")

    print(f"\nBatch processing complete.")

if __name__ == "__main__":
    main()