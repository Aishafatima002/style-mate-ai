
import numpy as np


def _detect_hemline(upper_crop):
    """Detect if the upper garment is SHORT or LONG by scanning for a vertical
    color transition in the lower half of the torso crop.

    HOW IT WORKS:
    - The 'reference color' is sampled from the chest region (top 15-25% of crop).
      This is definitely the upper garment's color.
    - We then scan downward in horizontal bands (from 50% to 85% height).
    - When a band's median color diverges from the reference by > COLOR_THRESH,
      it means we've hit the hemline — the bottom garment's color is bleeding in.
    - SHORT top  → hemline found above hip → crop is trimmed just above the transition.
    - LONG garment (kurta, tunic, dress) → no early transition → keep full crop.

    Returns:
        Trimmed upper crop (may be same as input if garment is long).
    """
    h, w = upper_crop.shape[:2]
    if h < 40:
        return upper_crop  # too small to analyze

    COLOR_THRESH = 42   # Euclidean RGB distance — tuned for fashion colors

    # Reference: MID-CHEST region (rows 28%–45% of crop height).
    # Deliberately avoids the top neckline zone (0-25%) where inner shirts,
    # collars, jewellery, or bag straps can pollute the reference color and
    # cause false hemline triggers that cut the crop too aggressively.
    chest_band = upper_crop[int(h * 0.28): int(h * 0.45), :]
    ref_color = np.median(chest_band.reshape(-1, 3), axis=0).astype(float)

    # Scan bands from 55% downward; look for first band that diverges significantly
    scan_points = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    hemline_row = None
    for pct in scan_points:
        row_start = int(h * pct)
        row_end   = min(h, row_start + max(8, int(h * 0.05)))
        band = upper_crop[row_start:row_end, :]
        band_color = np.median(band.reshape(-1, 3), axis=0).astype(float)
        dist = float(np.linalg.norm(band_color - ref_color))
        if dist > COLOR_THRESH:
            hemline_row = row_start
            break

    if hemline_row is not None:
        # Short top detected: trim crop just above the color transition
        # Add a small buffer (5% of h) so we don't cut off the hem itself
        cut_at = max(int(h * 0.40), hemline_row - int(h * 0.05))
        trimmed = upper_crop[:cut_at, :]
        if trimmed.size > 0:
            return trimmed

    # Long garment (kurta / tunic / long shirt) — keep full crop
    return upper_crop


def _pad_to_square(arr, fill=(220, 220, 220)):
    """Pad a crop to a square canvas with neutral background, preserving aspect ratio.

    This is critical: the CNN was trained on 224×224 images.  When a body-region
    crop is very narrow (e.g. 182×931, aspect ratio 0.20) and is simply resized to
    224×224 it gets squashed so severely that the network cannot recognise the
    garment type. Letterboxing fixes this by keeping the garment's proportions
    intact before the resize step in engine._prep().
    """
    h, w = arr.shape[:2]
    side = max(h, w)
    canvas = np.full((side, side, 3), fill, dtype=np.uint8)
    y_off = (side - h) // 2
    x_off = (side - w) // 2
    canvas[y_off:y_off + h, x_off:x_off + w] = arr
    return canvas


def get_box(points, width, height, padding=0.3):
    """Return (x1, y1, x2, y2) bounding box for the given landmark points.

    Padding is expressed as a fraction of the bounding-box span on each axis.
    We use a reduced default of 0.3 (was 0.5) to avoid over-extending the lower
    crop into the shoes/feet region, which previously caused the lower crop to
    contain mostly feet and background, leading to wrong predictions like Heels.
    """
    xs = [p.x * width for p in points]
    ys = [p.y * height for p in points]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    pad_x = (x_max - x_min) * padding
    pad_y = (y_max - y_min) * padding

    x1 = max(0, int(x_min - pad_x))
    x2 = min(width, int(x_max + pad_x))
    y1 = max(0, int(y_min - pad_y))
    y2 = min(height, int(y_max + pad_y))

    return x1, y1, x2, y2


def crop_garments(image, landmarks, upper_length="short"):
    """Extract upper and lower garment regions from a full-body image.

    Args:
        image: Full-body image RGB array.
        landmarks: MediaPipe Pose landmark list.
        upper_length: 'short' for shirts, t-shirts, sweatshirts, tops, jackets (cuts at waist/hip).
                      'long' for kurtas, tunics, long coats (extends past hips into thighs).

    Returns:
        (upper_crop, lower_crop) — both letterboxed to square so that
        engine._prep() can resize to 224×224 without distorting proportions.
    """
    height, width = image.shape[:2]

    shoulders = [landmarks[11], landmarks[12]]
    hips = [landmarks[23], landmarks[24]]
    ankles = [landmarks[27], landmarks[28]]

    shoulder_y = float(np.mean([lm.y * height for lm in shoulders]))
    hip_y = float(np.mean([lm.y * height for lm in hips]))
    ankle_y = float(np.mean([lm.y * height for lm in ankles]))

    torso_span = max(20.0, hip_y - shoulder_y)
    leg_span = max(20.0, ankle_y - hip_y)

    xs_top = [lm.x * width for lm in shoulders + hips]
    x_min_top, x_max_top = min(xs_top), max(xs_top)
    pad_x = (x_max_top - x_min_top) * 0.3
    ux1 = max(0, int(x_min_top - pad_x))
    ux2 = min(width, int(x_max_top + pad_x))
    uy1 = max(0, int(shoulder_y - torso_span * 0.25))

    if upper_length == "short":
        # Short / standard top: cut cleanly at waist / hip level so trousers don't bleed into the top
        uy2 = min(height, int(shoulder_y + torso_span * 0.95))
    else:
        # Long garment (Kurta / Tunic / Dress): extends past hips down into thighs
        uy2 = min(height, int(hip_y + leg_span * 0.35))

    upper_box = (ux1, uy1, ux2, uy2)

    xs_bot = [lm.x * width for lm in hips + ankles]
    x_min_bot, x_max_bot = min(xs_bot), max(xs_bot)
    pad_x_bot = (x_max_bot - x_min_bot) * 0.3
    lx1 = max(0, int(x_min_bot - pad_x_bot))
    lx2 = min(width, int(x_max_bot + pad_x_bot))

    if upper_length == "short":
        ly1 = max(0, int(hip_y - leg_span * 0.05))
    else:
        ly1 = max(0, int(hip_y + leg_span * 0.20))

    ly2 = min(height, int(hip_y + leg_span * 0.88))
    lower_box = (lx1, ly1, lx2, ly2)

    ux1, uy1, ux2, uy2 = upper_box
    lx1, ly1, lx2, ly2 = lower_box

    upper_crop = image[uy1:uy2, ux1:ux2]
    lower_crop = image[ly1:ly2, lx1:lx2]

    if upper_crop.size == 0 or lower_crop.size == 0:
        raise ValueError("Invalid garment crop: one or both regions are empty")

    # For short top, run hemline detector to catch any remaining pants bleed
    if upper_length == "short":
        upper_crop = _detect_hemline(upper_crop)

    # Letterbox both crops to square so CNN receives properly proportioned input
    upper_sq = _pad_to_square(upper_crop)
    lower_sq = _pad_to_square(lower_crop)

    return upper_sq, lower_sq
