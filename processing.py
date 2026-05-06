import cv2
import numpy as np
from typing import Optional


# ── ArUco detection ────────────────────────────────────────────────────────

def _make_detector():
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    p = cv2.aruco.DetectorParameters()
    # Improve detection on challenging images
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 53
    p.adaptiveThreshWinSizeStep = 4
    p.minMarkerPerimeterRate = 0.02
    p.maxMarkerPerimeterRate = 4.0
    try:
        det = cv2.aruco.ArucoDetector(d, p)
        return det, d
    except AttributeError:
        return None, d


_DETECTOR, _DICT = _make_detector()


def detect_markers(img: np.ndarray) -> list[dict]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if _DETECTOR is not None:
        corners, ids, _ = _DETECTOR.detectMarkers(gray)
    else:
        p = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(gray, _DICT, parameters=p)

    if ids is None:
        return []
    return [
        {"id": int(ids[i][0]), "corners": corners[i].reshape(4, 2)}
        for i in range(len(ids))
    ]


# ── Homography ─────────────────────────────────────────────────────────────

def compute_homography(
    markers: list[dict], marker_size_mm: float
) -> tuple[Optional[np.ndarray], float]:
    """
    Returns H (image pixels → world mm) and mm_per_px (rough scale from image).
    Multiple separate markers are handled by bootstrapping world coords from the
    first marker and accumulating all correspondences for a global RANSAC fit.
    """
    if not markers:
        return None, 0.0

    S = float(marker_size_mm)
    # ArUco corner order: TL, TR, BR, BL
    local_w = np.array([[0, 0], [S, 0], [S, S], [0, S]], dtype=np.float32)

    img0 = markers[0]["corners"].astype(np.float32)
    H, _ = cv2.findHomography(img0, local_w)

    if len(markers) > 1 and H is not None:
        all_img = img0.copy()
        all_world = local_w.copy()
        for m in markers[1:]:
            img_m = m["corners"].astype(np.float32)
            # Project into world frame of marker 0 using current H
            world_m = cv2.perspectiveTransform(img_m.reshape(1, -1, 2), H).reshape(4, 2)
            all_img = np.vstack([all_img, img_m])
            all_world = np.vstack([all_world, world_m])
        H, _ = cv2.findHomography(all_img, all_world, cv2.RANSAC, 3.0)

    if H is None:
        return None, 0.0

    # Estimate scale from first marker
    img0 = markers[0]["corners"].astype(np.float32)
    px_side = float(np.linalg.norm(img0[1] - img0[0]))
    mm_per_px = S / px_side if px_side > 0 else 1.0
    return H, mm_per_px


# ── Image warping ──────────────────────────────────────────────────────────

WorldBounds = tuple[float, float, float, float]  # min_x, min_y, max_x, max_y in mm


def warp_image(
    img: np.ndarray,
    H: np.ndarray,
    mm_per_px: float,
    output_px_per_mm: float = 10.0,
    max_dim: int = 4000,
) -> tuple[np.ndarray, float, WorldBounds]:
    """
    Warp image to a rectilinear, top-down view.
    Returns (warped_image, actual_px_per_mm, world_bounds).
    world_bounds = (min_x, min_y, max_x, max_y) in world mm — pass to
    warp_to_bounds() to align a second image to the same grid.
    """
    h, w = img.shape[:2]
    corners_img = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    corners_world = cv2.perspectiveTransform(corners_img.reshape(1, -1, 2), H).reshape(4, 2)

    min_x, min_y = corners_world.min(axis=0)
    max_x, max_y = corners_world.max(axis=0)

    out_w = int((max_x - min_x) * output_px_per_mm)
    out_h = int((max_y - min_y) * output_px_per_mm)

    if max(out_w, out_h) > max_dim:
        scale = max_dim / max(out_w, out_h)
        out_w = int(out_w * scale)
        out_h = int(out_h * scale)
        output_px_per_mm *= scale

    T = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float64)
    S = np.diag([output_px_per_mm, output_px_per_mm, 1.0])
    H_warp = S @ T @ H.astype(np.float64)

    warped = cv2.warpPerspective(img, H_warp, (out_w, out_h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(200, 200, 200))
    return warped, output_px_per_mm, (min_x, min_y, max_x, max_y)


def warp_to_bounds(
    img: np.ndarray,
    H: np.ndarray,
    bounds: WorldBounds,
    output_px_per_mm: float,
) -> np.ndarray:
    """
    Warp img into a fixed world-space grid defined by bounds.
    Used to align a reference photo to the same pixel grid as the tool photo,
    even if the camera moved slightly between shots (each image uses its own H).
    """
    min_x, min_y, max_x, max_y = bounds
    out_w = int((max_x - min_x) * output_px_per_mm)
    out_h = int((max_y - min_y) * output_px_per_mm)
    T = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float64)
    S = np.diag([output_px_per_mm, output_px_per_mm, 1.0])
    H_warp = S @ T @ H.astype(np.float64)
    return cv2.warpPerspective(img, H_warp, (out_w, out_h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(200, 200, 200))


# ── Silhouette extraction ──────────────────────────────────────────────────

def _erase_markers(img: np.ndarray, ppmm: float, fill: tuple = (255, 255, 255)) -> np.ndarray:
    """
    Re-detect ArUco markers in the warped image and paint over them.
    In warped space markers are near-perfect rectangles so detection is reliable.
    A 5 mm padding avoids the quiet-zone border leaking into the mask.
    """
    result = img.copy()
    markers = detect_markers(img)
    if not markers:
        return result
    pad = int(5 * ppmm)
    for m in markers:
        c = m["corners"].astype(int)
        x1 = max(0, c[:, 0].min() - pad)
        y1 = max(0, c[:, 1].min() - pad)
        x2 = min(img.shape[1], c[:, 0].max() + pad)
        y2 = min(img.shape[0], c[:, 1].max() + pad)
        result[y1:y2, x1:x2] = fill
    return result


def _color_aware_signal(img: np.ndarray, dark_on_light: bool, color_weight: float) -> np.ndarray:
    """
    Compute a per-pixel 'how different from background' signal in [0,255].
    color_weight=0 → pure grayscale.  color_weight>0 → saturation adds to the
    signal, so colored tools score higher than neutral gray shadows even when
    they have the same luminance.
    """
    if color_weight <= 0:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray if not dark_on_light else (255 - gray)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    V = hsv[:, :, 2]  # brightness 0-255
    S = hsv[:, :, 1]  # saturation 0-255

    if dark_on_light:
        signal = (255.0 - V) + S * color_weight   # dark + saturated → high
    else:
        signal = V + S * color_weight              # bright + saturated → high

    return np.clip(signal, 0, 255).astype(np.uint8)


def _gradient_snap(mask: np.ndarray, gray: np.ndarray,
                    edge_min: int, ppmm: float) -> np.ndarray:
    """
    Iteratively peel boundary pixels whose local gradient is below edge_min.

    Shadow penumbras are gradual intensity ramps (low gradient) → get peeled away.
    Tool edges are sharp transitions (high gradient) → stop the erosion naturally.

    The peel continues until every boundary pixel lies on a sharp gradient or the
    mask has been eroded by at most 50 mm.  The remnant shadow right at the
    tool-shadow junction (where both are dark and gradient is moderate) is a
    fundamental limit — it adds ~1-3 mm to the silhouette, absorbed by kerf margin.
    """
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sharp = np.sqrt(gx**2 + gy**2) >= edge_min

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    current = mask.astype(bool)
    max_iters = min(int(ppmm * 50), 600)

    for _ in range(max_iters):
        eroded   = cv2.erode(current.astype(np.uint8), k3).astype(bool)
        boundary = current & ~eroded
        if not np.any(boundary):
            break
        weak = boundary & ~sharp
        if not np.any(weak):
            break
        current = current & ~weak

    return (current.astype(np.uint8) * 255)


def _outer_contours_only(mask: np.ndarray, fill_px: int, noise_px: int) -> np.ndarray:
    """
    Close gaps to fill interior detail, then keep only outer contours.
    Discards holes, text, surface markings — leaves one solid region per tool.
    """
    k_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*fill_px+1, 2*fill_px+1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_fill)

    k_noise = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*noise_px+1, 2*noise_px+1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_noise)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(mask)
    cv2.drawContours(clean, contours, -1, 255, cv2.FILLED)
    return clean


def extract_silhouette(
    img: np.ndarray,
    threshold: int = 128,
    dark_on_light: bool = True,
    ppmm: float = 10.0,
    fill_interior_mm: float = 3.0,
    color_weight: float = 0.0,
    edge_guided: bool = False,
    edge_min: int = 20,
) -> np.ndarray:
    bg = (255, 255, 255) if dark_on_light else (0, 0, 0)
    img = _erase_markers(img, ppmm, fill=bg)

    signal = _color_aware_signal(img, dark_on_light, color_weight)
    # signal is always "high = tool-like", so threshold as BINARY
    _, mask = cv2.threshold(signal, threshold, 255, cv2.THRESH_BINARY)

    if edge_guided:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = _gradient_snap(mask, gray, edge_min, ppmm)

    fill_px  = max(2, int(fill_interior_mm * ppmm))
    noise_px = max(1, int(1.0 * ppmm))
    return _outer_contours_only(mask, fill_px, noise_px)


def extract_silhouette_reference(
    warped_tool: np.ndarray,
    warped_ref: np.ndarray,
    threshold: int = 25,
    ppmm: float = 10.0,
    fill_interior_mm: float = 3.0,
    edge_guided: bool = False,
    edge_min: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Background subtraction by differencing tool photo vs reference photo.
    Both must already be warped to the same world-space grid (use warp_to_bounds).
    Returns (mask, diff_image). threshold = minimum pixel difference to count as tool.

    Note: specular reflections that shift between shots will appear as false positives.
    A Gaussian blur (kernel ~ 2 mm) suppresses isolated specular highlights but cannot
    eliminate reflections that span several mm. A matte surface is the reliable fix.
    """
    # Erase markers from both so their edges don't produce diff artefacts
    warped_tool = _erase_markers(warped_tool, ppmm)
    warped_ref  = _erase_markers(warped_ref,  ppmm)

    h = min(warped_tool.shape[0], warped_ref.shape[0])
    w = min(warped_tool.shape[1], warped_ref.shape[1])
    tool = warped_tool[:h, :w]
    ref  = warped_ref[:h, :w]

    # Blur suppresses point-source specular highlights (kernel ~ 2 mm)
    blur_k = max(3, int(2 * ppmm) | 1)
    gray_tool = cv2.GaussianBlur(cv2.cvtColor(tool, cv2.COLOR_BGR2GRAY), (blur_k, blur_k), 0)
    gray_ref  = cv2.GaussianBlur(cv2.cvtColor(ref,  cv2.COLOR_BGR2GRAY), (blur_k, blur_k), 0)

    diff = cv2.absdiff(gray_tool, gray_ref)
    diff_vis = cv2.applyColorMap(
        cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX), cv2.COLORMAP_INFERNO
    )

    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    if edge_guided:
        gray_tool_full = cv2.cvtColor(warped_tool[:h, :w], cv2.COLOR_BGR2GRAY)
        mask = _gradient_snap(mask, gray_tool_full, edge_min, ppmm)

    fill_px  = max(2, int(fill_interior_mm * ppmm))
    noise_px = max(1, int(1.0 * ppmm))
    return _outer_contours_only(mask, fill_px, noise_px), diff_vis


# ── Depth model silhouette ─────────────────────────────────────────────────

_depth_pipe = None
_depth_lock = None


def _get_depth_pipe():
    global _depth_pipe, _depth_lock
    import threading
    if _depth_lock is None:
        _depth_lock = threading.Lock()
    with _depth_lock:
        if _depth_pipe is None:
            import torch
            from transformers import pipeline as hf_pipeline
            device = 0 if torch.cuda.is_available() else -1
            _depth_pipe = hf_pipeline(
                "depth-estimation",
                model="depth-anything/Depth-Anything-V2-Small-hf",
                device=device,
            )
    return _depth_pipe


def extract_silhouette_depth(
    warped: np.ndarray,
    ppmm: float,
    fill_interior_mm: float = 3.0,
    depth_percentile: float = 95.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract silhouette using monocular depth estimation (Depth Anything V2 Small).
    Colour/reflectance-independent — works on any tool colour or finish.
    First call downloads ~100 MB of model weights.

    depth_percentile: pixels in the top (100 - percentile)% of depth values are
    treated as objects.  95 = top 5% = closest region.  Lower values are more
    inclusive; raise if background noise contaminates the mask.

    Returns (mask, depth_vis) where depth_vis is a colourised depth map for preview.
    """
    from PIL import Image

    pipe = _get_depth_pipe()

    img_clean = _erase_markers(warped, ppmm, fill=(128, 128, 128))
    pil = Image.fromarray(cv2.cvtColor(img_clean, cv2.COLOR_BGR2RGB))
    result = pipe(pil)

    depth = result["predicted_depth"]
    if hasattr(depth, "numpy"):
        depth = depth.squeeze().numpy()
    else:
        depth = np.array(depth).squeeze()

    depth = cv2.resize(depth.astype(np.float32),
                       (warped.shape[1], warped.shape[0]),
                       interpolation=cv2.INTER_LINEAR)

    # Identify gray fill border pixels (warpPerspective fill = 200,200,200)
    gray_fill = np.all(np.abs(warped.astype(np.int32) - 200) < 8, axis=2)

    # Normalise to 0-255 using valid-area range
    valid = depth[~gray_fill]
    d_min, d_max = float(valid.min()), float(valid.max())
    if d_max <= d_min:
        blank = np.zeros(warped.shape[:2], dtype=np.uint8)
        return blank, blank

    depth_norm = ((depth - d_min) / (d_max - d_min) * 255).clip(0, 255).astype(np.uint8)
    depth_norm[gray_fill] = 0

    # Threshold: pixels in the top (100-percentile)% are closest to camera = on-board objects.
    # Depth Anything V2 uses disparity convention: larger value = closer.
    thresh = float(np.percentile(depth_norm[~gray_fill], depth_percentile))
    _, mask = cv2.threshold(depth_norm, int(thresh), 255, cv2.THRESH_BINARY)
    mask[gray_fill] = 0

    depth_vis = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)

    fill_px  = max(2, int(fill_interior_mm * ppmm))
    noise_px = max(1, int(1.0 * ppmm))
    mask = _outer_contours_only(mask, fill_px, noise_px)

    # Smooth the boundary: Gaussian blur-and-rethreshold removes sub-mm pixel
    # zigzags that RDP simplification can't eliminate.  ~1 mm sigma is enough to
    # clean depth-model noise without rounding real features.
    sigma = max(1.0, ppmm * 1.0)
    ksize = max(3, int(sigma * 4) | 1)
    mask_f = cv2.GaussianBlur(mask.astype(np.float32), (ksize, ksize), sigma)
    _, mask = cv2.threshold(mask_f, 127, 255, cv2.THRESH_BINARY)

    return mask.astype(np.uint8), depth_vis


# ── SVG generation ─────────────────────────────────────────────────────────

def _contours_to_path_d(contours, px_per_mm: float, epsilon_px: float) -> str:
    min_area_px2 = (2.0 * px_per_mm) ** 2  # ignore blobs smaller than ~4 mm²
    parts = []
    for c in contours:
        if cv2.contourArea(c) < min_area_px2:
            continue
        simp = cv2.approxPolyDP(c, epsilon_px, True)
        if len(simp) < 3:
            continue
        pts = simp.reshape(-1, 2) / px_per_mm
        d = f"M {pts[0,0]:.3f},{pts[0,1]:.3f}"
        for pt in pts[1:]:
            d += f" L {pt[0]:.3f},{pt[1]:.3f}"
        d += " Z"
        parts.append(d)
    return " ".join(parts)


def generate_svg(
    mask: np.ndarray,
    px_per_mm: float,
    epsilon_mm: float = 0.5,
    margin_mm: float = 0.0,
) -> str:
    # Dilate by (margin_mm + epsilon_mm) before extracting and simplifying contours.
    # The epsilon_mm buffer ensures that when RDP removes vertices and replaces curved
    # sections with straight chords, those chords never cut inside the original tool —
    # the chord can deviate at most epsilon_mm inward from the dilated contour, which
    # still leaves it epsilon_mm outside the original.  margin_mm sits on top for kerf.
    total_expand_px = max(1, int((margin_mm + epsilon_mm) * px_per_mm))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*total_expand_px+1, 2*total_expand_px+1))
    mask = cv2.dilate(mask, k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    if not contours:
        return _empty_svg(mask, px_per_mm)

    epsilon_px = epsilon_mm * px_per_mm
    path_d = _contours_to_path_d(contours, px_per_mm, epsilon_px)

    h, w = mask.shape
    W = w / px_per_mm
    H = h / px_per_mm

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg"\n'
        f'     width="{W:.3f}mm" height="{H:.3f}mm"\n'
        f'     viewBox="0 0 {W:.3f} {H:.3f}">\n'
        f'  <path d="{path_d}" fill="black" stroke="none"/>\n'
        "</svg>"
    )


def _empty_svg(mask, px_per_mm):
    h, w = mask.shape
    W, H = w / px_per_mm, h / px_per_mm
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.3f}mm" height="{H:.3f}mm" '
        f'viewBox="0 0 {W:.3f} {H:.3f}"></svg>'
    )
