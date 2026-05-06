import base64
import io
import tempfile
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from datum import generate_datum_svg
from typing import Optional

from processing import (
    compute_homography,
    detect_markers,
    extract_silhouette,
    extract_silhouette_depth,
    extract_silhouette_reference,
    generate_svg,
    warp_image,
    warp_to_bounds,
)

app = FastAPI(title="Shadow Board Generator")

STATIC = Path(__file__).parent / "static"
TMPDIR = Path(tempfile.gettempdir()) / "shadowboard"
TMPDIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


def _encode_jpg(img: np.ndarray, quality: int = 82) -> str:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode()


def _encode_png(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode()


def _scale_for_preview(img: np.ndarray, max_dim: int = 1400) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    s = max_dim / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s)))


# ── Pages ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    return (STATIC / "index.html").read_text()


# ── Datum sheet download ────────────────────────────────────────────────────

@app.get("/datum.svg")
async def datum_svg(
    marker_size_mm: float = 60.0,
    paper: str = "A4",
):
    svg = generate_datum_svg(marker_size_mm=marker_size_mm, paper=paper)
    name = f"datum_{paper}_{int(marker_size_mm)}mm.svg"
    return Response(
        content=svg,
        media_type="image/svg+xml",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


# ── Photo analysis (returns previews + session key) ─────────────────────────

def _load_img(raw: bytes) -> np.ndarray:
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Cannot decode image")
    return img


def _draw_overlay(img: np.ndarray, markers: list) -> np.ndarray:
    overlay = img.copy()
    for m in markers:
        c = m["corners"].astype(int)
        cv2.polylines(overlay, [c.reshape(-1, 1, 2)], True, (0, 220, 50), 3)
        ctr = c.mean(axis=0).astype(int)
        cv2.putText(overlay, f"ID {m['id']}", tuple(ctr - [20, 0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 50), 2)
    return overlay


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    ref_file: Optional[UploadFile] = File(None),
    marker_size_mm: float = Form(60.0),
    threshold: int = Form(128),
    dark_on_light: bool = Form(True),
    output_px_per_mm: float = Form(10.0),
    fill_interior_mm: float = Form(3.0),
    color_weight: float = Form(0.0),
    edge_guided: bool = Form(False),
    edge_min: int = Form(20),
    depth_mode: bool = Form(False),
    depth_percentile: float = Form(95.0),
):
    raw = await file.read()
    img = _load_img(raw)

    markers = detect_markers(img)
    if not markers:
        return JSONResponse({
            "success": False,
            "error": "No ArUco markers detected in the tool photo. Make sure the datum "
                     "sheet is visible, flat, and well-lit with no strong reflections.",
            "markers_found": 0,
        })

    H, mm_per_px = compute_homography(markers, marker_size_mm)
    if H is None:
        return JSONResponse({
            "success": False,
            "error": "Could not compute homography. Try with more markers visible.",
            "markers_found": len(markers),
        })

    warped, actual_ppmm, bounds = warp_image(img, H, mm_per_px, output_px_per_mm)

    token = str(uuid.uuid4())
    (TMPDIR / f"{token}.jpg").write_bytes(raw)

    ref_mode = False
    diff_img_b64 = None

    if depth_mode:
        mask, depth_vis = extract_silhouette_depth(warped, actual_ppmm, fill_interior_mm, depth_percentile)
        diff_img_b64 = _encode_jpg(_scale_for_preview(depth_vis))
    elif ref_file is not None:
        ref_raw = await ref_file.read()
        ref_img = _load_img(ref_raw)
        ref_markers = detect_markers(ref_img)

        if not ref_markers:
            H_ref = H
        else:
            H_ref, _ = compute_homography(ref_markers, marker_size_mm)
            if H_ref is None:
                H_ref = H

        warped_ref = warp_to_bounds(ref_img, H_ref, bounds, actual_ppmm)
        mask, diff_vis = extract_silhouette_reference(
            warped, warped_ref, threshold, actual_ppmm, fill_interior_mm,
            edge_guided, edge_min)
        (TMPDIR / f"{token}_ref.jpg").write_bytes(ref_raw)
        ref_mode = True
        diff_img_b64 = _encode_jpg(_scale_for_preview(diff_vis))
    else:
        mask = extract_silhouette(warped, threshold, dark_on_light, actual_ppmm,
                                   fill_interior_mm, color_weight, edge_guided, edge_min)

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    return JSONResponse({
        "success": True,
        "session": token,
        "ref_mode": ref_mode,
        "depth_mode": depth_mode,
        "markers_found": len(markers),
        "marker_ids": [m["id"] for m in markers],
        "width_mm":  round(warped.shape[1] / actual_ppmm, 1),
        "height_mm": round(warped.shape[0] / actual_ppmm, 1),
        "px_per_mm": round(actual_ppmm, 2),
        "detection_img": _encode_jpg(_scale_for_preview(_draw_overlay(img, markers))),
        "warped_img":    _encode_jpg(_scale_for_preview(warped)),
        "diff_img":      diff_img_b64,
        "mask_img":      _encode_png(_scale_for_preview(mask_bgr)),
    })


# ── Re-threshold only (returns new mask preview, fast) ─────────────────────

@app.post("/rethreshold")
async def rethreshold(
    session: str = Form(...),
    marker_size_mm: float = Form(60.0),
    threshold: int = Form(128),
    dark_on_light: bool = Form(True),
    output_px_per_mm: float = Form(10.0),
    fill_interior_mm: float = Form(3.0),
    color_weight: float = Form(0.0),
    edge_guided: bool = Form(False),
    edge_min: int = Form(20),
    depth_mode: bool = Form(False),
    depth_percentile: float = Form(95.0),
):
    path = TMPDIR / f"{session}.jpg"
    if not path.exists():
        raise HTTPException(404, "Session expired. Please re-upload.")

    img = _load_img(path.read_bytes())
    markers = detect_markers(img)
    if not markers:
        raise HTTPException(400, "No markers")

    H, mm_per_px = compute_homography(markers, marker_size_mm)
    warped, actual_ppmm, bounds = warp_image(img, H, mm_per_px, output_px_per_mm)

    ref_path = TMPDIR / f"{session}_ref.jpg"
    diff_img_b64 = None
    if depth_mode:
        mask, depth_vis = extract_silhouette_depth(warped, actual_ppmm, fill_interior_mm, depth_percentile)
        diff_img_b64 = _encode_jpg(_scale_for_preview(depth_vis))
    elif ref_path.exists():
        ref_img = _load_img(ref_path.read_bytes())
        ref_markers = detect_markers(ref_img)
        H_ref = compute_homography(ref_markers, marker_size_mm)[0] if ref_markers else H
        warped_ref = warp_to_bounds(ref_img, H_ref, bounds, actual_ppmm)
        mask, diff_vis = extract_silhouette_reference(
            warped, warped_ref, threshold, actual_ppmm, fill_interior_mm,
            edge_guided, edge_min)
        diff_img_b64 = _encode_jpg(_scale_for_preview(diff_vis))
    else:
        mask = extract_silhouette(warped, threshold, dark_on_light, actual_ppmm,
                                   fill_interior_mm, color_weight, edge_guided, edge_min)

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return JSONResponse({
        "mask_img":  _encode_jpg(_scale_for_preview(mask_bgr)),
        "diff_img":  diff_img_b64,
        "width_mm":  round(warped.shape[1] / actual_ppmm, 1),
        "height_mm": round(warped.shape[0] / actual_ppmm, 1),
    })


# ── SVG generation ──────────────────────────────────────────────────────────

@app.post("/generate-svg")
async def gen_svg(
    session: str = Form(...),
    marker_size_mm: float = Form(60.0),
    threshold: int = Form(128),
    dark_on_light: bool = Form(True),
    output_px_per_mm: float = Form(10.0),
    fill_interior_mm: float = Form(3.0),
    color_weight: float = Form(0.0),
    edge_guided: bool = Form(False),
    edge_min: int = Form(20),
    epsilon_mm: float = Form(0.5),
    margin_mm: float = Form(1.0),
    depth_mode: bool = Form(False),
    depth_percentile: float = Form(95.0),
    mask_override: Optional[str] = Form(None),
):
    path = TMPDIR / f"{session}.jpg"
    if not path.exists():
        raise HTTPException(404, "Session expired. Please re-upload.")

    raw = path.read_bytes()
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    markers = detect_markers(img)
    if not markers:
        raise HTTPException(400, "No markers detected")

    H, mm_per_px = compute_homography(markers, marker_size_mm)
    warped, actual_ppmm, bounds = warp_image(img, H, mm_per_px, output_px_per_mm)

    if mask_override:
        import base64 as _b64
        data_url = mask_override.split(",", 1)[-1]
        png_bytes = _b64.b64decode(data_url)
        png_arr   = np.frombuffer(png_bytes, np.uint8)
        mask_img  = cv2.imdecode(png_arr, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask_img, (warped.shape[1], warped.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
        _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    else:
        ref_path = TMPDIR / f"{session}_ref.jpg"
        if depth_mode:
            mask, _ = extract_silhouette_depth(warped, actual_ppmm, fill_interior_mm, depth_percentile)
        elif ref_path.exists():
            ref_img = _load_img(ref_path.read_bytes())
            ref_markers = detect_markers(ref_img)
            H_ref = compute_homography(ref_markers, marker_size_mm)[0] if ref_markers else H
            warped_ref = warp_to_bounds(ref_img, H_ref, bounds, actual_ppmm)
            mask, _ = extract_silhouette_reference(
                warped, warped_ref, threshold, actual_ppmm, fill_interior_mm,
                edge_guided, edge_min)
        else:
            mask = extract_silhouette(warped, threshold, dark_on_light, actual_ppmm,
                                      fill_interior_mm, color_weight, edge_guided, edge_min)

    svg = generate_svg(mask, actual_ppmm, epsilon_mm, margin_mm)

    return Response(
        content=svg,
        media_type="image/svg+xml",
        headers={"Content-Disposition": 'attachment; filename="shadow_board.svg"'},
    )
