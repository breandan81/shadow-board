import cv2
import numpy as np
import base64
import io
from PIL import Image

PAPER_SIZES = {
    "A4":     (210.0, 297.0),
    "Letter": (215.9, 279.4),
    "A3":     (297.0, 420.0),
}


def _marker_to_png_b64(aruco_dict, marker_id: int, size_px: int) -> str:
    img = np.zeros((size_px, size_px), dtype=np.uint8)
    try:
        img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size_px, img, 1)
    except AttributeError:
        img = cv2.aruco.drawMarker(aruco_dict, marker_id, size_px, img, 1)
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def generate_datum_svg(
    marker_size_mm: float = 60.0,
    paper: str = "A4",
    margin_mm: float = 15.0,
    gap_mm: float = 12.0,
) -> str:
    pw, ph = PAPER_SIZES.get(paper, PAPER_SIZES["A4"])
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    header_h = 22.0
    usable_w = pw - 2 * margin_mm
    usable_h = ph - margin_mm - header_h - margin_mm / 2

    cols = max(1, int((usable_w + gap_mm) / (marker_size_mm + gap_mm)))
    rows = max(1, int((usable_h + gap_mm) / (marker_size_mm + gap_mm + 6)))

    # 300 DPI equivalent pixel size
    size_px = max(64, int(marker_size_mm / 25.4 * 300))

    elements: list[str] = []
    mid = 0
    for row in range(rows):
        for col in range(cols):
            if mid >= 50:
                break
            x = margin_mm + col * (marker_size_mm + gap_mm)
            y = margin_mm + header_h + row * (marker_size_mm + gap_mm + 6)
            b64 = _marker_to_png_b64(aruco_dict, mid, size_px)
            elements.append(
                f'  <image x="{x}" y="{y}" width="{marker_size_mm}" height="{marker_size_mm}" '
                f'href="data:image/png;base64,{b64}" image-rendering="pixelated"/>'
            )
            # White background behind marker so it prints cleanly
            elements.append(
                f'  <rect x="{x}" y="{y}" width="{marker_size_mm}" height="{marker_size_mm}" '
                f'fill="none" stroke="#ccc" stroke-width="0.2"/>'
            )
            # ID label
            lx = x + marker_size_mm / 2
            ly = y + marker_size_mm + 4
            elements.append(
                f'  <text x="{lx}" y="{ly}" text-anchor="middle" '
                f'font-size="3.5" font-family="sans-serif">ID {mid}</text>'
            )
            mid += 1

    # Corner crosshairs (registration marks)
    cs = 5.0
    lw = 0.3
    crosses = []
    for cx, cy in [
        (margin_mm, margin_mm),
        (pw - margin_mm, margin_mm),
        (margin_mm, ph - margin_mm),
        (pw - margin_mm, ph - margin_mm),
    ]:
        crosses += [
            f'<line x1="{cx-cs}" y1="{cy}" x2="{cx+cs}" y2="{cy}" stroke="black" stroke-width="{lw}"/>',
            f'<line x1="{cx}" y1="{cy-cs}" x2="{cx}" y2="{cy+cs}" stroke="black" stroke-width="{lw}"/>',
            f'<circle cx="{cx}" cy="{cy}" r="1" fill="none" stroke="black" stroke-width="{lw}"/>',
        ]

    crosses_svg = "\n  ".join(crosses)
    markers_svg = "\n".join(elements)

    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
     width="{pw}mm" height="{ph}mm" viewBox="0 0 {pw} {ph}">
  <rect width="{pw}" height="{ph}" fill="white"/>
  <rect x="3" y="3" width="{pw-6}" height="{ph-6}"
        fill="none" stroke="#aaa" stroke-width="0.4" stroke-dasharray="3,2"/>

  <!-- Header -->
  <text x="{pw/2}" y="{margin_mm+8}" text-anchor="middle"
        font-size="6" font-family="sans-serif" font-weight="bold">
    Shadow Board Datum Sheet
  </text>
  <text x="{pw/2}" y="{margin_mm+14}" text-anchor="middle"
        font-size="4" font-family="sans-serif" fill="#333">
    PRINT AT 100% — DO NOT SCALE TO FIT
  </text>
  <text x="{pw/2}" y="{margin_mm+19}" text-anchor="middle"
        font-size="3.5" font-family="sans-serif" fill="#555">
    Marker size: {marker_size_mm:.0f} mm × {marker_size_mm:.0f} mm  |  Dictionary: DICT_4X4_50  |  {paper}
  </text>

  <!-- Registration marks -->
  {crosses_svg}

  <!-- ArUco markers -->
{markers_svg}
</svg>'''
