# Shadow Board Generator

Turn a photo of a tool into a laser-cuttable SVG shadow board outline — automatically corrected for perspective and scale using ArUco marker fiducials.

![Shadow board workflow: photo → SVG](https://raw.githubusercontent.com/breandan/shadow-board/main/docs/workflow.png)

## How it works

1. Print the datum sheet (a page of ArUco markers) and cut out as many as you need.
2. Scatter them flat on your shadow board surface around the tool — no fixed positions required.
3. Photograph the tool from above.
4. Upload the photo — the app detects the markers, corrects perspective, extracts the tool silhouette, and generates an SVG you can send straight to a laser cutter.

Scale and perspective are calibrated from the markers, so the SVG dimensions are accurate in millimetres regardless of camera angle or distance.

## Requirements

- Python 3.9+
- A camera or phone (any resolution works)
- A printer (for the datum sheet)

GPU is optional but recommended if you use depth mode (see below). The depth model (~100 MB) downloads automatically on first use.

## Installation

```bash
git clone https://github.com/breandan/shadow-board.git
cd shadow-board
./run.sh
```

`run.sh` creates a Python virtual environment, installs all dependencies, and starts the server. Open [http://localhost:8000](http://localhost:8000) in your browser.

To start the server again later without reinstalling:

```bash
source .venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## Usage

### Step 1 — Print the datum sheet

Click **Download Datum Sheet** on the Setup tab and print it at 100% scale (no scaling to fit). The sheet contains six distinct ArUco markers. You can use the whole sheet or cut the markers apart and scatter them — placement does not need to be precise or follow any pattern. Print on plain white paper and lay them flat on your shadow board.

You can adjust the marker size (default 60 mm) if you need a larger or smaller sheet. The same size must be entered in the app when processing photos.

### Step 2 — Take a photo

Place the tool on the datum sheet and photograph it from directly above. Tips for best results:

- At least one marker must be visible; more markers give better perspective correction. Spread them around the tool for best results.
- Avoid strong shadows or reflections on the markers.
- The sheet should be flat — curled edges will distort the homography.
- Fill the frame with the board as much as possible; more pixels = cleaner outline.

### Step 3 — Process the photo

On the **Process** tab, upload the photo. The app detects the markers, warps the image to a top-down view, and extracts the tool silhouette. You will see:

- **Detection**: markers highlighted on the original photo.
- **Warped**: perspective-corrected top-down view.
- **Mask**: the extracted tool silhouette (white = tool, black = background).

Use the controls on the right to tune the result:

| Setting | What it does |
|---|---|
| **Threshold** | Brightness cutoff for separating tool from background (standard mode). |
| **Dark on light** | Toggle if the tool is darker than the board surface. |
| **Fill interior (mm)** | Fills holes inside the tool outline (useful for open rings, handles with gaps). |
| **Margin (mm)** | Expands the outline outward — adds clearance so the tool fits easily into the pocket. |
| **Simplification (mm)** | RDP tolerance for smoothing the outline path. Higher = fewer points, less detail. |
| **Smooth outline (mm)** | Morphological closing radius. Rounds concave corners and fills small notches while always staying outside the tool. |
| **Output resolution** | Pixels per mm in the internal warped image. Higher = more detail, slower. |

### Silhouette extraction modes

**Standard (default)** — Thresholds the warped image by brightness. Works well for tools with good contrast against the board surface.

**Reference photo** — Upload a second photo of the empty board (no tool) alongside the tool photo. The app subtracts the two images to isolate the tool. More robust than brightness thresholding for textured or patterned boards.

**Depth model** — Uses [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) (monocular depth estimation) to identify the closest object in the scene. Because it uses geometry rather than colour, it works for metallic, chrome, or same-coloured tools that fool brightness-based methods. The **Sensitivity** slider controls which depth percentile is considered "tool" (default 95th — the top 5% closest pixels). Works best for tools with significant height (>~20 mm); very flat objects may not have enough depth signal.

### Step 4 — Edit the mask (optional)

After analysis the mask becomes an editable canvas. Use the **Erase** and **Draw** brushes to correct any false positives (stray marks on the board) or false negatives (missing parts of the tool). Adjust brush size with the slider. Use **Undo** to step back.

Edits are sent to the backend when you generate the SVG. If you have not painted, the SVG is generated fresh from the original image at full resolution.

### Step 5 — Generate and download the SVG

Click **Generate SVG**. A preview appears inline; click **Download SVG** to save the file.

The SVG uses real-world millimetre units and contains a single filled path per tool outline. Import it directly into LightBurn, RDWorks, Inkscape, or any other laser cutter software.

## Validation

The `test-objects/` directory contains OpenSCAD models for four test objects (rectangle, disk, L-shape, ring) with known dimensions. Print these in a high-contrast colour, photograph them on the datum sheet, and run:

```bash
python test-objects/validate.py
```

This processes each photo through the pipeline and compares the SVG bounding boxes against the ground truth dimensions, reporting error in mm.

## Troubleshooting

**"No ArUco markers detected"** — No markers are visible or the image is too dark/blurry. Make sure at least one marker is in frame, flat, and well-lit.

**Outline includes the board edge or other objects** — Lower the threshold (standard mode) or increase the depth sensitivity (depth mode). Use the mask paint tool to erase false positives.

**Outline is missing parts of the tool** — Raise the threshold, or switch to depth mode if the tool colour matches the board. Use the mask paint tool to fill in missing areas.

**Tool is metallic or same colour as the board** — Use depth mode. The tool needs to be at least ~20 mm tall for reliable depth signal at normal camera distances.

**SVG pockets are too tight / too loose** — Adjust the **Margin (mm)** setting. A positive margin expands the pocket outward; use 1–2 mm as a starting point for most tools.

## License

GPLv3
