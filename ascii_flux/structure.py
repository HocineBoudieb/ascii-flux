"""Structure-aware glyph conversion for the offline data factory.

Density ramps fill a subject's interior and leave its outline to blurred
gradients; recognizable ASCII art draws the outline with oriented strokes
(see structure-based ASCII art, Xu, Zhang & Wong, SIGGRAPH 2010). Here the
silhouette boundary and strong interior edges receive oriented glyphs, while
the interior keeps a light tone ramp. Never used at inference.
"""
from dataclasses import replace
import cv2
import numpy as np
from .contract import Contract

# Ordered vocabulary: space first, density ramp, then oriented strokes.
STRUCTURE_GLYPHS = " .:+*#@-_|/\\()'"
STRUCTURE = replace(Contract(), version="ascii-v5-structure-rgb64-1", glyphs=STRUCTURE_GLYPHS,
                    glyph_version="ramp-strokes-1")
RAMP = STRUCTURE_GLYPHS.index("."), STRUCTURE_GLYPHS.index("@")


def segment_plain_background(rgb, tolerance=28.):
    """Subject mask for a teacher image prompted on a plain light background.

    Background is estimated from the border; the mask is the largest connected
    component that differs from it. Returns (mask, reason or None).
    """
    image = rgb.astype(np.float32)
    border = np.concatenate([image[:4].reshape(-1, 3), image[-4:].reshape(-1, 3),
                             image[:, :4].reshape(-1, 3), image[:, -4:].reshape(-1, 3)])
    background = np.median(border, 0)
    if np.mean(np.linalg.norm(border-background, axis=1) < 2*tolerance) < .6:
        return None, "background_not_plain"
    # Studio backdrops are smooth gradients rather than one color: model the
    # background as a heavily blurred image of the border-filled frame.
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    small = cv2.resize(lab, (16, 16), interpolation=cv2.INTER_AREA)
    known = np.zeros((16, 16), bool); known[[0, -1]] = True; known[:, [0, -1]] = True
    plane = np.where(known[..., None], small, np.median(small[known], 0))
    for _ in range(30):
        plane = np.where(known[..., None], small, cv2.blur(plane, (3, 3)))
    bg_lab = cv2.resize(plane, (w, h), interpolation=cv2.INTER_CUBIC)
    distance = np.linalg.norm(lab-bg_lab, axis=2)
    mask = (distance > tolerance).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count < 2:
        return None, "no_subject"
    largest = 1+int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = labels == largest
    # Fill holes enclosed by the subject (white belly, gaps between features).
    flood = (~mask).astype(np.uint8)
    cv2.floodFill(flood, None, (0, 0), 0)
    mask = mask | flood.astype(bool)
    h, w = mask.shape
    if mask[:2].any() or mask[-2:].any() or mask[:, :2].any() or mask[:, -2:].any():
        return mask, "subject_touches_border"
    if mask.mean() < .03:
        return mask, "subject_too_small"
    return mask, None


def _fit(mask, contract, margin):
    """Crop box with the subject's aspect ratio preserved inside the grid."""
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max()+1, xs.min(), xs.max()+1
    grid_aspect = contract.rows*contract.cell_height_width/contract.cols  # height/width in pixels
    h, w = (y1-y0)*(1+2*margin), (x1-x0)*(1+2*margin)
    if h/w > grid_aspect:
        w = h/grid_aspect
    else:
        h = w*grid_aspect
    cy, cx = (y0+y1)/2, (x0+x1)/2
    return cy-h/2, cx-w/2, h, w


def to_glyphs(rgb, mask, palette, contract=STRUCTURE, margin=.06, sub=8, edge_threshold=.5):
    """One RGB frame plus subject mask to (glyphs, colors) uint8 [R, C]."""
    rows, cols = contract.rows, contract.cols
    top, left, h, w = _fit(mask, contract, margin)
    sx_cell = max(2, int(round(sub/contract.cell_height_width)))
    hs, ws = rows*sub, cols*sx_cell
    # Sample the crop at sub-cell resolution; outside the frame is background.
    matrix = np.float32([[w/ws, 0, left], [0, h/hs, top]])
    inverse = cv2.invertAffineTransform(matrix)
    fine_mask = cv2.warpAffine(mask.astype(np.float32), inverse, (ws, hs), flags=cv2.INTER_AREA,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    fine_rgb = cv2.warpAffine(rgb, inverse, (ws, hs), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    sy, sx = hs//rows, ws//cols
    cell = lambda a: a.reshape(rows, sy, cols, sx, *a.shape[2:]).mean((1, 3))
    coverage = cell(fine_mask)
    # Silhouette boundary orientation from the mask gradient.
    blurred = cv2.GaussianBlur(fine_mask, (0, 0), sy/3)
    gx, gy = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    # Interior structure: strong luminance edges inside the subject only.
    gray = cv2.cvtColor(fine_rgb, cv2.COLOR_RGB2GRAY)
    canny = cv2.Canny(cv2.GaussianBlur(gray, (0, 0), 1.2), 60, 140).astype(np.float32)/255*(cv2.erode(fine_mask, np.ones((5, 5)))>.5)
    ix, iy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    boundary = cell(np.hypot(gx, gy) > 1e-3)*(coverage > .08)*(coverage < .92)
    interior_edges = cell(canny)
    gx_c, gy_c = cell(gx), cell(gy)
    ix_c, iy_c = cell(ix*canny), cell(iy*canny)
    luminance = cell(gray.astype(np.float32))/255
    glyphs = np.zeros((rows, cols), np.uint8)
    inside = coverage > .35
    # Interior: light ramp from tone, kept sparse so outlines dominate.
    tone = np.clip(1-luminance, 0, 1)*.55+.05
    glyphs[inside] = np.round(RAMP[0]+tone[inside]*(RAMP[1]-RAMP[0])).astype(np.uint8)
    index = {g: i for i, g in enumerate(contract.glyphs)}

    def oriented(dx, dy, bottom):
        # Gradient is normal to the edge; the stroke follows the tangent.
        angle = (np.degrees(np.arctan2(dy, dx))+180) % 180
        if angle < 22.5 or angle >= 157.5:
            return index["|"]
        if angle < 67.5:
            return index["/"]
        if angle < 112.5:
            return index["_"] if bottom else index["-"]
        return index["\\"]

    edge = (interior_edges > .22) & inside
    for r, c in zip(*np.nonzero(edge)):
        glyphs[r, c] = oriented(ix_c[r, c], iy_c[r, c]*contract.cell_height_width, False)
    outline = (boundary > edge_threshold*.2) & (coverage > .12)
    for r, c in zip(*np.nonzero(outline)):
        # gy < 0 means mask increases upward: this is the subject's lower edge.
        glyphs[r, c] = oriented(gx_c[r, c], gy_c[r, c]*contract.cell_height_width, gy_c[r, c] < 0)
    glyphs[coverage <= .12] = 0
    # Colors: mean subject color per cell, quantized to the palette.
    weights = fine_mask[..., None]
    colors_sum = cell(fine_rgb.astype(np.float32)*weights)
    mean = colors_sum/np.maximum(coverage[..., None], 1e-6)
    pal = np.asarray(palette, np.float32)
    # Brighten toward display on black, as in the v4.2 color rendering.
    mean = np.clip(mean*1.35+45, 0, 255)
    colors = np.argmin(((mean[..., None, :]-pal)**2).sum(-1), -1).astype(np.uint8)
    colors[glyphs == 0] = contract.empty_color
    return glyphs, colors
