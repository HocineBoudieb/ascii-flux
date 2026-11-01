"""Offline video/mask -> glyphs. Never used by the inference runtime.

Tone, coverage, density gamma, hysteresis and HSV saturation are adapted from
the frozen v4.2 notebook. No procedural halo or synthetic volume is baked in.
"""
from dataclasses import asdict, dataclass
import hashlib
import platform
from importlib.metadata import version
from pathlib import Path
import cv2
import numpy as np
from .contract import digest


@dataclass(frozen=True)
class ConverterConfig:
    margin: float = 0.15
    smoothing: float = 0.25
    gamma: float = 0.8
    hysteresis: float = 0.7
    saturation: float = 1.15
    min_coverage: float = 0.12

    def __post_init__(self):
        if self.margin < 0 or not 0 < self.smoothing <= 1 or self.gamma <= 0 or not 0 <= self.min_coverage < 1:
            raise ValueError("Invalid converter configuration")
        if self.hysteresis < 0 or self.saturation < 0:
            raise ValueError("Invalid tone settings")


def converter_identity(config):
    # Includes adapter, schema and metrics: implementation changes invalidate provenance.
    h = hashlib.sha256()
    for name in ("offline.py", "factory.py", "storage.py", "contract.py"):
        h.update(name.encode())
        h.update(Path(__file__).with_name(name).read_bytes())
    return {"version": "offline-v5-1", "code_sha256": h.hexdigest(),
            "config": asdict(config), "config_sha256": digest(asdict(config)),
            "environment": {"python": platform.python_version(), "numpy": np.__version__,
                            "opencv": cv2.__version__, "pillow": version("Pillow")}}


def tracked_boxes(masks, contract, config):
    measurements = []
    for mask in masks:
        y, x = np.nonzero(mask)
        measurements.append(None if len(x) == 0 else
                            np.array([x.min(), y.min(), x.max() + 1, y.max() + 1], np.float32))
    valid = [i for i, b in enumerate(measurements) if b is not None]
    if not valid:
        raise ValueError("no_subject_mask")
    measurements = np.stack([measurements[min(valid, key=lambda j: abs(j-i))] for i in range(len(masks))])
    centers = (measurements[:, :2] + measurements[:, 2:]) / 2
    # One clip-wide scale avoids per-frame crop breathing. Padding may lie outside
    # the image and is rendered black, preserving the actual grid aspect ratio.
    aspect = contract.cols / (contract.rows * contract.cell_height_width)
    extents = (measurements[:, 2:] - measurements[:, :2]).max(0) * (1 + 2 * config.margin)
    width = max(float(extents[0]), float(extents[1]) * aspect, 1)
    height = width / aspect
    for i in range(1, len(centers)):
        centers[i] = config.smoothing * centers[i] + (1-config.smoothing) * centers[i-1]
    # Include the subject even when smoothing lags a fast translation.
    centers = np.maximum(centers, measurements[:, 2:] - [width/2, height/2])
    centers = np.minimum(centers, measurements[:, :2] + [width/2, height/2])
    return np.concatenate([centers - [width/2, height/2], centers + [width/2, height/2]], 1).astype(np.float32)


def _crop(array, box, width, height, nearest=False):
    xs = box[0] + (np.arange(width, dtype=np.float32) + .5) * (box[2]-box[0])/width - .5
    ys = box[1] + (np.arange(height, dtype=np.float32) + .5) * (box[3]-box[1])/height - .5
    mx, my = np.meshgrid(xs, ys)
    return cv2.remap(array, mx, my, cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _luminance(rgb):
    rgb = rgb.astype(np.float32) / 255
    return .55 * (rgb @ np.array([.2126, .7152, .0722], np.float32)) + .45 * rgb.max(-1)


def _change(current, previous, valid=None):
    active = (current != 0) | (previous != 0)
    if valid is not None:
        active &= valid
    return float(((current != previous) & active).sum() / max(1, active.sum()))


def convert_clip(frames, masks, contract, config=None):
    config = config or ConverterConfig()
    if len(frames) != contract.frames or len(masks) != contract.frames:
        raise ValueError("wrong_clip_length")
    if any(f.shape[:2] != m.shape or f.shape[-1] != 3 for f, m in zip(frames, masks)):
        raise ValueError("frame_mask_shape_mismatch")
    boxes = tracked_boxes(masks, contract, config)
    width, height = contract.cols * 4, contract.rows * 4
    rgbs, alphas, grays = [], [], []
    for frame, mask, box in zip(frames, masks, boxes):
        alpha = _crop(mask.astype(np.float32), box, width, height)
        premultiplied = _crop(frame.astype(np.float32) * mask[..., None], box, width, height)
        rgb = premultiplied / np.maximum(alpha[..., None], 1e-6)
        rgbs.append(rgb)
        alphas.append(alpha)
        grays.append(cv2.cvtColor(np.uint8(np.clip(premultiplied, 0, 255)), cv2.COLOR_RGB2GRAY))
    # Fixed clip-wide percentiles and CDF avoid pumping the contrast every frame.
    luminances = [_luminance(rgb) for rgb in rgbs]
    samples = np.concatenate([l[a > .5] for l, a in zip(luminances, alphas)])
    if len(samples) == 0:
        raise ValueError("subject_lost_in_crop")
    lo, hi = np.percentile(samples, [2, 98])
    reference = np.sort(np.clip((samples-lo)/max(hi-lo, .001), 0, 1))
    top = len(contract.glyphs)-1
    glyphs, colors, raw, compensated, camera, subject_sizes = [], [], [], [], [], []
    previous_level = None
    grid_y, grid_x = np.mgrid[:contract.rows, :contract.cols].astype(np.float32)
    palette = np.array(contract.palette, dtype=np.float32)
    for i, (rgb, alpha, lum) in enumerate(zip(rgbs, alphas, luminances)):
        coverage = cv2.resize(alpha, (contract.cols, contract.rows), interpolation=cv2.INTER_AREA)
        safe = np.maximum(coverage, 1e-6)
        base = np.clip((lum-lo)/max(hi-lo, .001), 0, 1)
        tone = .5*base + .5*np.searchsorted(reference, base)/len(reference)
        tone_cell = cv2.resize((tone*alpha).astype(np.float32), (contract.cols, contract.rows), interpolation=cv2.INTER_AREA)/safe
        target = np.clip(tone_cell, 0, 1)**config.gamma * np.sqrt(np.minimum(1, coverage/.6)) * top
        level = np.rint(target).clip(0, top).astype(np.uint8)
        transported, valid = None, None
        if previous_level is not None:
            # Backward optical flow on the already tracked crops, in pixel units.
            flow = cv2.calcOpticalFlowFarneback(grays[i], grays[i-1], None, .5, 3, 15, 3, 5, 1.2, 0)
            small = cv2.resize(flow, (contract.cols, contract.rows), interpolation=cv2.INTER_AREA)/4
            mx, my = grid_x + small[..., 0], grid_y + small[..., 1]
            valid = (mx >= 0) & (my >= 0) & (mx <= contract.cols-1) & (my <= contract.rows-1)
            transported = cv2.remap(previous_level, mx, my, cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            level = np.where(valid & (np.abs(target-transported) < config.hysteresis), transported, level).astype(np.uint8)
        level[coverage >= .35] = np.maximum(level[coverage >= .35], 1)
        level[coverage < config.min_coverage] = 0
        mean_rgb = cv2.resize((rgb*alpha[..., None]).astype(np.float32), (contract.cols, contract.rows), interpolation=cv2.INTER_AREA)/safe[..., None]
        hsv = cv2.cvtColor(np.clip(mean_rgb/255, 0, 1), cv2.COLOR_RGB2HSV)
        hsv[..., 1] = np.clip(hsv[..., 1]*config.saturation, 0, 1)
        hsv[..., 2] = .42 + .58*level/top
        shown = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)*255
        indices = ((shown[..., None, :]-palette)**2).sum(-1).argmin(-1).astype(np.uint8)
        indices[level == 0] = contract.empty_color
        if previous_level is not None:
            raw.append(_change(level, previous_level))
            compensated.append(_change(level, transported, valid))
            # Background translation is a camera-motion proxy, not subject motion.
            prev_gray = cv2.cvtColor(frames[i-1], cv2.COLOR_RGB2GRAY)
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            scale = min(1., 320/max(prev_gray.shape))
            size = (max(8, round(prev_gray.shape[1]*scale)), max(8, round(prev_gray.shape[0]*scale)))
            bg = cv2.resize((~(masks[i-1].astype(bool) | masks[i].astype(bool))).astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
            flow = cv2.calcOpticalFlowFarneback(cv2.resize(prev_gray, size), cv2.resize(curr_gray, size), None, .5, 3, 15, 3, 5, 1.2, 0)
            texture = cv2.resize(prev_gray, size)
            bg &= (np.abs(cv2.Laplacian(texture, cv2.CV_32F)) > 2)
            if bg.sum() >= 32:
                translation = np.median(flow[bg], axis=0)/scale
                crop_translation = (boxes[i, :2]+boxes[i, 2:]-boxes[i-1, :2]-boxes[i-1, 2:])/2
                camera.append(float(np.linalg.norm(translation-crop_translation)))
        subject_sizes.append(int((coverage >= config.min_coverage).sum()))
        glyphs.append(level)
        colors.append(indices)
        previous_level = level
    glyphs, colors = np.stack(glyphs), np.stack(colors)
    metrics = {
        "occupancy": float((glyphs != 0).mean()),
        "subject_cells_mean": float(np.mean(subject_sizes)),
        "subject_cells_min": min(subject_sizes),
        "glyph_change_raw_active": float(np.mean(raw)) if raw else 0.,
        "glyph_change_compensated_active": float(np.mean(compensated)) if compensated else 0.,
        "glyph_changes_per_second": float(np.count_nonzero(glyphs[1:] != glyphs[:-1])/((len(glyphs)-1)/contract.fps)),
        "mask_fraction": float(np.mean([np.any(m) for m in masks])),
        "residual_camera_motion_pixels_proxy": float(np.mean(camera)) if camera else None,
        "camera_proxy_measured_pairs": len(camera),
        "unique_grid_count": len({g.tobytes() for g in glyphs}),
        "distinct_grid_updates_per_second": float(np.any(glyphs[1:] != glyphs[:-1], axis=(1, 2)).sum()/((len(glyphs)-1)/contract.fps)),
        "seam_change_active": _change(glyphs[0], glyphs[-1]),
        "seam_scope": "diagnostic only: natural clips are not periodic",
    }
    return glyphs, colors, boxes, metrics


def quality_rejections(metrics, thresholds):
    reasons = []
    for setting, metric, minimum in (
        ("min_occupancy", "occupancy", True), ("max_occupancy", "occupancy", False),
        ("min_subject_cells", "subject_cells_min", True),
        ("max_compensated_change", "glyph_change_compensated_active", False),
        ("min_mask_fraction", "mask_fraction", True),
        ("max_residual_camera_motion", "residual_camera_motion_pixels_proxy", False),
    ):
        bound = thresholds.get(setting)
        if bound is not None:
            value = metrics[metric]
            if value is None:
                reasons.append(setting + ":unmeasured")
            elif (value < bound if minimum else value > bound):
                reasons.append(setting)
    return reasons

