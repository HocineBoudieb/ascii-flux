from dataclasses import replace
import numpy as np
import torch
from ascii_flux.cascade import CascadeConfig, CascadeGlyphModel, cascade_loss
from ascii_flux.deform import deform_conv2d
from ascii_flux.storage import validate_tokens
from ascii_flux.structure import STRUCTURE, to_glyphs

SMALL = replace(STRUCTURE, rows=8, cols=16, frames=1)


def small_model():
    return CascadeGlyphModel(SMALL, CascadeConfig(dim=32, depth=2, heads=4, text_tokens=4, text_dim=12,
                                                  silhouette_steps=4, glyph_steps=3))


def target(batch=2):
    g = torch.zeros((batch, 8, 16), dtype=torch.long)
    g[:, 2:6, 4:12] = SMALL.glyphs.index("#")
    g[:, 2, 4:12] = SMALL.glyphs.index("-")
    c = torch.full_like(g, SMALL.empty_color)
    c[g != 0] = 7
    return g, c


def test_patch_layout_round_trips_and_mirror_swaps_strokes():
    model = small_model()
    g, c = target()
    torch.testing.assert_close(model.grid(model.patches(g)), g)
    slash = torch.zeros((1, 8, 16), dtype=torch.long)
    slash[0, 0, 0] = SMALL.glyphs.index("/")
    mg, _ = model.mirror(slash, slash)
    assert mg[0, 0, -1] == SMALL.glyphs.index("\\")


def test_cascade_learns_and_samples_valid_tokens():
    torch.manual_seed(0)
    model = small_model()
    g, c = target()
    text = torch.randn(2, 4, 12)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(40):
        loss, _ = cascade_loss(model, g, c, model.text_context(text))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0]*.5
    model.eval()
    glyphs, colors, trace = model.sample(text, return_trace=True)
    assert len(trace) == 7 and glyphs.shape == (2, 8, 16)
    validate_tokens(glyphs.numpy().astype(np.uint8), colors.numpy().astype(np.uint8),
                    np.tile(np.array([[0, 0, 1, 1]], np.float32), (2, 1)), replace(SMALL, frames=2))


def test_structure_converter_draws_outline_strokes():
    rgb = np.full((256, 256, 3), 200, np.uint8)
    mask = np.zeros((256, 256), bool)
    mask[60:200, 40:220] = True
    g, c = to_glyphs(rgb, mask, STRUCTURE.palette)
    strokes = {STRUCTURE.glyphs.index(x) for x in "|-_/\\"}
    assert g.shape == (32, 64) and (g != 0).any()
    assert np.isin(g, list(strokes)).sum() > 20
    assert np.all(c[g == 0] == STRUCTURE.empty_color) and np.all(c[g != 0] < len(STRUCTURE.palette))


def test_pure_torch_deform_conv_matches_reference():
    import pytest
    reference = pytest.importorskip("torchvision.ops").deform_conv2d
    torch.manual_seed(0)
    x, w, b = torch.rand(2, 4, 9, 11), torch.rand(6, 4, 3, 3), torch.rand(6)
    offset, mask = torch.randn(2, 18, 9, 11)*2, torch.rand(2, 9, 9, 11)
    torch.testing.assert_close(deform_conv2d(x, offset, w, b, 1, 1, 1, mask),
                               reference(x, offset, w, b, (1, 1), (1, 1), (1, 1), mask), atol=1e-4, rtol=1e-4)
