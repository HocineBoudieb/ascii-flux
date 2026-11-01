from dataclasses import replace
import numpy as np
import torch
from ascii_flux.contract import DEFAULT
from ascii_flux.masked import MaskedConfig,MaskedGlyphFlow,masked_loss
from ascii_flux.storage import validate_tokens


def small_model():
    return MaskedGlyphFlow(replace(DEFAULT,rows=8,cols=16,frames=6),MaskedConfig(width=8,condition_dim=32))


def test_masked_learning_changes_weights_and_ignores_empty_colors():
    torch.manual_seed(42)
    model = small_model()
    g = torch.zeros((2,8,16),dtype=torch.long);g[:,2:6,5:11] = 3
    c = torch.full_like(g,255);c[g != 0] = 12
    mask = torch.rand_like(g.float()) < .7
    pg = torch.full_like(g,model.mask_glyph);pc = torch.full_like(g,model.mask_color)
    args = (g.masked_fill(mask,model.mask_glyph),c.masked_fill(mask,model.mask_color),pg,pc,
            torch.randn(2,768),mask.float().mean((1,2)),torch.zeros(2,dtype=torch.long))
    optimizer = torch.optim.AdamW(model.parameters(),lr=.003)
    losses = []
    for step in range(12):
        gl,cl = model(*args)
        assert gl.shape == (2,7,8,16) and cl.shape == (2,64,8,16)
        cl.retain_grad();loss = masked_loss(gl,cl,g,c,mask)
        optimizer.zero_grad();loss.backward()
        assert torch.isfinite(loss) and not torch.any(cl.grad.permute(0,2,3,1)[g == 0])
        optimizer.step();losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]


@torch.inference_mode()
def test_masked_sampler_only_emits_valid_native_tokens_and_keeps_grid_on_change():
    model = small_model().eval()
    state = model.initial_state()
    text = torch.zeros(1,768)
    for _ in range(20):
        state,_,_ = model.step(state,text)
        g,c = model.tokens(state)
        assert torch.isfinite(state).all() and state.abs().max() <= 255
        boxes = np.tile(np.array([0,0,8,8],np.float32),(6,1))
        validate_tokens(np.repeat(g.cpu().numpy().astype(np.uint8),6,axis=0),
                        np.repeat(c.cpu().numpy().astype(np.uint8),6,axis=0),boxes,model.contract)
    revised = model.recondition(state)
    torch.testing.assert_close(revised[:,:2],state[:,:2],rtol=0,atol=0)
    assert not torch.equal(revised[:,2:],state[:,2:])
