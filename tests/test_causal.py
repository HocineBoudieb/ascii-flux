from dataclasses import replace
import pytest
import torch
from ascii_flux.autoregressive import CausalConfig,CausalGlyphFlow,causal_loss
from ascii_flux.contract import DEFAULT


def model():
    return CausalGlyphFlow(replace(DEFAULT,rows=8,cols=16,frames=6),
                          CausalConfig(width=32,heads=4,layers=2,patches_per_step=2))


def test_patch_order_roundtrip_and_causal_prediction_has_no_future_target():
    torch.manual_seed(1)
    m = model().eval()
    g = torch.randint(7,(2,8,16));c = torch.randint(64,g.shape)
    assert torch.equal(m.unpatch(m.patches(g)),g)
    pg = torch.full_like(g,7);pc = torch.full_like(g,65)
    text,mode = torch.randn(2,768),torch.zeros(2,dtype=torch.long)
    a,b = m(g,c,pg,pc,text,mode)
    patches_g,patches_c = m.patches(g).clone(),m.patches(c).clone()
    patches_g[:,3:] = (patches_g[:,3:]+1)%7
    patches_c[:,3:] = (patches_c[:,3:]+1)%64
    changed_g,changed_c = m.unpatch(patches_g),m.unpatch(patches_c)
    x,y = m(changed_g,changed_c,pg,pc,text,mode)
    torch.testing.assert_close(a[:,:4],x[:,:4],atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(b[:,:4],y[:,:4],atol=1e-6,rtol=1e-5)
    loss = causal_loss(m,a,b,g,c)
    loss.backward()
    assert torch.isfinite(loss) and m.glyph_head.weight.grad.abs().sum() > 0


@torch.inference_mode()
def test_causal_sampling_keeps_valid_native_state_through_cycles_and_reconditioning():
    m = model().eval()
    state,text = m.initial_state(),torch.zeros(1,768)
    for _ in range(12):
        state,_,_ = m.step(state,text)
        g,c = m.tokens(state)
        assert g.min() >= 0 and g.max() < 7
        assert (c[g == 0] == 255).all() and (c[g != 0] < 64).all()
        assert torch.isfinite(state).all()
    revised = m.recondition(state)
    torch.testing.assert_close(revised[:,:2],state[:,:2],rtol=0,atol=0)
    assert (revised[:,4] == 0).all() and (revised[:,5] == 2).all()


@pytest.mark.parametrize('kind',['causal','masked','discrete'])
def test_experimental_checkpoint_requires_opt_in_and_roundtrips_native_weights(tmp_path,kind):
    from ascii_flux.contract import TEXT_SPEC
    from ascii_flux.inference import load_inference
    from ascii_flux.evaluation import export_onnx
    m = model().eval()
    if kind == 'masked':
        from ascii_flux.masked import MaskedConfig,MaskedGlyphFlow
        m = MaskedGlyphFlow(m.contract,MaskedConfig(width=8,condition_dim=32)).eval()
    elif kind == 'discrete':
        from ascii_flux.discrete import DiscreteConfig,DiscreteGlyphFlow
        m = DiscreteGlyphFlow(m.contract,DiscreteConfig(width=8,condition_dim=32)).eval()
    path = tmp_path/'causal.pt'
    torch.save({'format':f'ascii-flux-v5-{kind}-1','contract':m.contract.to_dict(),
                'model_config':m.config_dict(),'model':m.state_dict(),
                'provenance':{'text_spec':TEXT_SPEC,'allow_fixture':True,'allow_unreviewed':True,
                              'synthetic_encoder':True,'development_only':True,'recognition_validated':False}},path)
    with pytest.raises(ValueError,match='Development'):
        load_inference(path)
    actual,provenance = load_inference(path,allow_development=True)
    assert not provenance['recognition_validated']
    for key,value in m.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[key],value,rtol=0,atol=0)
    with pytest.raises(ValueError,match='no validated recurrent ONNX export'):
        export_onnx(actual,tmp_path/'not-an-export.onnx')
