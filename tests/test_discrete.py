from dataclasses import replace
import torch
from ascii_flux.contract import DEFAULT
from ascii_flux.discrete import DiscreteConfig,DiscreteGlyphFlow,reverse_probabilities


def test_categorical_reverse_matches_explicit_mixture_and_clean_endpoint():
    torch.manual_seed(1)
    p0 = torch.softmax(torch.randn(1,7,2,3),1)
    xt = torch.randint(7,(1,2,3))
    a,b = .3,.7
    result = reverse_probabilities(p0,xt,a,b)
    explicit = torch.zeros_like(p0)
    for row in range(2):
        for col in range(3):
            j = int(xt[0,row,col])
            for k in range(7):
                for i in range(7):
                    q_s = b*(i == k)+(1-b)/7
                    q_t_s = a/b*(i == j)+(1-a/b)/7
                    q_t_0 = a*(j == k)+(1-a)/7
                    explicit[0,i,row,col] += p0[0,k,row,col]*q_s*q_t_s/q_t_0
    torch.testing.assert_close(result,explicit)
    torch.testing.assert_close(reverse_probabilities(p0,xt,a,1.),p0)
    unchanged = reverse_probabilities(p0,xt,a,a).argmax(1)
    assert torch.equal(unchanged,xt)


@torch.inference_mode()
def test_discrete_state_remains_native_and_prompt_change_keeps_display():
    contract = replace(DEFAULT,cols=16,rows=8,frames=6)
    model = DiscreteGlyphFlow(contract,DiscreteConfig(width=8,condition_dim=32,emergence_steps=4,motion_steps=2))
    state,text = model.initial_state(),torch.zeros(1,768)
    for _ in range(12):
        state,_,_ = model.step(state,text)
        g,c = model.tokens(state)
        assert torch.isfinite(state).all() and state.abs().max() <= 255
        assert g.min() >= 0 and g.max() < 7
        assert (c[g == 0] == 255).all() and (c[g != 0] < 64).all()
    revised = model.recondition(state)
    torch.testing.assert_close(revised[:,1:3],state[:,1:3],rtol=0,atol=0)
