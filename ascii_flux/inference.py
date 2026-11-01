"""Load native model experiments without changing the NCA training identity."""
import torch
from .contract import Contract,TEXT_SPEC


def load_inference(path,device="cpu",allow_development=False):
    checkpoint = torch.load(path,map_location="cpu",weights_only=True)
    if checkpoint["format"] == "ascii-flux-v5-1":
        from .training import load_inference as load_nca
        return load_nca(path,device,allow_development)
    if checkpoint["format"] not in ("ascii-flux-v5-masked-1","ascii-flux-v5-causal-1","ascii-flux-v5-discrete-1"):
        raise ValueError("Unknown checkpoint format")
    provenance = checkpoint["provenance"]
    if provenance["text_spec"] != TEXT_SPEC:
        raise ValueError("Text representation mismatch")
    if (provenance["allow_fixture"] or provenance["allow_unreviewed"] or provenance.get("development_only")) and not allow_development:
        raise ValueError("Development checkpoint: explicitly enable development mode")
    contract = Contract.from_dict(checkpoint["contract"])
    if checkpoint["format"] == "ascii-flux-v5-discrete-1":
        from .discrete import DiscreteConfig,DiscreteGlyphFlow
        model = DiscreteGlyphFlow(contract,DiscreteConfig(**checkpoint["model_config"]))
    elif checkpoint["format"] == "ascii-flux-v5-masked-1":
        from .masked import MaskedConfig,MaskedGlyphFlow
        model = MaskedGlyphFlow(contract,MaskedConfig(**checkpoint["model_config"]))
    else:
        from .autoregressive import CausalConfig,CausalGlyphFlow
        model = CausalGlyphFlow(contract,CausalConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(),provenance
