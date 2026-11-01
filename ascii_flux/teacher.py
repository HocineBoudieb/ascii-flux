"""Offline image teacher for the shape corpus. Never imported at inference.

A text-to-image model draws whole subjects for many words; a salient-object
segmenter cuts them out; ``structure.to_glyphs`` turns them into native grids.
Every record produced from it is tagged ``synthetic: teacher-t2i`` with the
exact models, revisions, licences and prompt. The v5 model learns from the
resulting glyph grids only.
"""
import numpy as np

GENERATOR = {"model": "stabilityai/sd-turbo", "decoder": "madebyollin/taesd",
             "license": "Stability AI non-commercial research community license (sd-turbo); MIT (taesd)",
             "steps": 2, "guidance": 0.0, "resolution": 512}
SEGMENTER = {"model": "ZhengPeng7/BiRefNet_lite", "revision": "aa62cd87eafb9cc43056d08ef3615a14628b831d",
             "license": "MIT", "resolution": 512, "threshold": .5}
TEMPLATES = ("a toy figurine of a {word}, full body, standing on a white table, white background, product photo",
             "a simple flat illustration of a {word}, full body, side view, white background")


class Teacher:
    def __init__(self, device="cuda"):
        import torch
        from diffusers import AutoPipelineForText2Image, AutoencoderTiny
        from huggingface_hub import model_info
        from transformers import AutoModelForImageSegmentation
        from .deform import install
        install()
        self.torch, self.device = torch, device
        self.generator_revision = model_info(GENERATOR["model"]).sha
        self.decoder_revision = model_info(GENERATOR["decoder"]).sha
        pipe = AutoPipelineForText2Image.from_pretrained(GENERATOR["model"], revision=self.generator_revision,
                                                         torch_dtype=torch.float16)
        pipe.vae = AutoencoderTiny.from_pretrained(GENERATOR["decoder"], revision=self.decoder_revision,
                                                   torch_dtype=torch.float16)
        self.pipe = pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)
        self.segmenter = AutoModelForImageSegmentation.from_pretrained(
            SEGMENTER["model"], revision=SEGMENTER["revision"], trust_remote_code=True).eval().to(device).half()
        self.mean = torch.tensor([.485, .456, .406], device=device)[:, None, None]
        self.std = torch.tensor([.229, .224, .225], device=device)[:, None, None]

    def provenance(self):
        return {"generator": GENERATOR | {"revision": self.generator_revision, "decoder_revision": self.decoder_revision},
                "segmenter": SEGMENTER, "templates": list(TEMPLATES), "synthetic": "teacher-t2i"}

    def draw(self, prompts, seeds):
        torch = self.torch
        generators = [torch.Generator(self.device).manual_seed(int(s)) for s in seeds]
        images = self.pipe(list(prompts), num_inference_steps=GENERATOR["steps"], guidance_scale=GENERATOR["guidance"],
                           generator=generators, height=GENERATOR["resolution"], width=GENERATOR["resolution"]).images
        return [np.asarray(image.convert("RGB")) for image in images]

    def masks(self, images):
        torch = self.torch
        with torch.inference_mode():
            x = torch.stack([torch.from_numpy(i).to(self.device).permute(2, 0, 1).float()/255 for i in images])
            x = torch.nn.functional.interpolate(x, size=(SEGMENTER["resolution"],)*2, mode="bilinear", align_corners=False)
            probability = self.segmenter(((x-self.mean)/self.std).half())[-1].sigmoid().float()
            probability = torch.nn.functional.interpolate(probability, size=images[0].shape[:2], mode="bilinear", align_corners=False)
        return [p[0].cpu().numpy() > SEGMENTER["threshold"] for p in probability]


def check_mask(mask):
    """Reason to reject a teacher cut-out, or None."""
    import cv2
    if mask.mean() < .02:
        return "subject_too_small"
    if mask[:3].any() or mask[-3:].any() or mask[:, :3].any() or mask[:, -3:].any():
        return "subject_touches_border"
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    areas = np.sort(stats[1:, cv2.CC_STAT_AREA])[::-1]
    if count > 2 and areas[1] > .15*areas[0]:
        return "several_subjects"
    return None
