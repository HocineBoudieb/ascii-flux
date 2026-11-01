"""The complete representation is serialized in datasets and checkpoints."""
from dataclasses import asdict, dataclass
import hashlib
import json
import unicodedata


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def normalize_text(text):
    return unicodedata.normalize("NFC", str(text)).lower().strip()


@dataclass(frozen=True)
class Contract:
    version: str = "ascii-v5-ramp-rgb64-1"
    cols: int = 64
    rows: int = 32
    cell_height_width: float = 2.0
    glyphs: str = " .:+*#@"
    glyph_version: str = "density-1"
    palette_version: str = "rgb-cube-4-1"
    palette: tuple = tuple((r, g, b) for r in (0, 85, 170, 255)
                           for g in (0, 85, 170, 255) for b in (0, 85, 170, 255))
    empty_color: int = 255
    fps: float = 12.0
    frames: int = 48

    def __post_init__(self):
        if min(self.cols, self.rows, self.frames) <= 0 or self.fps <= 0 or self.cell_height_width <= 0:
            raise ValueError("Invalid grid, frame count or cadence")
        if not self.glyphs.startswith(" ") or len(set(self.glyphs)) != len(self.glyphs) or len(self.glyphs) > 256:
            raise ValueError("Ordered unique vocabulary must start with space")
        if not 1 <= len(self.palette) <= 255 or self.empty_color != 255:
            raise ValueError("Palette uses 0..254; 255 means no color")
        if any(len(c) != 3 or any(not 0 <= x <= 255 for x in c) for c in self.palette):
            raise ValueError("Invalid RGB palette")

    def to_dict(self):
        return json.loads(canonical(asdict(self)))

    @classmethod
    def from_dict(cls, value):
        v = dict(value)
        v["palette"] = tuple(tuple(c) for c in v["palette"])
        return cls(**v)

    @property
    def hash(self):
        return digest(self.to_dict())

    def require(self, other):
        if self.hash != Contract.from_dict(other).hash:
            raise ValueError("Incompatible dataset/checkpoint representation; create a new version and retrain")


DEFAULT = Contract()
TEXT_SPEC = {
    "model": "google/siglip2-base-patch16-224",
    "revision": "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
    "normalization": "NFC/lower/strip",
    "max_length": 64,
    "embedding": "Siglip2TextModel.pooler_output/l2",
    "dimension": 768,
}

