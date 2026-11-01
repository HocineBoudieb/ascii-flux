"""Compact standalone review player displaying the stored glyph/color tokens."""
import base64
import gzip
import hashlib
import json
from pathlib import Path
from .contract import Contract, digest
from .storage import atomic_file, read_clip


def load_review_manifest(dataset):
    """Recover a completed dataset from disk, independently of notebook globals."""
    path = Path(dataset)
    if path.is_dir() or path.suffix != ".json":
        path = path/"manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Aucun corpus terminé à cet emplacement : {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    content = dict(value)
    fingerprint = content.pop("fingerprint", None)
    if fingerprint != digest(content):
        raise ValueError("Empreinte du manifest incorrecte ; revue non générée")
    contract = Contract.from_dict(value["contract"])
    if contract.hash != value["contract_hash"]:
        raise ValueError("Contrat du corpus incohérent")
    root = path.parent.resolve()
    for name, expected in value["shards"].items():
        shard = (root/name).resolve()
        if not shard.is_relative_to(root):
            raise ValueError("Chemin de shard en dehors du corpus")
        h = hashlib.sha256()
        with shard.open("rb") as stream:
            for block in iter(lambda: stream.read(1024*1024), b""):
                h.update(block)
        if h.hexdigest() != expected:
            raise ValueError(f"Shard corrompu : {name}")
    return value


def review_page(manifest_path, output):
    manifest_path = Path(manifest_path)
    manifest = load_review_manifest(manifest_path)
    contract = Contract.from_dict(manifest["contract"])
    clips = []
    for record in manifest["clips"]:
        glyphs, colors, _, metadata = read_clip(manifest_path.parent, record, contract)
        # Compress the exact uint8 tokens, without altering them. Decode only the
        # selected clip in the browser instead of allocating 40M JSON numbers.
        packed = gzip.compress(glyphs.tobytes(order="C")+colors.tobytes(order="C"), compresslevel=6, mtime=0)
        clips.append({"id": record["clip_id"], "tokens_gzip": base64.b64encode(packed).decode("ascii"), "metadata": metadata})
    data = {"contract": contract.to_dict(), "clips": clips, "fingerprint": manifest["fingerprint"],
            "converter_hash": manifest["converter"]["code_sha256"], "codec": "gzip-u8-glyphs-colors-1"}
    payload = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    page = Path(__file__).with_name("review_player.html").read_text(encoding="utf-8").replace("__DATA__", payload)
    with atomic_file(output) as stream:
        stream.write(page.encode("utf-8"))
    return str(Path(output).resolve())

