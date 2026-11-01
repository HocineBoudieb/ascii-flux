import base64
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import pytest
from ascii_flux.contract import Contract
from ascii_flux.review import load_review_manifest, review_page
from ascii_flux.storage import read_clip
from tests.fixtures import make_corpus


def test_review_recovers_completed_dataset_without_notebook_state(tmp_path):
    manifest_path, _ = make_corpus(tmp_path/"corpus")
    value = load_review_manifest(manifest_path.parent)
    assert value == load_review_manifest(manifest_path)
    assert len(value["clips"]) == 4
    shard = manifest_path.parent/value["clips"][0]["shard"]
    with shard.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="corrompu"):
        load_review_manifest(manifest_path.parent)


def test_compact_review_preserves_all_glyph_and_color_tokens(tmp_path):
    manifest_path, _ = make_corpus(tmp_path/"corpus")
    output = tmp_path/"review.html"
    assert review_page(manifest_path, output) == str(output.resolve())
    html = output.read_text(encoding="utf-8")
    payload = html.split("const data=", 1)[1].split(";\n", 1)[0]
    value = json.loads(payload)
    manifest = load_review_manifest(manifest_path)
    contract = Contract.from_dict(manifest["contract"])
    for clip, record in zip(value["clips"], manifest["clips"]):
        glyphs, colors, _, _ = read_clip(manifest_path.parent, record, contract)
        expected = glyphs.tobytes()+colors.tobytes()
        assert gzip.decompress(base64.b64decode(clip["tokens_gzip"])) == expected
        assert "glyphs" not in clip and "colors" not in clip
    assert value["fingerprint"] == manifest["fingerprint"]
    assert "value=\"accept\"" in html and "value=\"pending\"" in html
    if shutil.which("node"):
        script = html.split("<script>", 1)[1].split("</script>", 1)[0]
        js_path = tmp_path/"review.js"
        js_path.write_text(script, encoding="utf-8")
        subprocess.run(["node", "--check", str(js_path)], check=True, capture_output=True)
        # Exercise the same browser-native gzip decoder used by the viewer.
        start = script.index("async function unpackClip(")
        end = script.index("async function show(", start)
        decoder = script[start:end]
        check = ("const data="+payload+"; const tokenCount=data.contract.frames*data.contract.rows*data.contract.cols;\n"+
                 decoder+"\nunpackClip(data.clips[0]).then(b=>process.stdout.write(Buffer.from(b).toString('base64')));")
        decoded = subprocess.run(["node", "-e", check], check=True, capture_output=True, text=True)
        first_g, first_c, _, _ = read_clip(manifest_path.parent, manifest["clips"][0], contract)
        assert base64.b64decode(decoded.stdout) == first_g.tobytes()+first_c.tobytes()

