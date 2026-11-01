"""Offline teacher shape corpus: words -> teacher images -> cut-outs -> glyph grids.

Tagged ``synthetic: teacher-t2i`` everywhere. Resumable by shard; each shard
is an NPZ of tokens plus a JSON sidecar published last, atomically.
Words are split by a hash of the word: reserved words never train.
"""
import hashlib
import io
import json
from pathlib import Path
import time
import numpy as np
from .contract import canonical, normalize_text
from .storage import atomic_file, write_json
from .structure import STRUCTURE, to_glyphs

VOCABULARY_VERSION = "lvvis-imagenet12k-1"


def _timm_info():
    import timm.data
    return Path(timm.data.__file__).parent/"_info"


def build_vocabulary(lvvis_annotations):
    """Ordered, de-duplicated words with their origins; common concepts first."""
    info = _timm_info()
    lemmas = dict(line.rstrip("\n").split("\t", 1) for line in (info/"imagenet_synset_to_lemma.txt").read_text(encoding="utf-8").splitlines() if "\t" in line)
    order, entries = [], {}

    def add(word, origin):
        word = normalize_text(word.split(",")[0].replace("_", " "))
        word = " ".join(word.split())
        if not word or any(ch.isdigit() for ch in word) or len(word) > 40:
            return
        if word not in entries:
            entries[word] = {"word": word, "origins": []}
            order.append(word)
        entries[word]["origins"].append(origin)

    for synset in (info/"imagenet_synsets.txt").read_text(encoding="utf-8").split():
        add(lemmas[synset], f"imagenet1k:{synset}")
    categories = json.loads(Path(lvvis_annotations).read_text(encoding="utf-8"))["categories"]
    for category in categories:
        name = category["name"].split("_(")[0]
        add(name, f"lvvis:{category['id']}")
    for synset in (info/"imagenet12k_synsets.txt").read_text(encoding="utf-8").split():
        if synset in lemmas:
            add(lemmas[synset], f"imagenet12k:{synset}")
    return [entries[w] for w in order]


def split_of(word, reserved_percent=5):
    value = int(hashlib.sha256(word.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "test_unseen" if value < reserved_percent else "val" if value < reserved_percent+3 else "train"


def build_shape_corpus(output, lvvis_annotations, keep=3, attempts=8, shard_words=100, max_words=None, device="cuda"):
    from .teacher import TEMPLATES, Teacher, check_mask
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    vocabulary = build_vocabulary(lvvis_annotations)[:max_words]
    write_json(output/"vocabulary.json", {"version": VOCABULARY_VERSION, "words": vocabulary})
    done = set()
    for sidecar in output.glob("shard-*.json"):
        done.update(r["word"] for r in json.loads(sidecar.read_text(encoding="utf-8"))["words"])
    pending = [entry for entry in vocabulary if entry["word"] not in done]
    shard_index = len(list(output.glob("shard-*.json")))
    teacher = Teacher(device)
    provenance = teacher.provenance() | {"contract": STRUCTURE.to_dict(), "contract_hash": STRUCTURE.hash,
                                         "vocabulary_version": VOCABULARY_VERSION, "keep": keep, "attempts": attempts}
    write_json(output/"provenance.json", provenance)
    started, processed = time.monotonic(), 0
    for first in range(0, len(pending), shard_words):
        glyphs, colors, words, reasons = [], [], [], {}
        for entry in pending[first:first+shard_words]:
            word, base = entry["word"], int(hashlib.sha256(entry["word"].encode("utf-8")).hexdigest()[:8], 16)
            accepted = []
            for batch in range(0, attempts, 4):
                if len(accepted) >= keep:
                    break
                seeds = [base+batch+i for i in range(4)]
                templates = [(batch+i) % len(TEMPLATES) for i in range(4)]
                images = teacher.draw([TEMPLATES[t].format(word=word) for t in templates], seeds)
                for image, mask, seed, template in zip(images, teacher.masks(images), seeds, templates):
                    reason = check_mask(mask)
                    if reason is None and len(accepted) < keep:
                        g, c = to_glyphs(image, mask, STRUCTURE.palette)
                        if (g != 0).mean() < .02:
                            reason = "grid_nearly_empty"
                        else:
                            accepted.append((g, c, seed, template))
                    if reason:
                        reasons[reason] = reasons.get(reason, 0)+1
            record = {"word": word, "origins": entry["origins"], "split": split_of(word), "samples": [],
                      "synthetic": "teacher-t2i"}
            for g, c, seed, template in accepted:
                record["samples"].append({"index": len(glyphs), "seed": seed, "template": template})
                glyphs.append(g)
                colors.append(c)
            words.append(record)
            processed += 1
        name = f"shard-{shard_index:05d}"
        payload = io.BytesIO()
        np.savez_compressed(payload, glyphs=np.asarray(glyphs, np.uint8).reshape(-1, STRUCTURE.rows, STRUCTURE.cols),
                            colors=np.asarray(colors, np.uint8).reshape(-1, STRUCTURE.rows, STRUCTURE.cols))
        with atomic_file(output/f"{name}.npz") as stream:
            stream.write(payload.getvalue())
        sha = hashlib.sha256(payload.getvalue()).hexdigest()
        write_json(output/f"{name}.json", {"npz_sha256": sha, "contract_hash": STRUCTURE.hash, "words": words,
                                           "rejections": reasons})
        shard_index += 1
        elapsed = time.monotonic()-started
        write_json(output/"progress.json", {"words_done": len(done)+processed, "words_total": len(vocabulary),
                                            "seconds_per_word": elapsed/max(processed, 1),
                                            "last_shard": name, "synthetic": "teacher-t2i"})
        print(canonical({"shard": name, "words_done": len(done)+processed, "of": len(vocabulary),
                         "s_per_word": round(elapsed/max(processed, 1), 2), "rejections": reasons}), flush=True)
