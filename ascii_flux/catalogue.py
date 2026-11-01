"""SQLite catalogue of immutable ASCII shards; frame counts are not epochs.

Payloads remain uint8 NPZ in TAR files. SQLite keeps only provenance, clip
locations and hashes, so a million grids need not be loaded into memory.
"""
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
from .contract import Contract, DEFAULT, canonical, digest
from .storage import atomic_file, read_clip, write_json


class Catalogue:
    def __init__(self, root, contract=DEFAULT, target_grids=1_000_000):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.contract = contract
        self.db = sqlite3.connect(self.root / "catalogue.sqlite", timeout=60)
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS runs(fingerprint TEXT PRIMARY KEY, manifest TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS videos(uid TEXT PRIMARY KEY, split TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS clips(
                uid TEXT PRIMARY KEY, run TEXT NOT NULL REFERENCES runs(fingerprint),
                video TEXT NOT NULL REFERENCES videos(uid), split TEXT NOT NULL,
                label TEXT NOT NULL, record TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS frames(
                clip TEXT NOT NULL REFERENCES clips(uid), t INTEGER NOT NULL,
                observation BLOB NOT NULL, tokens BLOB NOT NULL,
                PRIMARY KEY(clip,t));
            CREATE INDEX IF NOT EXISTS frame_observation ON frames(observation);
            CREATE INDEX IF NOT EXISTS frame_tokens ON frames(tokens);
            CREATE INDEX IF NOT EXISTS clip_split_label ON clips(split,label);
        """)
        with self.db:
            old = self.db.execute("SELECT value FROM settings WHERE key='contract'").fetchone()
            if old and old[0] != canonical(contract.to_dict()):
                raise ValueError("Catalogue representation mismatch; choose a new version")
            self.db.execute("INSERT OR IGNORE INTO settings VALUES('contract',?)", (canonical(contract.to_dict()),))
            self.db.execute("INSERT OR IGNORE INTO settings VALUES('target_grids',?)", (str(target_grids),))

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def contains_run(self, fingerprint):
        return bool(self.db.execute("SELECT 1 FROM runs WHERE fingerprint=?", (fingerprint,)).fetchone())

    def ingest(self, manifest_path):
        """Verify and commit a whole completed shard run, once, transactionally.

        Files are copied before the transaction; a crash can leave harmless
        unreferenced files. Repeating ingest then finishes the same transaction.
        """
        from .review import load_review_manifest
        from .factory import file_sha256
        path = Path(manifest_path).resolve()
        manifest = load_review_manifest(path)
        self.contract.require(manifest["contract"])
        fingerprint = manifest["fingerprint"]
        if self.contains_run(fingerprint):
            return False
        relative = Path("sets") / fingerprint
        destination = self.root / relative
        destination.mkdir(parents=True, exist_ok=True)
        for name, expected in manifest["shards"].items():
            source = (path.parent / name).resolve()
            target = (destination / name).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("Unsafe shard path")
            if not target.exists():
                with atomic_file(target) as stream, source.open("rb") as data:
                    shutil.copyfileobj(data, stream, 1024*1024)
            if file_sha256(target) != expected:
                raise ValueError("Stored shard checksum mismatch")
        with self.db:
            self.db.execute("INSERT INTO runs VALUES(?,?)", (fingerprint, canonical(manifest)))
            for record in manifest["clips"]:
                g, c, _, meta = read_clip(path.parent, record, self.contract)
                # Same source video stays in the same partition across snapshots.
                video = str(meta.get("source_video_uid") or f"{record['source']}:{record['video_id']}")
                old = self.db.execute("SELECT split FROM videos WHERE uid=?", (video,)).fetchone()
                if old and old[0] != record["split"]:
                    raise ValueError(f"Video split leakage: {video}")
                self.db.execute("INSERT OR IGNORE INTO videos VALUES(?,?)", (video, record["split"]))
                indices = meta.get("source_frame_indices", list(range(len(g))))
                paths = meta.get("source_frame_paths", [str(i) for i in indices])
                # Observation identity includes object: two subjects in a source
                # frame are different samples, while upsampling is a repetition.
                obj = str(meta.get("source_object_id") or record.get("instance_id", record["clip_id"]))
                sample = digest([video, obj, indices, meta.get("augmentation"), self.contract.hash, manifest["converter"]])
                if self.db.execute("SELECT 1 FROM clips WHERE uid=?", (sample,)).fetchone():
                    continue
                stored = dict(record, clip_id=record["clip_id"], shard=(relative/record["shard"]).as_posix())
                stored["catalogue_id"] = sample
                self.db.execute("INSERT INTO clips VALUES(?,?,?,?,?,?)", (
                    sample, fingerprint, video, record["split"], record["normalized_label"], canonical(stored)))
                frames = []
                for t in range(len(g)):
                    observation = hashlib.sha256(canonical([video, obj, paths[t]]).encode()).digest()
                    tokens = hashlib.sha256(g[t].tobytes()+c[t].tobytes()).digest()
                    frames.append((sample, t, observation, tokens))
                self.db.executemany("INSERT INTO frames VALUES(?,?,?,?)", frames)
        write_json(destination/"manifest.json", manifest)
        rejections = path.parent/"rejections.json"
        if rejections.exists():
            write_json(destination/"rejections.json", json.loads(rejections.read_text(encoding="utf-8")))
        return True

    def report(self):
        slots, observations, tokens = self.db.execute(
            "SELECT COUNT(*),COUNT(DISTINCT observation),COUNT(DISTINCT tokens) FROM frames").fetchone()
        target = int(self.db.execute("SELECT value FROM settings WHERE key='target_grids'").fetchone()[0])
        by_source = {}
        original_clips = augmented_clips = 0
        original_by_source = Counter()
        for (raw,) in self.db.execute("SELECT record FROM clips"):
            record = json.loads(raw)
            if record.get("augmentation"):
                augmented_clips += 1
            else:
                original_clips += 1
                original_by_source[record["source"]] += 1
        rejected = Counter()
        for (raw,) in self.db.execute("SELECT manifest FROM runs"):
            run = json.loads(raw)
            source = run["source"]["source"]
            rejected.update(run.get("report", {}).get("reasons", {}))
            counts = by_source.setdefault(source, {"clips": 0, "grid_slots": 0, "attempted_original_clips": 0, "conditions": run["source"]})
            actual = self.db.execute("SELECT COUNT(*) FROM clips WHERE run=?", (run["fingerprint"],)).fetchone()[0]
            counts["clips"] += actual
            counts["grid_slots"] += actual*self.contract.frames
            counts["attempted_original_clips"] += run.get("report", {}).get("attempted_full_clips", 0)
        for source, counts in by_source.items():
            counts["accepted_original_clips"] = original_by_source[source]
            counts["yield_original_clips"] = original_by_source[source]/max(1, counts["attempted_original_clips"])
        return {"target_grids": target, "clips": self.db.execute("SELECT COUNT(*) FROM clips").fetchone()[0],
                "grid_slots": slots, "distinct_source_object_observations": observations,
                "original_clips": original_clips, "augmented_clips": augmented_clips,
                "original_grid_slots": original_clips*self.contract.frames,
                "augmentation_grid_slots": augmented_clips*self.contract.frames,
                "distinct_token_grids": tokens, "repeated_observation_slots": slots-observations,
                "target_slots_reached": slots >= target, "target_distinct_grids_reached": tokens >= target,
                "split_clips": dict(self.db.execute("SELECT split,COUNT(*) FROM clips GROUP BY split")),
                "by_source": by_source, "rejection_reasons": dict(rejected),
                "status": "machine candidates; human quality validation required"}

    def snapshot(self, name="manifest-d2.json"):
        """Freeze a training view beside the DB; never change an existing view."""
        if Path(name).name != name:
            raise ValueError("Snapshot filename must stay in catalogue root")
        runs = [json.loads(row[0]) for row in self.db.execute("SELECT manifest FROM runs ORDER BY fingerprint")]
        if not runs:
            raise ValueError("Empty catalogue")
        converters = {canonical(run["converter"]) for run in runs}
        heldout = {canonical(run["config"]["heldout_categories"]) for run in runs}
        quality = {canonical(run["config"]["quality"]) for run in runs}
        if len(converters) != 1 or len(heldout) != 1 or len(quality) != 1:
            raise ValueError("A training snapshot requires one converter, held-out set and quality policy")
        records = [json.loads(row[0]) for row in self.db.execute("SELECT record FROM clips ORDER BY uid")]
        shards = {f"sets/{run['fingerprint']}/{name}": sha for run in runs for name, sha in run["shards"].items()}
        sources = {digest(r["source"]): r["source"] for r in runs}
        manifest = {"dataset_version": "ascii-v5-d2-catalogue-1", "contract": self.contract.to_dict(),
                    "contract_hash": self.contract.hash, "source": {"sources": [sources[key] for key in sorted(sources)]},
                    "converter": runs[0]["converter"], "config": {"heldout_categories": json.loads(next(iter(heldout))),
                    "quality": json.loads(next(iter(quality)))}, "clips": records, "shards": shards,
                    "report": self.report(), "catalogue_runs": [r["fingerprint"] for r in runs]}
        manifest["fingerprint"] = digest(manifest)
        target = self.root/name
        if target.exists() and json.loads(target.read_text(encoding="utf-8")) != manifest:
            raise FileExistsError("Snapshot is immutable; use a new name")
        write_json(target, manifest)
        return target


@contextmanager
def writer_lock(root):
    """OS advisory lock automatically releases after a killed process."""
    path = Path(root)/".writer.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            if not stream.read(1):
                stream.write(b"0"); stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield

