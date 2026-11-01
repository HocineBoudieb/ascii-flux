"""Public mirror profile for YouTube-VIS 2019, with exact source attribution.

VIS labels/RLE masks come from a research repack on Kaggle. The corresponding
original video frames come from a version-pinned YouTube-VOS archive on HF.
No new media rights are inferred from a mirror's availability or license label.
"""
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import zipfile
from .acquisition import download_file, _validate_annotations
from .contract import DEFAULT, digest
from .factory import file_sha256, validate_source
from .storage import atomic_file, write_json


TRANSFORM_VERSION = "kaggle-sequences-to-vis-1"


def normalize_repacked_annotations(raw, minimum_frames=24):
    """Repackage tokens/IDs only; do not rewrite category labels or synthesize masks.

    Videos with dropped/irregular listed frames are explicitly excluded because
    the initial converter expects a uniform annotated-frame cadence.
    """
    if not isinstance(raw, dict) or not isinstance(raw.get("sequences"), list):
        raise ValueError("Miroir d'annotations : format meta+sequences attendu")
    labels = raw.get("meta", {}).get("category_labels", {})
    if not labels:
        raise ValueError("Miroir d'annotations sans labels de catégorie")
    categories = [{"id": int(k), "name": str(v)} for k, v in labels.items()]
    videos, annotations, excluded, identities = [], [], [], set()
    for seq in raw["sequences"]:
        paths = seq.get("image_paths", [])
        paths = [PurePosixPath(p) for p in paths]
        if not paths or any(p.is_absolute() or ".." in p.parts or len(p.parts) != 3 or p.parts[0] != "JPEGImages" or
                            "\\" in str(p) or ":" in str(p) for p in paths):
            raise ValueError("Chemins d'images inattendus dans le miroir d'annotations")
        video_id = paths[0].parts[1]
        if video_id in identities or any(p.parts[1] != video_id for p in paths):
            raise ValueError("Identité de vidéo source incohérente ou dupliquée")
        if any(c not in "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_" for c in video_id):
            raise ValueError("Identifiant vidéo invalide")
        identities.add(video_id)
        try:
            frame_ids = [int(p.stem) for p in paths]
        except ValueError as exc:
            raise ValueError("Noms des images sans indices temporels") from exc
        reasons = []
        if len(paths) < minimum_frames:
            reasons.append("shorter_than_target_clip")
        if any(b-a != 5 for a, b in zip(frame_ids, frame_ids[1:])):
            reasons.append("irregular_annotated_frame_cadence")
        if len(seq.get("segmentations", [])) != len(paths) or seq.get("length") != len(paths):
            raise ValueError("Nombre d'images et de masques incohérent dans le miroir")
        if reasons:
            excluded.append({"video_id": video_id, "original_sequence_id": seq.get("id"), "reasons": reasons})
            continue
        videos.append({"id": video_id, "file_names": [str(p) for p in paths], "height": int(seq["height"]),
                       "width": int(seq["width"]), "length": len(paths), "original_sequence_id": seq.get("id"),
                       "source_frame_indices_30fps": frame_ids})
        for object_id, category in seq["categories"].items():
            if int(category) not in {c["id"] for c in categories}:
                raise ValueError("Catégorie d'objet inconnue")
            masks = []
            for frame_masks in seq["segmentations"]:
                counts = frame_masks.get(str(object_id))
                if counts is not None and not isinstance(counts, str):
                    raise ValueError("Masque COCO RLE compressé attendu")
                masks.append(None if not counts else {"size": [int(seq["height"]), int(seq["width"])], "counts": counts})
            annotations.append({"id": f"{video_id}-o{object_id}", "video_id": video_id,
                                "category_id": int(category), "segmentations": masks,
                                "source_object_id": str(object_id), "original_sequence_id": seq.get("id")})
    result = {"videos": videos, "annotations": annotations, "categories": categories}
    _validate_annotations(result)
    return result, {"transform_version": TRANSFORM_VERSION, "input_videos": len(raw["sequences"]),
                    "eligible_videos": len(videos), "eligible_instances": len(annotations),
                    "minimum_annotated_frames": minimum_frames, "excluded": excluded,
                    "labels": "preserved verbatim from the mirror; no action invented",
                    "ids": "original video directory IDs and object IDs; numeric sequence IDs retained as metadata"}


def _tar_target(staging, member):
    name = member.name
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name or any(part.rstrip(". ") != part for part in path.parts):
        raise ValueError("Chemin dangereux ou ambigu dans le TAR")
    if not (member.isfile() or member.isdir()):
        raise ValueError("Liens et fichiers spéciaux interdits dans le TAR")
    target = (staging/Path(*path.parts)).resolve()
    if not target.is_relative_to(staging):
        raise ValueError("Chemin hors du dossier d'extraction")
    return target


def extract_selected_tar(archive, destination, wanted, archive_sha256=None):
    """Keep only images named by the selected VIS annotations; publish atomically."""
    destination = Path(destination).resolve()
    selection = set(wanted)
    receipt = {"archive_sha256": archive_sha256 or file_sha256(archive), "selection_sha256": digest(sorted(selection))}
    marker = destination/".extracted.json"
    if destination.exists():
        if marker.is_file() and json.loads(marker.read_text(encoding="utf-8")) == receipt:
            return destination
        raise FileExistsError("Extraction existante d'une autre archive/sélection ; choisir un autre dossier")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=destination.name+".", dir=destination.parent)).resolve()
    try:
        count = 0
        with tarfile.open(archive, "r:*") as tar:
            for member in tar:
                target = _tar_target(staging, member)
                if not member.isfile() or member.name not in selection:
                    continue
                with tar.extractfile(member) as source, atomic_file(target) as stream:
                    shutil.copyfileobj(source, stream, length=1024*1024)
                count += 1
                if count % 2000 == 0:
                    print(f"Extraction : {count} images VIS…", flush=True)
        if not count:
            raise ValueError("Aucune image de l'archive ne correspond aux annotations VIS")
        write_json(staging/".extracted.json", receipt)
        if destination.exists():
            raise FileExistsError("Dossier apparu pendant l'extraction")
        os.replace(staging, destination)
        return destination
    except BaseException:
        if staging.parent == destination.parent and staging.name.startswith(destination.name+"."):
            shutil.rmtree(staging)
        raise


def prepare_public_ytvis2019(data_root, source_template, profile, downloader=None):
    """Public downloads with pinned hashes; never contact CodaLab or ask for login."""
    root = Path(data_root).resolve()/"ytvis-2019-public"
    source = dict(source_template)
    source.update(snapshot="YouTube-VIS-2019/train; public mirrors profile "+profile["profile"],
                  terms_checked_on=profile["checked_on"], local_media_provenance={
                      "distribution": "Kaggle VIS annotations + matching original YouTube-VOS frames on Hugging Face",
                      "annotations_mirror": profile["annotations"], "frames_mirror": profile["frames"]})
    validate_source(source)  # Inherited source conditions checked before any data acquisition.
    print("Miroirs publics : Kaggle (labels/masques VIS) + Hugging Face (images source).", flush=True)
    print("Conditions originales conservées : recherche non commerciale ; annotations CC BY 4.0.", flush=True)
    print("Conditions :", profile["terms_url"], flush=True)
    ann = profile["annotations"]
    ann_file = root/"downloads/annotations.zip"
    ann_receipt = download_file(ann["url"], ann_file, "annotations", downloader, ann["sha256"])
    with zipfile.ZipFile(ann_file) as zipped:
        names = [n for n in zipped.namelist() if n.endswith(".json")]
        if len(names) != 1:
            raise ValueError("Une seule annotation JSON attendue dans la révision épinglée")
        with zipped.open(names[0]) as stream:
            raw = json.load(stream)
    minimum_frames = math.ceil(DEFAULT.frames/DEFAULT.fps*profile["source_fps"])
    data, conversion = normalize_repacked_annotations(raw, minimum_frames)
    print(f"Annotations : {conversion['eligible_videos']} vidéos admissibles ; exclusions temporelles journalisées.", flush=True)
    media = profile["frames"]
    print(f"Images source : archive TAR de {media['bytes']/1e9:.2f} Go. Les fichiers acquis sont réutilisés.", flush=True)
    media_file = root/"downloads/train.tar"
    media_receipt = download_file(media["url"], media_file, "frames", downloader, media["sha256"])
    wanted = {"train/"+name for video in data["videos"] for name in video["file_names"]}
    extracted = extract_selected_tar(media_file, root/"frames", wanted, media_receipt["sha256"])
    frames_root = extracted/"train"
    complete, missing = [], []
    for video in data["videos"]:
        absent = [p for p in video["file_names"] if not (frames_root/p).is_file()]
        if absent:
            missing.append({"video_id": video["id"], "reasons": ["media_missing_in_vos_mirror"], "missing_frames": absent})
        else:
            complete.append(video)
    if not complete:
        raise ValueError("Aucune vidéo complète disponible dans le miroir d'images")
    ids = {v["id"] for v in complete}
    data["videos"] = complete
    data["annotations"] = [a for a in data["annotations"] if a["video_id"] in ids]
    conversion["media_exclusions"] = missing
    conversion["available_videos"] = len(complete)
    conversion["available_instances"] = len(data["annotations"])
    annotations_path = root/"annotations.normalized.json"
    # Verify the dimensions of actual image files before publishing readiness.
    from PIL import Image
    for video in complete:
        with Image.open(frames_root/video["file_names"][0]) as image:
            if image.size != (video["width"], video["height"]):
                raise ValueError(f"Dimensions incohérentes pour la vidéo {video['id']}")
    write_json(annotations_path, data)
    write_json(root/"mirror-conversion-report.json", conversion)
    source["local_media_provenance"].update(annotations_archive_sha256=ann_receipt["sha256"],
        media_archive_sha256=media_receipt["sha256"], normalization=TRANSFORM_VERSION,
        conversion_report_sha256=file_sha256(root/"mirror-conversion-report.json"))
    source["cadence_evidence"] = {"url": profile["cadence_evidence"], "source_fps": profile["source_fps"],
                                   "scope": "only sequences with constant filename stride 5 at original 30 fps"}
    prepared = {"annotations": str(annotations_path), "annotations_sha256": file_sha256(annotations_path),
                "frames_root": str(frames_root), "source_fps": profile["source_fps"], "source": source,
                "videos": len(complete), "instances": len(data["annotations"]),
                "listed_frames": sum(len(v["file_names"]) for v in complete), "mirror_profile": profile["profile"]}
    write_json(root/"prepared.json", prepared)
    print(f"Prêt pour D1 : {prepared['videos']} vidéos, {prepared['instances']} instances, sans compte CodaLab.", flush=True)
    return prepared
