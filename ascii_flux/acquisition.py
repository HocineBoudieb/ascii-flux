"""Acquire an authorized YouTube-VIS 2019 training archive, before conversion.

The organizer portal requires a login to reveal download links. We accept those
links, never manufacture mirror URLs, and never request account credentials.
"""
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tarfile
import tempfile
import time
from urllib.parse import urlparse
import zipfile
from .factory import file_sha256, validate_source
from .storage import atomic_file, write_json


PORTAL = "https://codalab.lisn.upsaclay.fr/competitions/6064#participate-get_data"
DATASET_PAGE = "https://youtube-vos.org/dataset/vis/"
TERMS = "https://youtube-vos.org/dataset/term/"
PAPER = "https://arxiv.org/html/1905.04804v2#S4"
DOWNLOAD_USER_AGENT = "ASCII-Flux/5.0 (+offline research dataset acquisition)"


def default_data_root(project_root, content_root="/content"):
    """Use Drive only when it is actually mounted; local notebooks stay local."""
    content = Path(content_root)
    if content.is_dir():
        drive = content/"drive/MyDrive"
        return drive/"ascii_flux" if drive.is_dir() else content/"ascii_flux_data"
    return Path(project_root).resolve()/"data"


def _http_url(url):
    parsed = urlparse(str(url).strip())
    if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("Fournir un lien HTTP(S) de fichier, sans identifiants de compte.")
    if "/folders/" in parsed.path:
        raise ValueError("Fournir le lien du fichier, pas celui d'un dossier Google Drive.")
    return str(url).strip()


def download_file(url, destination, kind, downloader=None, expected_sha256=None):
    """Resume in a stable partial file; publish only after content validation."""
    url = _http_url(url)
    destination = Path(destination)
    receipt_path = destination.with_suffix(destination.suffix+".receipt.json")
    if destination.exists():
        if not receipt_path.exists():
            raise ValueError(f"Fichier existant sans reçu de provenance : {destination}. Choisir un autre dossier.")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt["url"] != url or receipt["sha256"] != file_sha256(destination):
            raise ValueError(f"Archive existante incompatible avec le lien demandé : {destination}")
        if expected_sha256 and receipt["sha256"] != expected_sha256:
            raise ValueError("SHA-256 différent de la révision attendue")
        return receipt
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix+".partial")
    partial_source = partial.with_suffix(partial.suffix+".source.json")
    if partial_source.exists():
        if json.loads(partial_source.read_text(encoding="utf-8"))["url"] != url:
            raise ValueError("Téléchargement partiel d'une autre URL ; choisir un nouveau dossier d'acquisition.")
    elif partial.exists():
        raise ValueError("Téléchargement partiel sans provenance ; choisir un nouveau dossier d'acquisition.")
    write_json(partial_source, {"url": url, "kind": kind})
    if downloader is None:
        try:
            import gdown
        except ImportError as exc:
            raise ImportError("gdown manque. Relancer l'installation initiale puis redémarrer le noyau ; aucun paquet n'est installé ici.") from exc
        downloader = gdown.download
    try:
        # gdown's default Chrome/39 identity receives HTML from Kaggle's public
        # API. Identify this API client consistently with the verified download.
        result = downloader(url=url, output=str(partial), quiet=False, resume=True, fuzzy=True,
                            use_cookies=False, user_agent=DOWNLOAD_USER_AGENT)
    except Exception as exc:
        raise OSError("Échec du téléchargement. Vérifier le lien et les droits d'accès au fichier : " + str(exc)) from exc
    if not result or not partial.is_file():
        raise OSError("Téléchargement incomplet ou accès refusé. Vérifier le lien officiel ; ne pas coller de mot de passe.")
    # A login page, Drive quota message or confirmation HTML is not an archive.
    try:
        if kind == "frames":
            if not zipfile.is_zipfile(partial) and not tarfile.is_tarfile(partial):
                raise ValueError("Le lien images n'a retourné ni ZIP ni TAR. Vérifier l'accès au fichier dans le navigateur.")
        elif not zipfile.is_zipfile(partial):
            try:
                data = json.loads(partial.read_text(encoding="utf-8"))
            except (ValueError, UnicodeError) as exc:
                raise ValueError("Le lien annotations n'a retourné ni un JSON ni un ZIP d'annotations.") from exc
            _validate_annotations(data)
    except ValueError:
        # A completed HTML/error response cannot be resumed as archive bytes.
        partial.unlink(missing_ok=True)
        partial_source.unlink(missing_ok=True)
        raise
    receipt = {"url": url, "kind": kind, "sha256": file_sha256(partial), "bytes": partial.stat().st_size}
    if expected_sha256 and receipt["sha256"] != expected_sha256:
        raise ValueError("SHA-256 du téléchargement différent de la révision attendue ; archive non publiée")
    with partial.open("r+b") as stream:
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, destination)
    write_json(receipt_path, receipt)
    partial_source.unlink(missing_ok=True)
    return receipt


def _publish_directory(staging, destination):
    # Antivirus/indexing handles can briefly prevent renaming a completed
    # directory on Windows. Preserve the validated staging tree if it persists.
    for attempt in range(15):
        try:
            os.replace(staging, destination)
            return
        except PermissionError:
            if attempt == 14:
                raise
            time.sleep(1)


def extract_zip(archive, destination, durable_files=True):
    """Atomic extraction. Reject path traversal, links and Windows path aliases."""
    destination = Path(destination).resolve()
    receipt = {"archive_sha256": file_sha256(archive)}
    marker = destination/".extracted.json"
    if destination.exists():
        if marker.exists() and json.loads(marker.read_text(encoding="utf-8")) == receipt:
            return destination
        raise FileExistsError(f"Dossier d'extraction existant sans reçu correspondant : {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    for previous in destination.parent.glob(destination.name+".*"):
        previous_marker = previous/".extracted.json"
        if previous.is_dir() and previous_marker.exists() and json.loads(previous_marker.read_text(encoding="utf-8")) == receipt:
            _publish_directory(previous, destination)
            return destination
    staging = Path(tempfile.mkdtemp(prefix=destination.name+".", dir=destination.parent)).resolve()
    try:
        with zipfile.ZipFile(archive) as zipped:
            members = zipped.infolist()
            for member in members:
                name = member.filename
                path = PurePosixPath(name)
                if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
                    raise ValueError("Chemin dangereux dans le ZIP")
                if any(part.rstrip(". ") != part for part in path.parts):
                    raise ValueError("Chemin ambigu dans le ZIP")
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError("Lien symbolique interdit dans le ZIP")
                target = (staging/Path(*path.parts)).resolve()
                if not target.is_relative_to(staging):
                    raise ValueError("Chemin hors du dossier d'extraction")
            for member in members:
                target = staging/Path(*PurePosixPath(member.filename).parts)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if durable_files:
                    with zipped.open(member) as source, atomic_file(target) as stream:
                        shutil.copyfileobj(source, stream, length=1024*1024)
                else:
                    # Reconstructible media cache in an unpublished private
                    # directory. ZIP checks each CRC; the downloaded archive is
                    # already durable. Corpus shards always use fsync separately.
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zipped.open(member) as source, target.open("wb") as stream:
                        shutil.copyfileobj(source, stream, length=1024*1024)
        write_json(staging/".extracted.json", receipt)
        if destination.exists():
            raise FileExistsError(f"Dossier apparu pendant l'extraction : {destination}")
        _publish_directory(staging, destination)
        return destination
    except BaseException:
        if not (staging/".extracted.json").exists() and staging.parent == destination.parent and staging.name.startswith(destination.name+"."):
            shutil.rmtree(staging)
        raise


def _validate_annotations(data):
    if not isinstance(data, dict) or not all(isinstance(data.get(k), list) and data[k] for k in ("videos", "annotations", "categories")):
        raise ValueError("Choisir les annotations TRAIN YouTube-VIS : videos, annotations et categories doivent être présents et non vides.")
    if any(not isinstance(v.get("file_names"), list) or not v["file_names"] for v in data["videos"]):
        raise ValueError("Les annotations ne contiennent pas les noms des images vidéo.")
    if any(not isinstance(a.get("segmentations"), list) for a in data["annotations"]):
        raise ValueError("Masques d'instances absents du fichier d'annotations.")
    return data


def find_annotations(root):
    root = Path(root)
    if root.is_file():
        return root, _validate_annotations(json.loads(root.read_text(encoding="utf-8")))
    found = []
    for path in sorted(root.rglob("*.json")):
        if path.name.startswith(".") or path.stat().st_size < 100:
            continue
        try:
            data = _validate_annotations(json.loads(path.read_text(encoding="utf-8")))
            found.append((path, data))
        except (ValueError, UnicodeError, AttributeError):
            continue
    preferred = [item for item in found if "train" in str(item[0].relative_to(root)).lower()]
    found = preferred or found
    if len(found) != 1:
        raise ValueError(f"Trouvé {len(found)} fichiers d'annotations utilisables. Fournir le chemin exact des annotations d'entraînement.")
    return found[0]


def find_frames_root(root, data):
    root = Path(root).resolve()
    first = data["videos"][0]["file_names"][0]
    relative = PurePosixPath(first)
    if relative.is_absolute() or ".." in relative.parts or "\\" in first or ":" in first:
        raise ValueError("Chemin d'image invalide dans les annotations")
    matches = []
    # Find only the first named file, then validate every listed frame under its root.
    for candidate in root.rglob(relative.name):
        parent = candidate
        for _ in relative.parts:
            parent = parent.parent
        if (parent/Path(*relative.parts)).resolve() != candidate.resolve():
            continue
        if parent not in matches:
            matches.append(parent)
    if len(matches) != 1:
        raise ValueError(f"Impossible de choisir la racine des images ({len(matches)} correspondances). Vérifier l'archive train.")
    selected = matches[0].resolve()
    missing = []
    count = 0
    for video in data["videos"]:
        for name in video["file_names"]:
            path = (selected/name).resolve()
            if not path.is_relative_to(selected):
                raise ValueError("Image en dehors du dossier source")
            if not path.is_file() and len(missing) < 5:
                missing.append(name)
            count += 1
    if missing:
        raise ValueError(f"Archive d'images incomplète ; exemples manquants : {missing}")
    return selected, count


def prepare_ytvis2019(data_root, annotations_url, frames_url, source_template, source_fps=6., downloader=None):
    """Download -> validate -> extract -> persist exact source, paths and cadence.

    6 Hz is the listed annotated-frame cadence for the 2019 release described in
    section 4 of the paper (30 fps sampled every 5 frames), not the raw-video FPS.
    Only filenames listed by the instance annotations are read by the converter.
    """
    if not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("Cadence des images annotées invalide")
    source = dict(source_template)
    source.update(snapshot="YouTube-VIS-2019/train", local_media_provenance={
        "annotations_url": _http_url(annotations_url), "frames_url": _http_url(frames_url), "portal": PORTAL})
    validate_source(source)
    root = Path(data_root).resolve()/"ytvis-2019"
    print("Source : YouTube-VIS 2019 train. Annotations CC BY 4.0 ; données pour recherche non commerciale.", flush=True)
    print("Conditions :", TERMS, flush=True)
    print("Téléchargement des annotations…", flush=True)
    ann_file = root/"downloads/annotations.download"
    ann_receipt = download_file(annotations_url, ann_file, "annotations", downloader)
    if zipfile.is_zipfile(ann_file):
        ann_path, data = find_annotations(extract_zip(ann_file, root/"annotations"))
    else:
        ann_path, data = find_annotations(ann_file)
    print("Téléchargement des images d'entraînement (archive volumineuse)…", flush=True)
    frames_file = root/"downloads/train.zip"
    frames_receipt = download_file(frames_url, frames_file, "frames", downloader)
    extracted = extract_zip(frames_file, root/"frames")
    print("Vérification des images listées dans les annotations…", flush=True)
    frames_root, count = find_frames_root(extracted, data)
    source["local_media_provenance"].update(annotations_sha256=ann_receipt["sha256"], frames_sha256=frames_receipt["sha256"])
    source["cadence_evidence"] = {"url": PAPER, "scope": "2019 listed annotated frames; original 30 fps / stride 5", "source_fps": source_fps}
    prepared = {"annotations": str(ann_path.resolve()), "annotations_sha256": file_sha256(ann_path),
                "frames_root": str(frames_root), "source_fps": source_fps, "source": source,
                "videos": len(data["videos"]), "instances": len(data["annotations"]), "listed_frames": count}
    write_json(root/"prepared.json", prepared)
    print(f"Prêt pour D1 : {prepared['videos']} vidéos, {prepared['instances']} instances, {count} images listées.", flush=True)
    return prepared


def load_prepared(data_root, profile="ytvis-2019"):
    if profile not in ("ytvis-2019", "ytvis-2019-public"):
        raise ValueError("Profil d'acquisition inconnu")
    path = Path(data_root)/profile/"prepared.json"
    if not path.exists():
        return None
    prepared = json.loads(path.read_text(encoding="utf-8"))
    annotations = Path(prepared["annotations"])
    if not annotations.is_file() or file_sha256(annotations) != prepared["annotations_sha256"]:
        raise ValueError("Annotations déplacées ou modifiées depuis l'acquisition. Relancer la préparation.")
    if not Path(prepared["frames_root"]).is_dir():
        raise ValueError("Images déplacées depuis l'acquisition. Relancer la préparation.")
    validate_source(prepared["source"])
    return prepared
