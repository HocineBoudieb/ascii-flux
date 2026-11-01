"""Build an autonomous Colab notebook, preserving its identity and stable cell IDs."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ascii_flux.storage import write_json


def main():
    files = [ROOT/"pyproject.toml", *sorted((ROOT/"ascii_flux").glob("*.py")),
             *sorted((ROOT/"ascii_flux").glob("*.html")), *sorted((ROOT/"configs").glob("*.json"))]
    sources = {p.relative_to(ROOT).as_posix(): p.read_text(encoding="utf-8") for p in files}
    source_hash = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    cells = []

    def add(kind, identifier, text):
        cell = {"cell_type": kind, "id": identifier, "metadata": {}, "source": text.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add("markdown", "intro-v5", """# ASCII Flux v5 — usine D1 et baseline F1

Ce notebook contient ses sources Python et fonctionne sans dépôt externe.
La v4.2 reste gelée. Les clips créés par D1 sont des candidats à revoir ; aucun
corpus validé ni poids préentraîné v5 n'est fourni.
**Tous les jalons restent à valider.** Le convertisseur travaille hors ligne ;
l'inférence NCA lit et produit directement des jetons glyphe/couleur.

Exécuter l'installation au début d'une session vierge, laisser le noyau redémarrer,
puis reprendre à la cellule suivante. Les archives YouTube-VIS doivent être obtenues
selon les [conditions officielles](https://youtube-vos.org/dataset/term/) :
annotations CC BY 4.0, données pour recherche non commerciale uniquement.
Les étapes de revue humaine ne sont pas exécutables automatiquement par « Tout exécuter ».
""")
    bootstrap = '''# Installation AVANT les imports scientifiques, puis redémarrage du noyau.
import hashlib, json, os, signal, subprocess, sys
from pathlib import Path
SOURCES = __SOURCES__
SOURCE_HASH = __HASH__
WORK = Path('/content/ascii_flux_v5') if Path('/content').exists() else Path.cwd()/'ascii_flux_v5_colab'
WORK.mkdir(parents=True, exist_ok=True)
for name, content in SOURCES.items():
    target = WORK/name
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix+'.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        stream.write(content); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, target)
marker = WORK/'.dependencies-installed'
if not marker.exists() or marker.read_text() != SOURCE_HASH:
    already = [name for name in ('numpy','PIL','cv2','torch','transformers','huggingface_hub') if name in sys.modules]
    if already:
        raise RuntimeError('Redémarrer le noyau avant installation : dépendances déjà importées '+str(already))
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-e', str(WORK)+'[train,vis,export]'])
    marker.write_text(SOURCE_HASH)
    print('Installation terminée. Redémarrage requis avant la cellule suivante.', flush=True)
    if 'google.colab' in sys.modules:
        os.kill(os.getpid(), signal.SIGKILL)
    raise SystemExit('Redémarrer le noyau, puis reprendre à la cellule suivante.')
print('Dépendances déjà installées ; aucune installation dans ce noyau.')
'''.replace("__SOURCES__", repr(sources)).replace("__HASH__", repr(source_hash))
    add("code", "bootstrap-v5", bootstrap)
    add("code", "paths-v5", '''# Après redémarrage : aucune installation dans les cellules suivantes.
import json, sys
from pathlib import Path
WORK = Path('/content/ascii_flux_v5') if Path('/content').exists() else Path.cwd()/'ascii_flux_v5_colab'
sys.path.insert(0, str(WORK))
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Périphérique :', DEVICE)
# Google Drive est utilisé seulement s'il est déjà monté.
from ascii_flux.acquisition import default_data_root
DATA_ROOT = default_data_root(WORK.parent)
ANNOTATIONS = None
FRAMES_ROOT = None
DATASET = DATA_ROOT/'v5/d1-public-001'
RUN = DATA_ROOT/'runs/v5-001'
DOWNLOAD_DATA = True  # False : préparer les chemins sans télécharger les archives.
print('Données :', DATA_ROOT)
print('Profil : YouTube-VIS 2019 TRAIN via miroirs publics Kaggle + Hugging Face, sans connexion CodaLab.')
''')
    add("markdown", "acquisition-method-v5", """## Télécharger YouTube-VIS avant D1

Le profil par défaut utilise deux **miroirs publics vérifiés le 31 octobre 2026** :

- [Kaggle : annotations YouTube-VIS train 2019, version 1](https://www.kaggle.com/datasets/farnoosharefi/youtube-vis-train2019)
  fournit les catégories et les masques COCO RLE, dans un format réorganisé.
- [Hugging Face : images source YouTube-VOS 2019](https://huggingface.co/datasets/yinloonga/YouTube_VOS_2019)
  fournit les images originales correspondantes. La révision et le SHA-256 du TAR
  sont épinglés. Seules les images des vidéos VIS admissibles sont extraites.

**Aucune connexion CodaLab, aucun token et aucune URL à renseigner.**
L'archive d'annotations fait environ 64 Mo et celle des images **9,26 Go** ; prévoir
aussi la place pour les images extraites. Les téléchargements terminés sont réutilisés.
Pour préparer seulement les chemins, régler `DOWNLOAD_DATA = False` dans la cellule précédente.

Le notebook adapte le format d'annotations sans inventer de masque ou de label.
Les vidéos dont les images annotées ont une cadence irrégulière et les vidéos trop
courtes sont exclues avec un motif dans `mirror-conversion-report.json`. La couverture
du miroir d'images est vérifiée et les absences sont également publiées.
Les annotations CC BY 4.0 et les [conditions originales de recherche non commerciale](https://youtube-vos.org/dataset/term/)
sont consignées avant l'acquisition. Un miroir n'accorde pas de nouveaux droits.
Le profil lit les images annotées à 6 Hz (30 Hz / intervalle 5,
[article §4](https://arxiv.org/html/1905.04804v2#S4)).

Si un lien renvoie une page de connexion, un quota Google Drive ou du HTML,
l'acquisition s'arrête avec un message ; ce contenu n'est jamais traité comme un dataset.
""")
    add("code", "acquire-v5", '''from ascii_flux.acquisition import load_prepared
from ascii_flux.public_source import prepare_public_ytvis2019
PREPARED = None
try:
    PREPARED = load_prepared(DATA_ROOT, profile='ytvis-2019-public')
    if PREPARED is None:
        if globals().get('DOWNLOAD_DATA', True):
            profile = json.loads((WORK/'configs/ytvis2019_public.json').read_text(encoding='utf-8'))
            template = json.loads((WORK/'configs/ytvis_source.json').read_text(encoding='utf-8'))
            PREPARED = prepare_public_ytvis2019(DATA_ROOT, template, profile)
        else:
            print('Acquisition en attente : régler DOWNLOAD_DATA = True pour télécharger les miroirs publics.')
    if PREPARED is not None:
        ANNOTATIONS = Path(PREPARED['annotations'])
        FRAMES_ROOT = Path(PREPARED['frames_root'])
        SOURCE_FPS = PREPARED['source_fps']
        print('Annotations :', ANNOTATIONS)
        print('Images :', FRAMES_ROOT)
        print('Cadence des images annotées :', SOURCE_FPS, 'Hz ; origine et hash enregistrés.')
except (ValueError, OSError, ImportError) as error:
    print('Acquisition non terminée :', error)
    print('Vérifier le message et la connexion réseau, puis relancer cette cellule. D1 reste en attente.')
''')
    add("markdown", "d1-method", """## D1 — convertir des clips réels

La configuration produit au plus 200 candidats. Elle laisse les seuils non calibrés.
La rampe initiale est ` .:+*#@`, la grille 64×32, la palette 64 couleurs, la
cadence 12 Hz et la longueur 48 images. Vérifier les noms des catégories réservées
dans le snapshot choisi. Les vidéos trop courtes sont rejetées ; les répétitions
dues au rééchantillonnage restent tracées.
La conversion CPU peut prendre plusieurs minutes. Un message revient toutes les
20 secondes et le compteur avance à chaque lot sauvegardé (25 clips par défaut).
Une cellule occupée sans exception n'est pas nécessairement bloquée.
""")
    add("code", "convert-v5", '''from ascii_flux.factory import build_dataset, file_sha256
manifest = None
if globals().get('PREPARED') is None:
    print('D1 en attente : exécuter d’abord la cellule « Télécharger YouTube-VIS avant D1 ».')
else:
    source = PREPARED['source']
    config = json.loads((WORK/'configs/d1.json').read_text(encoding='utf-8'))
    try:
        if (DATASET/'manifest.json').exists():
            from ascii_flux.training import load_manifest
            from ascii_flux.offline import ConverterConfig, converter_identity
            previous, _ = load_manifest(DATASET/'manifest.json')
            if (previous['annotation_sha256'] != file_sha256(ANNOTATIONS) or
                previous['source'] != source or previous['config'] != config or
                previous['converter'] != converter_identity(ConverterConfig(**config['converter']))):
                raise ValueError('Ce run correspond à une autre source/configuration. Choisir un nouveau chemin DATASET dans la cellule des chemins.')
            manifest = previous
            print('Run D1 existant vérifié et réutilisé :', DATASET)
        else:
            from ascii_flux.progress import dataset_progress
            with dataset_progress(DATASET, config['max_clips']):
                manifest = build_dataset(ANNOTATIONS, FRAMES_ROOT, source, config, DATASET, SOURCE_FPS)
        summary = {key: value for key, value in manifest['report'].items() if key != 'source'}
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print('Provenance complète conservée dans :', DATASET/'manifest.json')
    except (ValueError, OSError) as error:
        print('D1 non lancé :', error)
''')
    add("code", "review-v5", '''# Cette cellule peut être relancée seule : elle recharge le corpus depuis le disque.
import importlib, sys
from pathlib import Path
import ascii_flux.review as review_tools
from ascii_flux.acquisition import default_data_root
from IPython.display import FileLink, display
review_tools = importlib.reload(review_tools)
if 'DATASET' not in globals():
    DATASET = default_data_root(Path.cwd())/'v5/d1-public-001'
if not (Path(DATASET)/'manifest.json').is_file():
    print('Aucun corpus terminé trouvé dans', DATASET)
    print('Terminer D1 ou corriger DATASET. La revue lit le disque et ne dépend pas d’une variable manifest en mémoire.')
else:
    DATASET = Path(DATASET)
    manifest = review_tools.load_review_manifest(DATASET)
    REVIEW_HTML = DATASET.parent/(DATASET.name+'-review.html')
    review_tools.review_page(DATASET/'manifest.json', REVIEW_HTML)
    print(f"Lecteur prêt : {len(manifest['clips'])} clips. Taille : {REVIEW_HTML.stat().st_size/1e6:.2f} Mo.")
    print('Ouvrir ce fichier HTML directement dans un navigateur :', REVIEW_HTML.resolve())
    try:
        link_path = REVIEW_HTML.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        link_path = REVIEW_HTML.resolve()
    display(FileLink(str(link_path)))
    if 'google.colab' in sys.modules:
        from google.colab import files
        files.download(str(REVIEW_HTML))
    print('Le lecteur exporte reviews.json. Le conserver dans le répertoire du corpus correspondant.')
''')
    add("markdown", "human-gate-v5", """## Revue et calibration humaines

Observer forme, masque, couleurs et scintillement ; renseigner le relecteur et les
décisions, puis exporter `reviews.json`. Fixer les seuils de qualité à partir de
cette revue, consigner la méthode et le compromis fidélité/scintillement. Créer une
nouvelle version de configuration et un nouveau dataset, puis revoir les candidats.
Ne pas modifier le manifest d'un corpus existant.

Pour les cellules suivantes, `TRAIN_DATASET` doit désigner ce nouveau corpus.
L'entraînement refuse les seuils non calibrés ou l'absence de train/val acceptés.
Le corpus D1 n'établit pas à lui seul un corpus d'entraînement suffisant.
""")
    add("code", "embed-v5", '''TRAIN_DATASET = DATA_ROOT/'v5/d1-public-002'
REVIEWS = TRAIN_DATASET/'reviews.json'
from ascii_flux.text import cache_embeddings
EMBEDDINGS_READY = False
if not (TRAIN_DATASET/'manifest.json').is_file() or not REVIEWS.is_file():
    print('F1 en attente : terminer D1, la calibration et la revue du corpus TRAIN_DATASET avant l’encodage.')
else:
    training_manifest = json.loads((TRAIN_DATASET/'manifest.json').read_text(encoding='utf-8'))
    if not training_manifest['config']['quality'].get('calibrated'):
        print('F1 en attente : les seuils du corpus choisi doivent être calibrés après revue D1.')
    else:
        cache_embeddings(TRAIN_DATASET/'manifest.json', TRAIN_DATASET/'text.npz', DEVICE)
        EMBEDDINGS_READY = True
''')
    add("code", "train-v5", '''from ascii_flux.training import TrainConfig, train
settings = TrainConfig(**json.loads((WORK/'configs/train.json').read_text()))
# Pour reprendre : RESUME = RUN/'latest.pt'. Même configuration, corpus et revue.
RESUME = None
if not globals().get('EMBEDDINGS_READY'):
    print('Entraînement en attente : préparer le corpus validé et ses embeddings dans la cellule précédente.')
else:
    model = train(TRAIN_DATASET/'manifest.json', TRAIN_DATASET/'text.npz', RUN,
                  config=settings, reviews_path=REVIEWS, device=DEVICE, resume=RESUME)
''')
    add("code", "measure-v5", '''from ascii_flux.training import load_inference
from ascii_flux.text import TextEncoder
from ascii_flux.evaluation import measure_rollout, export_onnx
from ascii_flux.storage import write_json
if not (RUN/'best.pt').is_file():
    print('Mesures en attente : aucun checkpoint entraîné dans', RUN)
else:
    model, provenance = load_inference(RUN/'best.pt', DEVICE)
    encoder = TextEncoder(DEVICE)
    vectors = encoder.encode(['cat', 'dog'])
    diagnostics = measure_rollout(model, vectors[0], steps=600, switch_embedding=vectors[1])
    diagnostics['checkpoint_provenance'] = provenance
    write_json(RUN/'rollout.json', diagnostics)
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    export_onnx(model, RUN/'step.onnx')
''')
    add("markdown", "limits-v5", """## Lecture des résultats et suite

La validation utilise l'entropie croisée sur jetons, pas une reconnaissance humaine.
Un état fini et figé est un échec de dynamique. Les diagnostics ne mesurent pas le
temps d'apparition d'une forme lisible, la transition sémantique, les langues ou
la stabilité visuelle sur dix minutes. Organiser ces évaluations sur vidéos
réservées et concepts réservés, puis comparer les autres architectures.

L'export ONNX est contrôlé numériquement ; il ne prouve pas les performances dans
un navigateur. Le prototype local `python -m ascii_flux serve --checkpoint …/best.pt`
utilise Python pour le calcul et un canvas pour afficher les jetons. Le portage
WebGPU/WASM, le corpus réel validé et tous les jalons restent à réaliser.
""")
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "colab": {"name": "ASCII_Flux_v5_D1_F1.ipynb"},
        "ascii_flux": {"source_sha256": source_hash, "status": "unvalidated research implementation"}}, "cells": cells}
    write_json(ROOT/"ASCII_Flux_v5_D1_F1.ipynb", notebook)
    print("Notebook autonome généré :", source_hash)


if __name__ == "__main__":
    main()

