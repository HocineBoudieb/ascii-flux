"""Standalone local-GPU D2 notebook, carrying the same audited Python sources."""
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ascii_flux.storage import write_json
from build_notebook import main as build_d1

ROOT = Path(__file__).resolve().parents[1]


def main():
    build_d1()
    base = json.loads((ROOT/"ASCII_Flux_v5_D1_F1.ipynb").read_text(encoding="utf-8"))
    cells = []
    def add(kind, key, text):
        cell = {"cell_type": kind, "id": key, "metadata": {}, "source": text.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    add("markdown", "intro-d2", """# ASCII Flux v5 — un million de grilles et entraînement local

Sélectionner le noyau **ASCII Flux - Radeon GPU**, installé sur ce PC, puis
exécuter les cellules. Le premier bootstrap peut demander un redémarrage ;
reprendre alors à la cellule des chemins. Ne pas réinstaller de dépendances
dans une session qui les a déjà importées.

L'objectif est un million de grilles 64×32 dans des clips de 48 images,
avec des vidéos réelles YouTube-VIS et LV-VIS. Les images annotées sont à
6 Hz / 5 Hz, rééchantillonnées à 12 Hz ; les versions miroir sont des
augmentations étiquetées. Un million de grilles stockées ne signifie donc
pas un million d'observations indépendantes. Les trois compteurs sont publiés.

Les conditions de chaque source restent enregistrées séparément : usage de
recherche non commerciale ; LV-VIS CC BY-NC-SA 4.0. La validation visuelle D1
consignée concerne le convertisseur sur échantillon, pas tous les clips ni
la reconnaissance du modèle entraîné. Cette dernière reste à mesurer.

**État des essais locaux du 31 octobre :** le NCA après 20 000 pas échoue au
contrôle visuel. Un million de grilles et une perte en baisse ne valident pas F1.
Des essais équilibrés sur six concepts comparent aussi un modèle masqué et
un transformeur causal natif ; leurs checkpoints sont séparés et expérimentaux.
La mémorisation de six images réelles est un diagnostic de capacité seulement.
Le test de diffusion catégorielle a également échoué au contrôle utilisateur :
aucun sujet reconnu sur les six générations présentées. Les références réelles
« cheval » ont été jugées majoritairement lisibles, séparément des sorties du modèle.

La conversion est reprenable par shards. Elle produit un snapshot immuable,
puis les vrais embeddings du texte et un NCA natif de 1,29 M paramètres.
L'apprentissage utilise des pertes sur glyphes et couleurs, avec FP16 et
des lots de 8 clips testés sur la Radeon de 8 Go. Le rendu d'inférence
n'utilise ni pixels intermédiaires, ni convertisseur vidéo.
""")
    bootstrap = copy.deepcopy(next(c for c in base["cells"] if c["id"] == "bootstrap-v5"))
    bootstrap["id"] = "bootstrap-d2"
    cells.append(bootstrap)
    add("code", "paths-d2", '''import json, sys
from pathlib import Path
WORK = Path('/content/ascii_flux_v5') if Path('/content').exists() else Path.cwd()/'ascii_flux_v5_colab'
sys.path.insert(0, str(WORK))
from ascii_flux.acquisition import default_data_root
DATA_ROOT = default_data_root(WORK.parent)
OUTPUT = DATA_ROOT/'v5/million-001'
print('Données :', DATA_ROOT)
print('Rapport :', OUTPUT/'progress.json')
import torch
if not torch.cuda.is_available():
    raise RuntimeError('Choisir le noyau « ASCII Flux - Radeon GPU ». Aucun gros entraînement CPU n’est lancé.')
print('GPU :', torch.cuda.get_device_name(0), '| PyTorch', torch.__version__)
''')
    add("markdown", "run-help-d2", """La cellule suivante enchaîne acquisition, conversion, indexation et entraînement.
Les téléchargements et les lots terminés sont réutilisés. Sur ce PC, le travail
peut déjà tourner en arrière-plan : vérifier le rapport ci-dessous avant de lancer
un second processus. Les fichiers `latest.pt` et `best.pt` apparaissent dans
`data/runs/native-million-001` pendant l'apprentissage. Une interruption permet
la reprise ; conserver le PC éveillé pendant le calcul.
""")
    add("code", "run-d2", '''from ascii_flux.pipeline import run_million
run_million(WORK, DATA_ROOT, workers=4)
''')
    add("code", "status-d2", '''# Cette cellule peut aussi être lancée seule après réouverture du notebook.
import json
from pathlib import Path
root = globals().get('OUTPUT', Path.cwd()/'data/v5/million-001')
for name in ('pipeline-status.json', 'progress.json'):
    path = root/name
    if path.exists():
        report = json.loads(path.read_text(encoding='utf-8'))
        keys = ('phase', 'source', 'clips', 'original_clips', 'augmented_clips', 'grid_slots',
                'distinct_source_object_observations', 'distinct_token_grids', 'error')
        print(name, json.dumps({k: report[k] for k in keys if k in report}, ensure_ascii=False, indent=2))
''')
    add("markdown", "causal-help-d2", """## Essai causal natif sur six concepts

Optionnel et désactivé pour « Tout exécuter ». Utilise le corpus réel existant,
équilibre les concepts et les vidéos, et conserve des vidéos de validation.
Ce test ne prouve pas la reconnaissance. Le modèle produit directement deux
distributions par cellule (glyphe, couleur), sans codec ou image intermédiaire.
Les blocs regroupent des cellules pour l'attention. La génération est séquentielle ;
24 images complètes distinctes par seconde ne sont pas acquises.
""")
    add("code", "causal-d2", '''RUN_CAUSAL_EXPERIMENT = False
if RUN_CAUSAL_EXPERIMENT:
    from ascii_flux.causal_training import CausalTrainConfig, train_causal
    run = DATA_ROOT/'runs/causal-six-001'
    config = CausalTrainConfig(steps=8000)
    train_causal(OUTPUT/'manifest-million-001.json', OUTPUT/'text-million-001.npz',
                 OUTPUT/'reviews-million-001.json', run, config,
                 resume=run/'latest.pt' if (run/'latest.pt').exists() else None)
else:
    print('Essai causal désactivé. F1 reste à valider sur des générations reconnaissables.')
''')
    notebook = {"cells": cells, "metadata": copy.deepcopy(base["metadata"]), "nbformat": 4, "nbformat_minor": 5}
    notebook["metadata"]["kernelspec"] = {"display_name": "ASCII Flux - Radeon GPU", "language": "python", "name": "ascii-flux-amd"}
    write_json(ROOT/"ASCII_Flux_v5_D2_Million_GPU.ipynb", notebook)


if __name__ == "__main__":
    main()
