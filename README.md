# ASCII Flux v5

Implémentation initiale de la spécification [AGENTS.md](AGENTS.md), au 31 octobre 2026.
La v4.2 et ses fichiers sont conservés sans modification. Le convertisseur D1
a produit 198 clips réels et reçu une validation visuelle globale de l'utilisateur.
Des seuils initiaux en conservent 150 pour un premier essai GPU avec de vrais
embeddings SigLIP. Ces premiers checkpoints vérifient la chaîne d'apprentissage ;
la reconnaissance du modèle, la stabilité et les performances cibles restent à mesurer.
Le corpus D2 contient 1 138 080 grilles dans 23 710 clips, dont 988 862 grilles
distinctes après rééchantillonnage et augmentations. Les 20 000 pas d'apprentissage
GPU sont terminés. **Le contrôle visuel a échoué : le NCA produit des taches
et se dégrade en déroulé. F1 et F2 ne sont pas validés.** Le nombre de grilles
et la baisse de perte n'établissent pas la reconnaissance.

Trois essais natifs séparés comparent maintenant une reconstruction de glyphes
masqués (3,78 M paramètres) et un transformeur causal par blocs de cellules
(4,19 M), ainsi qu'un débruitage catégoriel des glyphes (3,78 M). Ils équilibrent six concepts réels : cat, dog, horse, bird, fish,
person. Les partitions restent séparées par vidéo ; les poids et résultats
sont explicitement expérimentaux. Ils ne remplacent pas le modèle de l'interface
tant que leurs générations restent insuffisantes.
Le retour utilisateur sur les six générations du dernier essai est **« aucun »**
à la question des sujets reconnus. Le même utilisateur a jugé la majorité des
huit références réelles « cheval » lisibles : le contrôle des données et celui
du modèle sont conservés séparément. Ces retours sont qualitatifs, pas une mesure
en choix forcé. Le bilan est dans `outputs/recognition-audit-001/conclusion.json`.

## Stratégie formes : professeur hors ligne + transformeur en cascade

Après l'échec visuel des modèles entraînés sur vidéos seules, les formes viennent
d'un **professeur texte → image utilisé hors ligne** (décision consignée dans AGENTS.md) :

1. `ascii_flux/teacher.py` : SD-Turbo (2 pas, décodeur TAESD) dessine
   « a toy figurine of a {mot}, full body… » ; BiRefNet_lite (MIT, révision
   épinglée) détoure. Rejets motivés : sujet coupé par le bord, plusieurs sujets,
   sujet trop petit. `ascii_flux/deform.py` remplace la convolution déformable
   de torchvision (pas de build ROCm compatible), vérifiée contre la référence.
2. `ascii_flux/structure.py` : contour en traits orientés `/ \ | - _`, intérieur
   léger, proportions du sujet conservées. Nouveau contrat `ascii-v5-structure-rgb64-1`
   (15 glyphes) : incompatible avec les poids précédents.
3. `ascii_flux/shape_corpus.py`, `scripts/build_shape_corpus.py` : 11 298 mots
   (ImageNet-1k, LV-VIS, ImageNet-12k ; liste de timm), 3 formes gardées par mot,
   reprise par shard. Mots réservés par hachage : 5 % `test_unseen`, 3 % `val`,
   jamais entraînés. Chaque enregistrement est tagué `synthetic: teacher-t2i`.
   Planche de contrôle : `outputs/teacher-structure-probe-001/samples.png`.
4. `ascii_flux/cascade.py`, `cascade_training.py` : transformeur masqué (MaskGIT/Muse)
   à backbone partagé, attention croisée sur 16 jetons SigLIP, silhouette par
   patchs 2×2 puis glyphes et couleurs à l'intérieur, guidance sans classifieur,
   miroir horizontal des traits en augmentation.

```powershell
.\.venv-amd\Scripts\python.exe scripts/build_shape_corpus.py
.\.venv-amd\Scripts\python.exe scripts/train_cascade.py --output data/runs/cascade-001
.\.venv-amd\Scripts\python.exe scripts/preview_cascade.py --checkpoint data/runs/cascade-001/best.pt --output outputs/cascade-preview-001 --words cat horse giraffe
```

Les formes sont statiques : le mouvement (vidéos réelles) reste à ajouter.

**Interface en direct** : double-cliquer sur `Lancer_ASCII_Live.cmd` (ou
`scripts/live.py`), puis taper. Canvas plein écran, saisie en bas, aucun autre
texte. Chaque image affichée est un pas d'échantillonnage du modèle
(`ascii_flux/live.py`) : silhouette puis glyphes, environ 20 pas. Une nouvelle
consigne est encodée une fois (anti-rebond 120 ms) et redessine la silhouette
sans effacer la grille ; une case ne change que lorsque le modèle la décide.
Une fois la forme terminée, la grille reste immobile : aucune animation n'est apprise.

## Interface locale

Double-cliquer sur **`Lancer_ASCII_Flux.cmd`**, ou ouvrir
**http://127.0.0.1:8766** si le serveur tourne déjà. Le lanceur utilise le meilleur
modèle de `data/runs/native-million-001/best.pt` et le GPU Radeon.

Un grand canvas noir affiche directement les glyphes et couleurs du modèle.
La barre de saisie transmet chaque modification sans touche Entrée et sans délai
volontaire ; le calcul du texte ne remet pas la grille à zéro. Les requêtes
obsolètes sont ignorées si la frappe dépasse la vitesse de l'encodeur.
Le petit bouton à droite permet de recommencer explicitement l'animation.

Le checkpoint actuel a échoué au contrôle visuel de reconnaissance. L'interface
reste un outil de diagnostic du modèle. Aucun convertisseur d'images n'intervient
dans cette démo.

```powershell
.\.venv-amd\Scripts\python.exe scripts/launch_ui.py
```

Les essais peuvent être repris séparément, avec optimiseur et états aléatoires :

```powershell
.\.venv-amd\Scripts\python.exe scripts/train_masked.py --resume --steps 6000
.\.venv-amd\Scripts\python.exe scripts/train_causal.py --resume --steps 8000
.\.venv-amd\Scripts\python.exe scripts/train_discrete.py --resume --steps 2000
```

### Essai masqué vocabulaire ouvert (`masked-open-001`)

Les essais à six concepts ne disposaient que de 18 à 21 clips originaux par
sujet. Cet essai entraîne le même objectif masqué sur **les 793 labels** du
corpus million, pour partager les formes entre concepts :

- tirage des labels proportionnel à la racine du nombre de vidéos (`label_balance=0.5`) ;
- plusieurs formulations génériques par label (`car_(automobile)` → `car`,
  `a car`, `car moving`, `car (automobile)`), encodées par le même SigLIP gelé
  (`scripts/embed_phrases.py`). Aucun dictionnaire de synonymes ou de traduction ;
- choix du checkpoint sur les labels vus en validation seulement ;
- **gain du texte** (entropie croisée sans texte moins avec texte) publié sur
  les labels vus, sur les 23 labels de validation jamais entraînés et, une seule
  fois en fin d'essai, sur les catégories réservées. C'est un proxy automatique
  du transfert vers des mots nouveaux, pas une reconnaissance humaine.

```powershell
.\.venv-amd\Scripts\python.exe scripts/embed_phrases.py --device cuda
.\.venv-amd\Scripts\python.exe scripts/train_masked.py --open-vocabulary --phrases data/v5/million-001/phrases-million-001.npz --width 96 --batch-size 16 --steps 30000 --validate-every 1000 --output data/runs/masked-open-001
```

Ajouter `--resume` pour reprendre. Le contrôle visuel en choix forcé reste
l'étape qui décide.

Résultat du 31 octobre 2026 (30 000 pas, 50 min, meilleur état au pas 24 000) :

- Gain du texte en validation mélangée : quasi nul. Sur grille entièrement
  vide, le bon texte bat un texte d'un autre label (CE 3,17 contre 3,72 sur
  labels vus ; 3,08 contre 3,89 sur labels jamais entraînés), mais bat à peine
  l'absence de texte sur labels vus (3,17 contre 3,18).
- Échantillonnage par défaut (8 pas, guidance 2, température 0,7) :
  `outputs/masked-open-preview-001` montre la même ellipse centrée pour tous
  les concepts ; le texte ne change que couleur et densité.
- Échantillonnage 32 pas, guidance 5, température 1 :
  `outputs/masked-open-preview-002-g5-s32` différencie les silhouettes
  (girafe, catégorie réservée, avec des pattes ; flamant horizontal), mais
  chat, cheval et oiseau restent des taches. Observation qualitative de
  l'auteur du code, pas une reconnaissance humaine.

`scripts/preview_masked.py` produit aussi des planches depuis les checkpoints
causaux. Chaque contrôle conserve une copie des poids, leur hash, le pas réel,
les réglages et les jetons générés. Le cache de labels y sert uniquement à
l'évaluation ; l'interface encode réellement tout texte saisi avec SigLIP.
Le test `masked-six-capacity-001` mémorise six images réelles : il vérifie la
capacité du réseau et ne prouve aucune généralisation ni animation apprise.
L'audit `outputs/subject-audit-001` révèle notamment seulement 18 clips originaux
« cat » et 21 « horse », avec des vues partielles parmi les exemples.

## Un million de grilles sur ce PC

Ouvrir **`ASCII_Flux_v5_D2_Million_GPU.ipynb`** avec le noyau
**ASCII Flux - Radeon GPU**. Ce noyau isolé utilise PyTorch 2.14.0 / ROCm 10.1.0
et a réussi un vrai passage avant/arrière sur la Radeon RX 9060 XT de 8 Go.
Les résultats du test matériel sont dans `outputs/gpu-verification.json`.
Il n'est pas nécessaire de modifier le pilote ou de désactiver des protections Windows.

Le pipeline complet est aussi disponible en terminal, depuis ce dossier :

```powershell
.\.venv-amd\Scripts\python.exe -m ascii_flux million
```

Les mêmes commandes reprennent les lots terminés et les checkpoints. Une seule
instance du pipeline et un seul écrivain du catalogue peuvent tourner à la fois.
L'installation reproductible du noyau est dans `scripts/setup_radeon.ps1` ;
l'exécuter au début d'une session, puis sélectionner ou redémarrer le noyau.

Les sources sont YouTube-VIS 2019 et **LV-VIS train**, dont les catégories et les
masques sont fournis. Les téléchargements sont vérifiés par SHA-256 ; conditions
YouTube-VIS et CC BY-NC-SA 4.0 de LV-VIS conservées séparément par clip.
L'archive officielle LV-VIS train contient 3 076 vidéos et 16 017 objets annotés.
Les actions ne sont pas inventées : les légendes restent des catégories de sujets.

Les grilles sont en 64×32 avec 48 pas par clip, à 12 Hz. Les observations
annotées sont à 6 Hz pour YouTube-VIS et 5 Hz pour LV-VIS. Des versions miroir
sont ajoutées, explicitement étiquetées `horizontal_flip`, dans la même partition
que leur vidéo d'origine. Le seuil visé concerne **un million de grilles stockées**,
pas un million de vidéos, de clips ou d'observations indépendantes.

Le dossier `data/v5/million-001` contient :

- `catalogue.sqlite` : index des clips, sources, partitions, observations et hashes des grilles ;
- `sets/` : tableaux `uint8` compressés par shards, avec métadonnées et rejets ;
- `progress.json` : grilles stockées, observations source distinctes, grilles distinctes,
  originaux et augmentations, rendement et motifs de rejet par source ;
- `pipeline-status.json` : phase active ou erreur précise ;
- `manifest-million-001.json` : vue immuable utilisée pour l'entraînement, créée
  seulement après conversion suffisante et vérification des shards.

Le catalogue peut être consulté sans charger les grilles en mémoire :

```powershell
.\.venv-amd\Scripts\python.exe -m ascii_flux catalogue --root data/v5/million-001
```

L'apprentissage lit progressivement les shards, avec un cache borné à 64 clips.
Le réglage 8 Go utilise FP16 et des lots de 8 clips, testés à environ 2,1 Go
de mémoire GPU allouée avec tous les objectifs. L'accumulation est aussi prise en
charge : 2 clips × 4 accumulations a été vérifié. L'apprentissage comprend
les pertes d'émergence, dynamique, changement de concept et
déroulés longs. La perte sur les glyphes donne initialement un poids 4 aux
cellules occupées pour limiter la domination du fond vide ; ce choix reste
à comparer sur validation. `latest.pt` sauvegarde aussi l'optimiseur, le scaler FP16,
le meilleur état et les états aléatoires. Les augmentations et révisions du texte
font partie de la provenance du run.

Les checkpoints complets apparaissent dans `data/runs/native-million-001`.
`data/runs/gpu-real-preflight-001` est seulement le premier essai de 10 mises
à jour sur 150 clips réels ; il ne constitue pas F1. La revue humaine enregistrée
porte sur l'échantillon du convertisseur. Les décisions de rejet individuelles
restent prioritaires et aucun clip n'est faussement marqué accepté par un humain.
Les partitions utilisent les identités vidéo fournies par chaque source ; une
déduplication sémantique de vidéos réencodées entre sources n'est pas acquise.

## Avis technique

La décision de séparer l'usine à données vidéo du modèle en glyphes donne un
objectif vérifiable. Le premier investissement utile est la revue du convertisseur
à 64×32 : un corpus illisible ou scintillant ne permet pas de valider le modèle.
Le contexte global est une hypothèse raisonnable pour coordonner une forme, et
les transitions doivent être apprises sur l'état du modèle. L'automate reste une
baseline à comparer ; aucun résultat ne justifie encore sa préférence aux autres
architectures. Le nombre de paramètres ne suffit pas à garantir 24 pas/s.

Quelques précisions appliquées :

- L'état récurrent est un tenseur **par cellule de la grille de glyphes**, avec
  deux têtes de lecture ; il ne s'agit pas d'un latent d'image. L'entraînement
  ne calcule aucune perte en pixels.
- La première représentation utilise seulement la rampe ` .:+*#@`. Les contours
  demandent une comparaison visuelle et une nouvelle version s'ils sont ajoutés.
  La palette initiale est un cube RGB 4×4×4. La couleur 255 désigne l'absence
  de couleur, et n'appartient pas à la palette de 64 couleurs.
- YouTube-VIS fournit des catégories de sujets, pas des descriptions d'actions.
  On conserve le sujet seul sans inventer une action. Une supervision d'actions
  devra venir de labels réels ou d'une annotation hors ligne contrôlée.
- La cadence des **images annotées extraites** doit être fournie. Si elle est
  inférieure à 12 Hz, des images sont répétées et leurs indices conservés ;
  cela ne produit aucune nouvelle information temporelle.
- Le mouvement du fond est un **proxy** de caméra, indisponible si le fond
  manque de texture. Il n'est pas remplacé par une mesure du mouvement du sujet.
- Un déroulé de 500 pas fini ou une baisse d'entropie croisée ne prouve ni
  reconnaissance humaine, ni mouvement convaincant, ni stabilité sur dix minutes.

## Fichiers

- `ascii_flux/contract.py` : représentation immuable et empreinte de compatibilité.
- `ascii_flux/acquisition.py` : téléchargement et extraction des archives autorisées,
  détection des fichiers, reçus de provenance et réutilisation des données acquises.
- `ascii_flux/public_source.py`, `configs/ytvis2019_public.json` : profil public
  préconfiguré, adaptation des annotations réorganisées et extraction des images correspondantes.
- `ascii_flux/offline.py`, `factory.py` : usine locale YouTube-VIS → jetons.
- `ascii_flux/review.py` : lecteur de revue D1 autonome, décisions exportables.
- `ascii_flux/model.py`, `text.py`, `training.py` : baseline NCA, encodeur texte
  seul SigLIP 2 gelé, entraînement et reprise atomique.
- `ascii_flux/masked.py`, `masked_training.py` : essai de débruitage en jetons,
  inspiré de l'objectif masqué de MaskGIT, sans son codec d'image.
- `ascii_flux/autoregressive.py`, `causal_training.py` : essai causal sur des
  blocs de glyphes natifs, deux têtes par cellule, contexte de la grille précédente.
  Son export ONNX récurrent n'est pas encore validé.
- `ascii_flux/discrete.py`, `discrete_training.py` : posterior de diffusion
  catégorielle uniforme sur les glyphes, tête couleur distincte, initialisation
  depuis les poids masqués. Ce test n'a pas validé la reconnaissance.
- `ascii_flux/inference.py` : chargement des formats expérimentaux sans modifier
  l'identité du code d'entraînement NCA existant.
- `ascii_flux/runtime.py`, `player.html` : prototype navigateur avec calcul
  Python local. La cadence du modèle est indépendante du worker texte ; un
  changement de consigne ne réinitialise pas la grille.
- `ascii_flux/evaluation.py` : diagnostics ≥500 pas et export ONNX avec parité.
- `ASCII_Flux_v5_D1_F1.ipynb` : notebook Colab autonome contenant une copie de
  ces sources. À régénérer après changement avec `python scripts/build_notebook.py`.

## Installation

Python 3.11 ou plus récent. Pour Colab, utiliser la première cellule d'installation
et redémarrer le noyau avant d'importer les dépendances. En terminal :

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[train,vis,export,test]"
.\.venv\Scripts\python.exe -m pytest -q
```

Pas de Gradio, Cairo, VAE, Supra, segmentation ou conversion d'images dans
l'inférence. Les dépendances du convertisseur ne sont pas importées par le runtime.
Les bornes `transformers<5` et `huggingface-hub<1` sont conservées.

## D1 : préparer les données réelles

Lire les [conditions officielles YouTube-VIS](https://youtube-vos.org/dataset/term/)
avant de télécharger les archives depuis la [page de la source](https://youtube-vos.org/dataset/vis/).
Vérification le 31 octobre 2026 : annotations CC BY 4.0 ; données réservées à la
recherche non commerciale. Les droits des vidéos et les conditions complètes
restent applicables. Le programme ne télécharge pas des vidéos depuis YouTube.
L'étape d'acquisition télécharge les archives du dataset depuis les miroirs configurés.

### Si les données ne sont pas encore téléchargées

Le notebook contient une cellule **« Télécharger YouTube-VIS avant D1 »**, désormais
préconfigurée **sans connexion CodaLab et sans URL à renseigner** :

- [Kaggle : annotations YouTube-VIS 2019 train, version 1](https://www.kaggle.com/datasets/farnoosharefi/youtube-vis-train2019),
  environ 64 Mo. Catégories et masques COCO RLE sont réorganisés en `meta` + `sequences`.
- [Hugging Face : images YouTube-VOS 2019](https://huggingface.co/datasets/yinloonga/YouTube_VOS_2019),
  archive TAR de **9,26 Go**, révision `7a0eb80c729b0dbbceaea15950e86ed606ab21ed`.
  Ce sont les images source auxquelles se rapportent les annotations VIS ; la
  correspondance des identifiants, images, masques et dimensions est vérifiée.

Les liens de téléchargement sont épinglés dans `configs/ytvis2019_public.json`,
avec le SHA-256 de chaque archive. Les données sont acquises automatiquement par
la cellule du notebook. Régler `DOWNLOAD_DATA = False` pour reporter le téléchargement.
Le client de téléchargement s'identifie explicitement comme ASCII Flux :
l'ancien identifiant Chrome de `gdown` provoquait une réponse HTML de Kaggle,
même avec un statut HTTP 200. Le correctif est testé avec le vrai client et
le téléchargement complet des 64 197 762 octets d'annotations, SHA-256 vérifié.
Prévoir également l'espace des images extraites ; seules celles des vidéos VIS
admissibles sont extraites, et les autres fichiers YouTube-VOS restent dans le TAR.

Le format d'annotations est adapté sans réécriture des labels ou des masques.
Les vrais identifiants des répertoires vidéo servent au découpage ; les identifiants
numériques de séquence et les identifiants d'objets du miroir sont conservés.
Sur les 2 238 séquences du miroir, 1 267 ont au moins 24 images annotées à intervalle
constant et sont admissibles avant contrôle de disponibilité des images.
Les vidéos trop courtes ou avec des images omises sont journalisées dans
`mirror-conversion-report.json`. Cela évite d'inventer une cadence uniforme pour
un échantillonnage irrégulier. Toute image manquante dans le miroir est également
consignée, et la couverture finale est publiée.

Le notebook télécharge les archives, enregistre URL et SHA-256, vérifie leur
contenu, les extrait et retrouve les fichiers. Il conserve un `prepared.json`
et réutilise les acquisitions terminées. Une page HTML, une erreur de quota ou
un contenu dont le SHA-256 diffère n'est jamais accepté comme dataset. Les
conditions originales YouTube-VOS/VIS de recherche non commerciale restent
applicables ; l'hébergement public n'accorde pas de nouveaux droits. La mention
« Unknown » du miroir d'annotations et l'absence de déclaration du miroir d'images
sont conservées comme informations d'hébergement, sans remplacer ces conditions.

Le profil public 2019 utilise les images **listées dans les annotations**, à 6 Hz :
l'[article, section 4](https://arxiv.org/html/1905.04804v2#S4) décrit une annotation
sur une image toutes les cinq à 30 Hz. Les images supplémentaires d'une archive
`all_frames` sont ignorées. Ce profil ne s'applique pas à une autre version ou
à des annotations rééchantillonnées.

En local, les données vont dans `data/` à côté du notebook. Dans Colab, elles
vont dans `/content/ascii_flux_data`, ou sur Google Drive **s'il est déjà monté**.
Il n'y a plus de chemin Google Drive imposé en exécution locale. Sans acquisition
terminée, les cellules D1 et suivantes affichent l'étape à réaliser et attendent,
sans tenter de convertir ou d'entraîner sur des fichiers inexistants. Le run du
notebook est nommé `v5/d1-public-001`, distinct d'un éventuel run acquis par le portail.

Vérification réelle locale : 32 Mio du début du TAR et l'archive complète des
annotations ont permis de retrouver quatre vidéos complètes et leurs six instances.
Quatre clips ont ensuite été convertis dans `data/source-verification/real-clip-sample/`.
Ce contrôle prouve l'accès sans connexion et la compatibilité sur cet échantillon ;
il ne prouve pas la qualité visuelle D1.

L'acquisition complète a également été exécutée avec le client du notebook :
les deux archives ont le SHA-256 attendu, et les 42 897 images des 1 267 vidéos
admissibles (2 081 instances) ont été extraites et retrouvées, sans vidéo manquante
dans cette sélection. Le cache local `data/ytvis-2019-public/prepared.json` est prêt.
Relancer la cellule d'acquisition le réutilise sans nouveau téléchargement, puis
la cellule D1 peut convertir les candidats. La revue humaine reste à réaliser.

### Préparation manuelle ou autre snapshot

Renseigner le snapshot exact et l'origine des archives dans
`configs/ytvis_source.json`. Extraire localement les images et le JSON officiel
qui contient `videos`, `annotations`, `categories` et les masques d'instances.
Vérifier la cadence des images annotées pour ce snapshot. Les chemins de
`file_names` doivent être relatifs à `--frames-root`.

```powershell
python -m ascii_flux convert --annotations data/ytvis/instances_train.json --frames-root data/ytvis/JPEGImages --source configs/ytvis_source.json --config configs/d1.json --source-fps CADENCE_VERIFIEE --output data/v5/d1-001
python -m ascii_flux review --manifest data/v5/d1-001/manifest.json --output outputs/review-d1-001.html
```

La configuration limite le passage à 200 **clips complets candidats**. Les vidéos
trop courtes sont rejetées, sans boucle ou remplissage. Les catégories sont
échantillonnées en alternance avec une graine fixe. `giraffe` et `airplane` sont
réservées initialement ; adapter les noms exacts à la version choisie avant le run.
**Toute vidéo contenant une catégorie réservée va en test_unseen**, y compris
ses autres instances. Train/validation/test sont attribués par vidéo source.

La conversion CPU peut prendre plusieurs minutes. Le notebook affiche un message
toutes les 20 secondes et compte les clips sauvegardés à chaque lot (25 par défaut).
Le suivi observe les fichiers publiés sans modifier le convertisseur ni son empreinte.
Une exécution déjà en cours peut se terminer normalement avec l'ancien affichage.

Ouvrir le HTML de revue : examiner la forme, le masque et le mouvement ; exporter
`reviews.json` avec les décisions et le nom du relecteur. Les métriques sont
visibles par clip. `report.json` indique le rendement et les rejets ; aucune revue
automatique n'est assimilée à une décision humaine.

La cellule de revue recharge le manifest et vérifie les shards sur disque, même
après perte de la variable Python `manifest`. Elle n'exige aucune nouvelle
conversion. Le lecteur est un fichier HTML autonome à ouvrir directement dans
un navigateur ; dans Colab, la cellule le télécharge. Les jetons uint8 sont
compressés sans perte et seul le clip sélectionné est décompressé. Ce format
évite une sortie HTML massive dans les éditeurs de notebooks. Les décisions
peuvent être reprises par import et sont aussi conservées dans le stockage local
du navigateur lorsque celui-ci est disponible. Toujours exporter `reviews.json`.

Le run local `data/v5/d1-public-001` contient maintenant **198 clips issus de
200 candidats**, avec deux rejets `no_subject_mask`. Son lecteur est disponible
dans `data/v5/d1-public-001-review.html` (environ 4,2 Mo). La revue et la calibration
restent à effectuer ; les marqueurs `human_review_completed=false` et
`D1_complete=false` indiquent cette étape en attente, pas un échec de conversion.

Les seuils de `configs/d1.json` sont volontairement `null` et `calibrated=false`.
Les fixer après cette revue, consigner la méthode et les compromis, puis créer
une **nouvelle configuration et un nouveau run**. Les sorties sont immuables.
Faire une nouvelle revue pour chaque version du convertisseur. Le programme
ne déclare jamais automatiquement D1 terminé.

Stockage : shards TAR contenant, par clip, un NPZ sans pickle (`glyphs`, `colors`
uint8 `[T,R,C]`, `box` float32 `[T,4]`) et les métadonnées JSON. Les fichiers sont
écrits en temporaire, flush/fsync, puis publiés par remplacement atomique ; le
répertoire du run est publié seulement une fois tous les shards terminés.
Les hash des images source, des annotations, du convertisseur et des shards sont
conservés. `box` garde la trajectoire en coordonnées source, y compris le padding.
Les métriques de caméra peuvent être nulles ; un seuil requis sur une mesure
absente rejette le clip avec un motif explicite.

## F1 : encoder puis entraîner

L'encodeur gelé charge uniquement la tour texte, sans tour de vision, à la
révision spécifiée dans AGENTS.md. Normalisation NFC/minuscules, accents conservés,
64 jetons. Les embeddings sont mis en cache avant l'entraînement.
Ce checkpoint SigLIP 2 déclare une architecture `SiglipTextModel` dans son
`config.json` : le chargeur respecte cette configuration et vérifie l'absence
de poids manquants, plutôt que de construire une tour avec le mauvais type.

```powershell
python -m ascii_flux embed --manifest data/v5/d1-002/manifest.json --output data/v5/d1-002/text.npz --device cuda
python -m ascii_flux train --manifest data/v5/d1-002/manifest.json --embeddings data/v5/d1-002/text.npz --reviews data/v5/d1-002/reviews.json --config configs/train.json --device cuda --output runs/v5-001
```

L'entraînement exige des seuils déclarés calibrés et une revue humaine, par clip
ou sur échantillon du même convertisseur, avec train et validation non vides.
La portée de la revue est enregistrée dans le checkpoint. `--allow-unreviewed` est une option
de développement explicite, enregistrée dans le checkpoint. Les fixtures sont
exclues sans `--allow-fixture` et ne deviennent jamais des données de corpus réelles.

La baseline de **1 285 735 paramètres** comprend 48 canaux d'état, perception 3×3,
contexte spatial réduit, coordonnées de grille, FiLM, bruit et mises à jour
stochastiques. Les couleurs vides n'ont aucune perte. Les objectifs sont l'émergence,
la prédiction teacher forcing puis récurrente, le changement de conditionnement
sur un état conservé, et une supervision après des déroulés de 512 pas avec un
gradient limité à leur fin. Cette dernière ne suffit pas à prévenir une grille
figée ; l'évaluation la signale séparément.

`latest.pt` conserve modèle, optimiseur, meilleur modèle, historique et états
aléatoires Python/NumPy/Torch/CUDA. La reprise autorise seulement une augmentation
du nombre total de pas ; les autres paramètres et la provenance doivent correspondre.
Elle est testée à résultat identique sur CPU. Reprendre avec le même appel complété
de `--resume runs/v5-001/latest.pt`. Ce checkpoint de reprise contient des états
Python sérialisés : le charger uniquement depuis ses propres runs locaux.
`best.pt` est l'artefact d'inférence, chargé avec `weights_only=True`.

## Mesures, export et démo locale

```powershell
python -m ascii_flux evaluate --checkpoint runs/v5-001/best.pt --prompt "cat" --switch-prompt "dog" --steps 600 --output outputs/rollout.json
python -m ascii_flux export --checkpoint runs/v5-001/best.pt --output outputs/step.onnx
python -m ascii_flux serve --checkpoint runs/v5-001/best.pt --port 8766
```

Ouvrir `http://127.0.0.1:8766`. La saisie remplace le conditionnement après environ
250 ms sans remise à zéro. Une réinitialisation est explicite. Les inférences texte
anciennes sont ignorées si une nouvelle révision arrive. Le modèle vise une horloge
fixe ; les créneaux manqués sont comptés, sans prétendre tenir la cadence demandée.

Ce prototype effectue le calcul en Python local. **F2 navigateur WebGPU/WASM n'est
pas réalisé**. L'export ONNX livre un pas avec état, texte, bruit et masque de mise
à jour explicites ; la parité numérique est vérifiée avant publication. Il faudra
implémenter et mesurer un runtime navigateur pour atteindre le jalon F2.

Les diagnostics publient latence, changements de glyphes, unicité, occupation,
amplitude d'état et gel éventuel. Le temps de lisibilité, la transition sémantique,
la reconnaissance humaine et la stabilité visuelle sur dix minutes restent non
mesurés. Un modèle qui tient 500 pas avec une grille figée ne réussit pas la stabilité.

## Vérification analytique locale

```powershell
python scripts/smoke.py
```

Ce script utilise seulement des blocs procéduraux et des vecteurs texte aléatoires
tagués comme fixtures. Il vérifie conversion, stockage, entraînement, reprise,
déroulé et export si disponible. Ses artefacts sont dans `outputs/analytical-smoke`.
**Ils ne prouvent ni génération de concepts, ni compréhension, ni D1/F1.**

## Travail restant

1. Vérifier la qualité sur les nouvelles sources et affiner les seuils initiaux D1.
2. Terminer D2, entraîner et comparer les architectures sur les mêmes partitions.
3. Organiser une reconnaissance humaine en choix forcé, séparant concepts vus
   et catégories réservées ; mesurer transitions, langues et fonctionnement long.
4. Porter et chronométrer le pas ONNX dans WebGPU/WASM ; aucune cible de performance
   ou compréhension ouverte n'est acquise à ce stade.

## Sources d'implémentation consultées

- [Format et API YouTube-VIS](https://github.com/youtubevos/cocoapi/blob/master/PythonAPI/pycocotools/ytvos.py)
- [SigLIP 2, documentation Transformers 4.57](https://huggingface.co/docs/transformers/v4.57.1/model_doc/siglip2)
- [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/) : inspiration
  pour les mises à jour locales stochastiques ; l'architecture conditionnée et les
  objectifs de ce dépôt sont des hypothèses d'implémentation propres à évaluer.
- [LV-VIS officiel et conditions](https://github.com/haochenheheda/LVVIS),
  [article : images extraites à 5 Hz](https://arxiv.org/abs/2304.01715).
- [Installation officielle PyTorch AMD](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html)
  pour les roues ROCm 10.1.0 ciblant `gfx1200`. Le test de rétropropagation local
  concerne cette version récente, pas l'ancienne pile Windows 7.2 limitée à l'inférence.
