# Poids du modèle

Les fichiers `.pt` ne sont pas dans Git. Leur identité (SHA-256, taille, pas,
configuration, corpus, code) est dans [`weights/manifest.json`](weights/manifest.json).
Vérifier un fichier reçu avec `Get-FileHash best.pt -Algorithm SHA256`, puis le
charger avec `torch.load(..., weights_only=True)`. Aucun de ces poids n'est validé
par une reconnaissance humaine en choix forcé.

## `cascade-001` — poids actuels (tag `weights-cascade-001`)

`data/runs/cascade-001/best.pt`, SHA-256
`0aa1d4dc89e6e0a88cb7678f73963d6099deb67b526533452873f71f998e6ceb`, 8 851 326
paramètres, état EMA du pas 29 000. Formes statiques : **aucun mouvement appris**.

### Comment ils ont été générés

**1. Vocabulaire.** `ascii_flux/shape_corpus.py` construit 11 298 mots concrets,
dans cet ordre : les 1 000 classes ImageNet-1k, les 1 196 catégories LV-VIS, puis
les synsets ImageNet-12k (listes et lemmes fournis avec `timm`). Chaque mot est
affecté par hachage SHA-256 à `train` (92 %), `val` (3 %) ou `test_unseen` (5 %).
Les mots `val` et `test_unseen` ne sont jamais montrés à l'entraînement.

**2. Professeur image, hors ligne seulement** (`ascii_flux/teacher.py`).
SD-Turbo (révision `b261bac6`, 2 pas, guidance 0, 512 px, décodeur TAESD) dessine
chaque mot avec deux gabarits alternés : « a toy figurine of a {mot}, full body,
standing on a white table, white background, product photo » et « a simple flat
illustration of a {mot}, full body, side view, white background ». Graines
déterministes dérivées du hachage du mot. BiRefNet_lite (révision `aa62cd87`,
MIT) détoure le sujet ; sa convolution déformable est remplacée par
`ascii_flux/deform.py` pour tourner sur la Radeon. Rejets : sujet touchant le
bord, plusieurs sujets, sujet trop petit, grille presque vide. Jusqu'à
8 tentatives par mot, 3 formes gardées au plus.

**3. Conversion en glyphes** (`ascii_flux/structure.py`). Le sujet est cadré en
conservant ses proportions dans la grille 64×32 (cellule 2:1). Le contour reçoit
des traits orientés selon le gradient du masque (`| / \ - _`), les arêtes
internes fortes aussi, l'intérieur une rampe légère ` .:+*#@` ; couleur moyenne
par cellule quantifiée sur la palette RGB 4×4×4. Contrat
`ascii-v5-structure-rgb64-1`. La planche de contrôle a été validée visuellement
par l'utilisateur avant le passage à l'échelle. Tout le corpus est tagué
`synthetic: teacher-t2i`.

**4. Corpus figé.** L'entraînement a utilisé les 76 premiers shards de
`data/v5/shapes-001` : 6 883 mots d'entraînement, 225 mots de validation,
21 675 grilles (empreinte `0426202b…`).

**5. Texte.** SigLIP 2 base, révision `75de2d55`, gelé. Pour chaque mot,
« {mot} » et « a {mot} » sont encodés ; le modèle reçoit les 16 premiers jetons
de `last_hidden_state`. Aucun dictionnaire ni routage : l'encodeur porte le sens.

**6. Modèle** (`ascii_flux/cascade.py`). Transformeur masqué à la MaskGIT/Muse,
dimension 256, 8 blocs (auto-attention + attention croisée sur les jetons texte),
patchs de 2×2 cellules. Le même réseau joue deux rôles : décider la silhouette
(patchs occupés), puis remplir glyphes et couleurs à l'intérieur. Texte retiré
10 % du temps pour la guidance sans classifieur (3 à l'échantillonnage).

**7. Entraînement** (`ascii_flux/cascade_training.py`, `scripts/train_cascade.py`).
40 000 pas, lots de 32, AdamW (lr 4e-4, échauffement 1 000 pas, décroissance
cosinus, weight decay 0,05), FP16, EMA 0,999, miroir horizontal des grilles avec
inversion des traits `/`↔`\`, graine 20261031. Pertes : entropie croisée sur la
silhouette masquée, puis sur glyphes et couleurs masqués. Validation tous les
1 000 pas sur les mots jamais entraînés ; `best.pt` garde l'état EMA du meilleur
pas (29 000). Radeon RX 9060 XT 8 Go, PyTorch 2.14.0 ROCm 10.1.0, environ 5 h.

### Reproduire

```powershell
.\.venv-amd\Scripts\python.exe -m pip install -e ".[train,teacher]"
.\.venv-amd\Scripts\python.exe scripts/build_shape_corpus.py --output data/v5/shapes-001
# Les 76 premiers shards : le cache de jetons fixe ce nombre au lancement.
.\.venv-amd\Scripts\python.exe scripts/train_cascade.py --output data/runs/cascade-001
```

La génération dépend du matériel et des noyaux GPU : un corpus régénéré n'est
pas garanti identique bit à bit, d'où l'empreinte du corpus dans le manifest.
`cascade.py` a été réorganisé après l'entraînement (échantillonnage découpé en
pas pour l'interface en direct) ; les paramètres sont inchangés et ces poids se
chargent avec le code du tag. La source exacte au moment de l'entraînement est
antérieure au premier commit Git ; son empreinte figure dans le checkpoint.

### Ce qu'ils savent faire, ce qu'ils ne savent pas faire

- Silhouettes distinctes par mot, y compris pour une partie des mots jamais
  entraînés (lecture de l'auteur : parapluie, ara, chat, léopard des neiges, husky).
- Échecs connus : chimpanzé ; détails fins (cornes, touches, roues) ; quelques
  fragments isolés ; intérieur chargé de `:`.
- Pas de mouvement ; français non mesuré ; pas d'annonce « concept inconnu ».
- Conditions : le corpus dérive de SD-Turbo (licence de recherche non commerciale
  de Stability AI) ; usage de recherche non commercial.

## Poids antérieurs (échecs documentés)

Listés dans le manifest avec leur SHA-256, pour traçabilité et comparaison :
`native-million-001` (NCA), `masked-six-001`, `causal-six-001`, `discrete-six-001`,
`masked-six-capacity-001` (diagnostic de mémorisation) et `masked-open-001`.
Ils utilisent l'ancien contrat `ascii-v5-ramp-rgb64-1` et des corpus vidéo ;
voir [BILAN_V5.md](../BILAN_V5.md). Ils ne sont pas tagués.
