# ASCII Flux v5 — modèle de flux ASCII natif : spécification et consignes de développement

## Objectif et décision

Créer un **modèle de flux temps réel qui génère directement de l'ASCII art animé**
à partir d'une consigne. L'état du modèle est la grille de caractères elle-même :
à chaque pas, il fait évoluer la grille selon le texte. Taper ou modifier la consigne
change le conditionnement ; la grille continue d'évoluer vers le nouveau concept,
sans redémarrer.

Décision du 31 octobre 2026 : la v4.2 (texte → image Supra → détourage → rig → glyphes)
est abandonnée comme chemin de génération. Elle produisait des images converties
en ASCII ; rien dans le système ne raisonnait en caractères. La v5 apprend
**dans l'espace des glyphes** sur un dataset obtenu en convertissant des vidéos
labellisées en ASCII.

Distinction à respecter partout : le convertisseur vidéo → ASCII est une **usine à
données hors ligne**, jamais un composant d'inférence. Le modèle apprend sur des
grilles de glyphes, comme un générateur d'images apprend sur des photos sans les
coller. Ne pas présenter le modèle comme « sans aucune image » à l'entraînement :
son savoir visuel vient de vidéos converties.

Décision du 31 octobre 2026 (après échec visuel du modèle masqué sur 793 labels
vidéo, verdict utilisateur « rien n'est lisible ») : les **formes** viennent d'un
professeur texte → image utilisé **hors ligne seulement** (`ascii_flux/teacher.py` :
SD-Turbo + décodeur TAESD, détourage BiRefNet_lite), converti par un convertisseur
à glyphes de contour (`ascii_flux/structure.py`). Ce corpus est tagué
`synthetic: teacher-t2i` et annoncé comme tel ; les vidéos réelles restent la source
du **mouvement**. Le modèle cible devient un transformeur masqué avec attention
croisée sur les jetons texte, en cascade silhouette → glyphes. Aucun générateur
d'images à l'inférence : le contrat d'inférence ci-dessous reste inchangé.

Statut au 1er novembre 2026 : les corpus locaux et des poids expérimentaux v5
existent. Le run cascade-001 a atteint 40 000 pas ; son meilleur checkpoint est
au pas 29 000. La reconnaissance et le mouvement appris restent à valider.
Les choix marqués *initial* restent des points de départ à mesurer.

## Contrat d'inférence

Interdit à l'inférence : générateur d'images, VAE, pixels intermédiaires,
détourage, rig géométrique, conversion image → ASCII. Une sortie obtenue en rendant
puis en convertissant une image n'est pas une sortie v5.

Entrées d'un pas : état caché de la grille, embedding du texte, bruit. Sorties :
un glyphe et une couleur par cellule, lus depuis l'état. Le rendu affiche ces
jetons tels quels (texte coloré sur fond noir, lueur facultative côté affichage).

- L'encodeur de texte tourne **une fois par révision** de la consigne (debounce
  ~250 ms), jamais à chaque image.
- Un changement de consigne remplace le conditionnement sans réinitialiser l'état.
  Une réinitialisation explicite reste disponible. Ne pas appeler « morphing »
  une transition dont la continuité n'a pas été mesurée.
- Le pas du modèle tourne à cadence fixe, indépendamment du calcul du texte.
- Cibles, non acquises : ≥24 pas/s sur une grille 64×32 en CPU portable ou
  navigateur (WebGPU/WASM), forme lisible en <1 s après une nouvelle consigne,
  aucune dérive visible sur 10 minutes de fonctionnement continu.

## Représentation ASCII

Ces choix font partie du **contrat de dataset et de checkpoint** : tout changement
crée une nouvelle version de dataset et invalide les poids.

- Grille *initiale* : 64 colonnes × 32 lignes ; rapport de cellule hauteur/largeur
  enregistré dans les métadonnées. Fond noir, sujet seul.
- Vocabulaire de glyphes fixe, ordonné et versionné. *Initial* : rampe de densité
  ` .:+*#@` plus un jeu de glyphes de contour à mesurer (`/\|-_()` etc.). L'espace
  est le glyphe vide.
- Couleur : palette fixe quantifiée, *initiale* 64 couleurs, versionnée. Une
  cellule vide n'a pas de couleur. Une variante monochrome reste possible.
- Jeton de sortie : deux têtes (glyphe, couleur), pas un produit cartésien.
- Cadence du dataset *initiale* : 12 images/s, clips de 48 images. Le modèle peut
  tourner plus vite que la cadence des données ; ne pas annoncer plus d'images
  distinctes par seconde que celles réellement produites.

## Dataset : vidéos labellisées → ASCII

### Rôle et principe

La qualité du convertisseur est le **plafond du modèle**. Si un chat converti
n'est pas reconnaissable, aucun modèle ne fera mieux. Le convertisseur est
validé visuellement avant tout passage à l'échelle.

### Sources, par priorité

1. **YouTube-VIS** : ~40 catégories (animaux, véhicules, personnes…), masque de
   chaque instance sur chaque image annotée. Label et détourage fournis : point
   de départ du jalon D1.
2. **Animal Kingdom** : comportements animaux annotés.
3. **Kinetics-700** : actions humaines (labels d'action, pas de masques).
4. **OpenVid-1M** : légendes libres ; demande détourage et réécriture des légendes.

Vérifier et consigner la licence de chaque source avant téléchargement ; la plupart
sont réservées à la recherche. Conserver identifiant, révision/snapshot, URL et
conditions par clip. Ne pas mélanger les conditions entre sources.

### Pipeline du convertisseur

1. Décodage, rééchantillonnage à la cadence du dataset, découpe en clips.
2. Masque du sujet : masque fourni par la source en priorité ; sinon propagation
   SAM 2 depuis une boîte/point initial ; signaler la méthode par clip.
3. Recadrage qui suit le sujet (boîte lissée dans le temps, marge fixe, rapport
   d'aspect de la grille) pour annuler le mouvement de caméra. Conserver la
   trajectoire de la boîte : le déplacement global du sujet peut être réinjecté
   ou appris séparément, ne pas le perdre silencieusement.
4. Fond mis à noir hors masque.
5. Ton et couleur par cellule, puis glyphe. Le rendu couleur v4.2 (`tone_map`,
   `color_cells`, rampe, saturation) est le point de départ.
6. **Stabilisation temporelle** : hystérésis sur les niveaux de glyphe, transport
   par le flux optique ou par le masque. Sans elle, le modèle apprend le
   scintillement du convertisseur.
7. Quantification sur la palette versionnée.
8. Score de qualité et rejet motivé (voir ci-dessous).
9. Écriture des jetons et des métadonnées.

### Labels

- Garder la légende/le label d'origine **et** une forme normalisée « sujet + action »
  courte. Une réécriture par LLM est hors ligne, journalisée (modèle, révision,
  prompt) et contrôlée sur échantillon.
- Une légende de scène (« un homme dans une cuisine explique… ») doit être réduite
  au sujet détouré ; sinon rejeter.
- Aucun dictionnaire de mots ni routage par sous-chaînes à l'inférence ; c'est
  l'encodeur de texte qui porte le sens.

### Qualité et rejets

Mesurer par clip : part de la grille occupée, taille du sujet en cellules,
taux de glyphes changés par image (brut et compensé), mouvement de caméra
résiduel, part d'images avec masque. Seuils *initiaux* à fixer sur l'échantillon
D1. Conserver chaque motif de rejet et publier le rendement par source.
Une revue humaine sur échantillon accompagne chaque nouvelle version du
convertisseur ; les métriques automatiques ne la remplacent pas.

### Stockage et découpage

- Par clip : `glyphs` uint8 [T, R, C], `colors` uint8 [T, R, C], `box` [T, 4],
  métadonnées JSON (source, id, licence, cadence, label, légende, méthode de
  masque, version et hash du convertisseur, métriques qualité).
- Fichiers par shards ; écriture temporaire, flush/fsync et remplacement atomique.
- Découpage train/val/test **par vidéo source**, jamais par clip : deux clips
  d'une même vidéo vont dans la même partition. Réserver des catégories entières
  pour mesurer la généralisation.
- Aucune donnée synthétique (formes procédurales, ancien pipeline v4.2, Lottie)
  substituée au corpus principal sans être taguée et annoncée.

## Modèle de flux

### Architecture *initiale* (à comparer)

Automate cellulaire neuronal conditionné par le texte :

- État caché par cellule (canaux *initiaux* : 32–64), lecture glyphe/couleur
  par deux têtes linéaires.
- Perception locale (convolutions 3×3) **plus** un contexte global basse
  résolution (pooling ou attention sur une grille réduite) : un automate purement
  local coordonne mal une grande forme en peu de pas.
- Conditionnement texte par FiLM à chaque pas. Abandon du texte ~10 % pour la
  guidance sans classifieur.
- Mise à jour stochastique d'une partie des cellules par pas.
- Taille cible : 1–10 M paramètres, exportable ONNX.

Alternatives à évaluer avec les mêmes données et métriques : diffusion masquée /
flow matching discret sur jetons, petit transformeur spatio-temporel
autorégressif. Retenir l'architecture sur mesures, pas par préférence.

### Objectifs d'apprentissage

- **Émergence** : depuis une grille vide ou bruitée, atteindre des images du
  clip du concept demandé.
- **Dynamique** : prédire l'image suivante, en teacher forcing puis en déroulés
  multi-pas sur l'état du modèle (sinon dérive à l'inférence).
- **Changement de concept** : partir de l'état d'un concept A, conditionner sur B,
  atteindre B. Mesurer le temps de transition.
- Pertes en entropie croisée sur glyphes et couleurs. Pas de perte en pixels.
- Stabilité : déroulés longs (≥500 pas) sans effondrement, explosion ni
  immobilisation ; une grille figée n'est pas une stabilité réussie.

### Texte

Tokenisation → embedding → génération. Encodeur *initial* : SigLIP 2
`google/siglip2-base-patch16-224`, révision `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`,
gelé, texte normalisé NFC/minuscules, accents conservés, 64 jetons maximum.
Un encodeur produit un vecteur pour n'importe quel mot ; cela ne garantit pas
que le modèle sache le dessiner. Hors des concepts du corpus, annoncer
l'échec plutôt qu'une compréhension ouverte. Le multilingue dépend de l'encodeur
et des légendes ; le mesurer par langue.

## Validation et jalons

Métriques :

- Reconnaissance humaine en choix forcé par concept, sur concepts vus et réservés.
- Classifieur entraîné sur clips ASCII réservés comme proxy automatique, jamais
  présenté comme jugement humain.
- Glyphes changés par seconde, couture et dérive sur déroulés longs.
- Latence : pas du modèle, encodage du texte, temps jusqu'à une forme lisible,
  temps de transition entre concepts.

Jalons :

- **D1** : convertisseur sur ~200 clips YouTube-VIS, revue visuelle, seuils de
  qualité fixés, rendement publié.
- **D2** : dataset à l'échelle, rejets et licences documentés.
- **F1** : modèle sur les catégories YouTube-VIS, reconnaissable par des humains
  sur clips réservés.
- **F2** : démo navigateur temps réel : saisie → grille qui évolue en direct.
- **F3** : élargissement des sources et du vocabulaire, tests sur concepts réservés.

Les cibles (≥24 pas/s, <1 s, reconnaissance, scintillement) restent des objectifs
tant qu'elles ne sont pas mesurées.

## Héritage v4.2

Depuis le nettoyage du 1er novembre 2026, la v4.2 et son notebook sont retirés du
projet actif. Leur référence locale est conservée dans le dossier ignoré
`data/local-archives/2026-11-01/`. Ne pas les réintroduire dans le chemin v5.

- **Réutilisé dans le convertisseur** : rendu couleur (ton, rampe, saturation,
  hystérésis), lecteur HTML/canvas de l'interface, conventions de provenance.
- **Hors du chemin v5** : Supra2-IMG, Flan-T5, VAE, soft prompt, banque OpenMoji,
  GrabCut, MLS rigide, archétypes J0, M2 Fourier, MMLottie. Ils peuvent fournir un
  sous-ensemble de données tagué, jamais le corpus principal implicite.
- Leçons conservées : aucun routage par dictionnaire, aucune cosinus présentée
  comme probabilité, publier les compromis (scintillement contre fidélité),
  tests analytiques nommés comme tels.

Environnement : `transformers<5` exige `huggingface-hub<1`, donc `gradio<6.17`.
Installer les dépendances en début de session puis redémarrer le noyau ; ne jamais
installer un paquet au milieu d'un noyau qui a déjà importé ses dépendances.
Cairo n'est nécessaire qu'au rendu Lottie.

## Règles de travail

Notebooks autonomes pour Colab, provenance et rejets visibles. Préserver les
identités des fichiers lors des mises à jour. Sauvegardes atomiques ; reprise
avec poids, optimiseur, meilleur état et états aléatoires. Une démonstration sur
fixture ou données de synthèse est nommée comme telle et ne prouve ni génération
ni compréhension.

## Sources et conditions

- YouTube-VIS : https://arxiv.org/abs/1905.04804
- Animal Kingdom : https://openaccess.thecvf.com/content/CVPR2022/html/Ng_Animal_Kingdom_A_Large_and_Diverse_Dataset_for_Animal_Behavior_CVPR_2022_paper.html
- Kinetics-700 : https://arxiv.org/abs/1907.06987
- OpenVid-1M : https://arxiv.org/abs/2407.02371
- Growing Neural Cellular Automata (Mordvintsev et al., Distill 2020) : https://distill.pub/2020/growing-ca/
- SigLIP 2 : https://arxiv.org/abs/2502.14786
- SAM 2 : https://arxiv.org/abs/2408.00714

Licences à vérifier et consigner par source avant usage. Les choix d'architecture
et d'objectifs de ce document sont des décisions d'implémentation à évaluer, pas
des résultats attribués à ces publications.
