# Bilan du 1er novembre 2026 : formes professeur + cascade

**Premier modèle aux formes lisibles.** Après l'échec des essais sur vidéos
seules (ci-dessous), deux changements ont été faits ensemble, à la demande de
l'utilisateur (« la stratégie est mauvaise, inspire-toi de l'état de l'art ») :

1. **Données de forme distillées d'un professeur hors ligne** : SD-Turbo dessine
   chaque mot en figurine entière, BiRefNet_lite détoure, un convertisseur à
   traits orientés produit la grille. Corpus tagué `synthetic: teacher-t2i`.
   L'utilisateur a validé la planche de contrôle avant le passage à l'échelle.
2. **Modèle** : transformeur masqué en cascade (silhouette par patchs 2×2, puis
   glyphes et couleurs), attention croisée sur 16 jetons SigLIP, guidance sans
   classifieur. 8,85 M paramètres.

| Essai | Entraînement | Résultat observé |
|---|---:|---|
| Masqué vocabulaire ouvert, 7,7 M (`masked-open-001`) | 30 000 pas, 793 labels vidéo | même ellipse centrée pour tous les mots ; texte ne change que couleur et densité |
| **Cascade, 8,85 M (`cascade-001`)** | 40 000 pas, 6 883 mots professeur ; meilleur au pas 29 000 | silhouettes distinctes par mot ; « on voit bien une amélioration » (utilisateur) |

Sur une planche de 7 mots jamais entraînés, l'auteur du code lit parapluie, ara,
chat, léopard des neiges et husky ; tricératops et piano à queue sont approximatifs ;
chimpanzé reste informe. L'utilisateur a seulement jugé l'amélioration globale.
Ce ne sont pas des mesures en choix forcé.

Le proxy automatique (gain du texte sur l'entropie croisée de silhouette) reste
faible : 0,011 en validation, 0,022 sur test_unseen. Il ne reflète pas la
différence visible entre mots ; ne pas l'utiliser seul pour décider.

Une interface plein écran en direct (`Lancer_ASCII_Live.cmd`) montre chaque pas
d'échantillonnage ; un changement de consigne redessine la silhouette sans effacer
la grille. **Aucun mouvement n'est appris** : la forme reste immobile une fois
dessinée. Le corpus de formes est en pause à 8 400 mots sur 11 298.
Tests au 1er novembre : 64 passent, 1 sauté (référence torchvision absente de `.venv`).

---

# Bilan des essais du 31 octobre 2026

**Le modèle de flux reconnaissable n'est pas obtenu.** Le corpus est disponible,
mais les poids actuels ne satisfont pas F1. Aucune réussite sémantique n'est déduite
de la baisse de perte ou des tests fonctionnels.

Le corpus D2 conserve 1 138 080 grilles dans 23 710 clips, dont 988 862 grilles
distinctes avec augmentations et 243 900 observations source-objet distinctes.
Les sources, conditions, rejets et partitions par vidéo sont préservés.

| Essai | Entraînement | Résultat observé |
|---|---:|---|
| NCA initial, 1,29 M paramètres | 20 000 pas ; meilleur checkpoint à 10 000 | Taches génériques, dégradation en déroulé |
| Reconstruction masquée, 3,78 M | 6 000 pas, six concepts équilibrés | Formes moyennes ambiguës |
| Transformeur causal, 4,19 M | 8 000 pas, six concepts équilibrés | Formes incohérentes, effacement en génération autonome ; validation meilleure à 2 250 |
| Diffusion catégorielle des glyphes, 3,78 M | 2 000 pas après initialisation depuis le modèle masqué | Aucun des six sujets reconnu par l'utilisateur sur la planche présentée ; checkpoint évalué à 1 000 |

Les trois nouveaux essais utilisent cat, dog, horse, bird, fish et person, avec
échantillonnage équilibré des concepts et des vidéos. Ils apprennent directement
les jetons, avec deux têtes glyphe/couleur. L'inférence ne consulte pas de clips et
n'utilise aucun générateur d'images ou convertisseur.

L'utilisateur juge **la majorité des huit références réelles « cheval » lisibles**.
Ce retour porte sur ces huit images de données, pas sur chaque clip D2.
Sur les six générations du dernier essai, sa réponse est **« aucun »**.
Cette revue qualitative avec titres visibles n'est pas une étude en choix forcé.
Les deux revues sont enregistrées séparément, avec les identités des exemples et
les hashes des générations examinées.

Le test séparé `masked-six-capacity-001` reconstruit six images réelles mémorisées.
Il établit une capacité d'ajustement, sans prouver reconnaissance générale,
génération de nouvelles poses ou animation apprise. Il n'est pas déployé.

Les tests fonctionnels couvrent notamment le calcul causal sans lecture des cellules
futures, le posterior catégoriel, les IDs et couleurs valides, la conservation de
la grille lors d'une modification de texte, les formats de checkpoints, la reprise
NCA et la livraison des sources dans les notebooks. **57 tests passent.**
L'identité du code d'entraînement NCA correspond toujours à celle de ses poids.

Les essais sont terminés. Aucun nouveau checkpoint n'a remplacé les poids de
l'interface. La reconnaissance, une dynamique autonome convaincante, les langues,
les transitions et les performances cibles ne sont pas validées.

Le blocage à résoudre concerne désormais le générateur : apprendre une distribution
de silhouettes et de poses reconnaissables, puis conserver une dynamique sur ses
propres sorties. L'équilibrage des catégories n'a pas suffi, et prolonger les mêmes
pertes n'a pas amélioré la validation du transformeur. Une nouvelle relance massive
des mêmes réglages n'est donc pas justifiée par ces résultats.

Les détails sont dans [conclusion.json](outputs/recognition-audit-001/conclusion.json).
Les méthodes testées sont des implémentations expérimentales, inspirées de
[MaskGIT](https://arxiv.org/abs/2202.04200) et de la paramétrisation discrète de
[D3PM](https://arxiv.org/abs/2107.03006), sans reprendre leurs codecs ou leurs résultats.
