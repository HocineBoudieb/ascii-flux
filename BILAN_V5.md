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
