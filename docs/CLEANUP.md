# Nettoyage du 1er novembre 2026

La v4.2 est retirée du projet actif et du dépôt public. Les références locales
sont déplacées dans `data/local-archives/2026-11-01/`, exclu de Git :

- `ascii_flux_v42/`, y compris son ancien cache de modèles ;
- `ASCII_Flux_OpenVocabulary_couleur_interface (1).ipynb` ;
- `ascii_flux_v5_colab/`, ancienne copie générée que le bootstrap peut recréer ;
- `CLAUDE.MD`, vide ;
- l'ancien `.pytest_cache/`.

Cet archivage est réversible. Les corpus, checkpoints, sorties d'expériences et
environnements Python actifs restent locaux. La génération du corpus de formes
continue dans `data/v5/shapes-001`.

Git publie les sources v5, tests, configurations, notebooks autonomes et la
documentation. Les poids versionnés sont attachés aux releases, avec manifest
et SHA-256. Les corpus et les modèles tiers téléchargés ne sont pas publiés.

La découverte des packages est limitée à `ascii_flux` pour éviter d'embarquer
une ancienne copie de bootstrap lors d'une installation.
