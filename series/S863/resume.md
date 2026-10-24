# S863 — Résumé

**But :** Première amélioration de hauteur par banc de filtres neuronal causal.

**Méthode :** Le réseau apprend sept filtres FIR de 513 coefficients (initialisés par passe-bas), une porte de sélection dépendant du signal PCM et une transformation neuronale paire. Toutes les sorties restent 256 PCM natifs, sans banque de clips ni shift classique à l'inférence.

**Mesures :** TRAIN 27 notes ; CAL 4/6 ; 6 notes déjà examinées 3/6 (50 %). Sur celles-ci, distance log-mel médiane vs standard 15,374 dB, contre 11,707 dB pour S860. P99 calcul réseau seul CPU local ≈0,434 ms ; centres d'énergie des filtres ~254–258 samples (~5,8 ms).

**Conclusion : Hausse de justesse sur les notes réutilisées, mais dégradation de fidélité audio et retard de phase probable. Non promu.** Le détail figure dans [resultat.md](resultat.md). Les artefacts de poids/audio/source sont fournis dans l'archive de conversation, pas déposés en binaire dans cette branche. Exécution de recherche locale uniquement ; aucun nouveau run musical GitHub Actions ni gain officiel NOTE.
