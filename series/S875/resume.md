# S875 — Résumé : GRU correcteur multibande limité à ±20%

**Travail réellement exécuté :** Réseau GRU causal entraîné sur TRAIN, filtre FIR de 4 bandes qui se somment exactement au standard granulaire 8192 quand les gains valent 1, sans mélange de deux standards déphasés.

**Mesures :** 6/6 hauteurs correctes CAL; distance log-STFT médiane CAL 0,16548 (DSP) → 0,15381; AUDITED 0,20692 → 0,19086. Les six notes CAL et six anciennes AUDITED bénéficient toutes de cette mesure, mais S873 est meilleur sur CAL (0,14258).

**Verdict :** Refus de promotion, pas assez bon face à S873.

Détails : [resultat.md](resultat.md). Aucun déploiement ni promotion.

**Contraintes** : transposition de guitare aiguë vers grave −12 demi-tons, 256 nouveaux PCM à 44,1 kHz par appel, audio passé autorisé mais aucun futur. La latence intrinsèque du DSP standard n'est PAS nulle ni auditée ici; la performance CPU a été volontairement ignorée. Six têtes NOTE et MIDI inchangés, aucune promotion. 39 extraits réels GuitarSet S847 : 27 TRAIN, 6 CAL, 6 anciennes AUDITED déjà consultées, donc pas de généralisation indépendante. L'enseignant offline librosa −12 n'est jamais présent à l'inférence.
