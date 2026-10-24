# S878 — Résumé : Correction IA en cascade sur S873

**Travail réellement exécuté :** Appliquer la tête multibande entraînée S876 à la sortie du meilleur hybride antérieur S873. Le standard −12 demeure source de toute hauteur; correction acoustique seulement.

**Mesures :** CAL 6/6 notes justes; distance médiane 0,14258 → 0,13530 (−5,11 %). Anciennes AUDITED 6/6 et distance 0,17657 → 0,15855 (−10,21 %). MAIS phrase réelle : 0,40664 → 0,41977 (pire) et dernière seconde 0,52626 → 0,56701 (pire).

**Verdict :** Refus de promotion : régression mesurée sur la performance réelle malgré le gain notes isolées.

Détails : [resultat.md](resultat.md). Aucun déploiement ni promotion.

**Contraintes** : transposition de guitare aiguë vers grave −12 demi-tons, 256 nouveaux PCM à 44,1 kHz par appel, audio passé autorisé mais aucun futur. La latence intrinsèque du DSP standard n'est PAS nulle ni auditée ici; la performance CPU a été volontairement ignorée. Six têtes NOTE et MIDI inchangés, aucune promotion. 39 extraits réels GuitarSet S847 : 27 TRAIN, 6 CAL, 6 anciennes AUDITED déjà consultées, donc pas de généralisation indépendante. L'enseignant offline librosa −12 n'est jamais présent à l'inférence.
