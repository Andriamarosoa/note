# S880 — Résumé : Entraînement enrichi d'accords simulés

**Travail réellement exécuté :** 19 mélanges de deux notes réelles, composés uniquement à partir des 27 notes TRAIN et réentraînement de l'égaliseur à gain borné, depuis S876 (fine-tune) et depuis zéro, sélection sur CAL, phrase externe examinée seulement à la fin.

**Mesures :** CAL, distance EQ fine-tune 0,14728 ; EQ from scratch 0,14570. En cascade avec S873 sur vraie phrase : 0,41641 (fine-tune) et 0,41557 (scratch), contre S873 0,40664 ; dernière seconde pire également.

**Verdict :** Les deux variantes refusées, aucune correction de la régression sur vraie phrase.

Détails : [resultat.md](resultat.md). Aucun déploiement ni promotion.

**Contraintes** : transposition de guitare aiguë vers grave −12 demi-tons, 256 nouveaux PCM à 44,1 kHz par appel, audio passé autorisé mais aucun futur. La latence intrinsèque du DSP standard n'est PAS nulle ni auditée ici; la performance CPU a été volontairement ignorée. Six têtes NOTE et MIDI inchangés, aucune promotion. 39 extraits réels GuitarSet S847 : 27 TRAIN, 6 CAL, 6 anciennes AUDITED déjà consultées, donc pas de généralisation indépendante. L'enseignant offline librosa −12 n'est jamais présent à l'inférence.
