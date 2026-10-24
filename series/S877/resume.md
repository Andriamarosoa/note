# S877 — Résumé : Audit accords simulés et vrais passages sonores

**Travail réellement exécuté :** Six mélanges numériques de deux notes prélevées dans la banque authentique, ainsi qu'une véritable phrase GuitarSet de 3,5 s et sa dernière seconde. Référence offline −12 identique, jamais utilisée au runtime.

**Mesures :** S878 améliore la distance de S873 dans 5/6 mélanges, mais régresse sur [58,64]. Ces six mélanges ne sont PAS des accords effectivement joués. La phrase réelle est le contrôle décisif de la régression.

**Verdict :** Gain sur accords synthétiques insuffisant pour annoncer un gain sur vraie polyphonie.

Détails : [resultat.md](resultat.md). Aucun déploiement ni promotion.

**Contraintes** : transposition de guitare aiguë vers grave −12 demi-tons, 256 nouveaux PCM à 44,1 kHz par appel, audio passé autorisé mais aucun futur. La latence intrinsèque du DSP standard n'est PAS nulle ni auditée ici; la performance CPU a été volontairement ignorée. Six têtes NOTE et MIDI inchangés, aucune promotion. 39 extraits réels GuitarSet S847 : 27 TRAIN, 6 CAL, 6 anciennes AUDITED déjà consultées, donc pas de généralisation indépendante. L'enseignant offline librosa −12 n'est jamais présent à l'inférence.
