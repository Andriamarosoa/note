# Audit des codes Onset PCM256 — résumé et traçabilité (24 octobre 2026)

## But
Auditer **la vraie chaîne Onset PCM256 HGB référence 69,258433 %**, pas S834/S708. Examiner strictement la couverture des candidats, l'entraînement et ses splits, les features acoustiques, l'arbitre, les contraintes de latence, l'égalité cache/Live et l'intégrité de cinq autres têtes sans promotion.

## Sources figées
- Base : `codex/note-onset-learned-hgb-arbiter256` commit `e26cf43535c61795e5c2b1a63fbde85841edb79d`.
- Replay scientifique : https://github.com/Andriamarosoa/note/actions/runs/39957354552 ; test pass ; full80 rejoue 80 WAV, échoue **uniquement** à la règle scientifique de non-régression.
- Contrôle : 7607 TP, 3093 FP, 3660 FN → 69,258433 % ; HGB 7607 TP, 3096 FP, 3660 FN → 69,248976 %.
- Candidats incrémentaux GT récupérables : 802 (BN1 0/Funk2 4/Jazz1 0/Rock3 631/SS1 167).
- Même si tous les 802 étaient corrigés sans FP, F1 plafond **73,863586 %**.
- 48/80 FIT80 partagés avec S701 TRAIN, 32/80 FIT80 avec S701 HELD ; S613 TRAIN contient 20/80 ; source `series/ONSET_S701_256/diagnostic_scientifique.md`.

## Méthode
Inspection directe de `src/onset256/` (acoustic_attack, acoustics, acoustic_live, learned_hgb_arbiter, learned_hgb_live, model, runtime, selector) et `scripts/` (onset_acoustic_direct256, onset_learned_hgb_arbiter256, onset_event_integrity256, replay_native256), de workflows et rapports de provenance existants. Nouveaux tests d'audit ajoutés sur la branche **distincte** `codex/note-onset-code-audit-256-20261024`. Aucune réécriture de modèle, poids, noms, MIDI ou décodeurs officiels.

## Découvertes importantes
1. **CRITIQUE, preuve mathématique :** HGB ne peut pas créer une attaque non proposée par la référence : plafond ≤73,86 % si les FP restent identiques. Il faut un générateur acoustique indépendant.
2. **CRITIQUE, crédibilité des scores :** benchmark FIT80 exploré et partagé avec les enseignants historiques, non validation externe ; nouveaux gates sont par fold mais pas toute la chaîne amont.
3. **MAJEUR, anomalie de code :** `predict_proba` 195D recalculé deux fois par frontière 256 dans l'enveloppe HGB ; confirmer par test de compteur.
4. **MAJEUR, reproductibilité :** les modèles archivés 69,2062 % et réentraînés 69,2584 % divergent ; provenance à approfondir par fichiers et versions.
5. **MAJEUR, définition PCM256 :** hop/entraînement des têtes S613/S701 256, mais FFT sur 2048/4096/8192 **échantillons passés** ; si la contrainte vise toute fenêtre, ceci n'est pas strictement 256.
6. **MAJEUR, déploiement :** cinq checkpoints par composition OUTER au lieu d'un modèle de production unique mesuré en Live sur Windows/iRig.
7. **MODÉRÉ :** la métrique Onset ±80ms ne prouve pas un temps de réaction audio physique de 5,805ms.

## Détails
Lire [resultat.md](resultat.md) pour les liens vers les fonctions, preuves, calculs et plan de correction. Ce rapport est un audit **sans entraînement**, aucun nouveau F1 et aucune promotion.

**Interdits maintenus :** pas de fusion des six axes, pas de retour aux modèles S708-1024 pour la chaîne Onset, pas de lookahead, pas d'attente ajoutée ni prétention >80 % sans 80 WAV fiables et contrôle p99 sur matériel.
