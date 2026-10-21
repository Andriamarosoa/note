---
serie: S832
statut: termine
type: experience
reference: S708
branche: codex/note-s832-guarded-local256
run_id: 39644879595
commit_sha: 97c9348fa641da2c3daa56ed921f157924f2c1e1
promotion: false
---

# S832 — Veto local conservateur Exact-K PCM256 : test terminé

## Objectif
Tester si les trois preuves musicales privées S831 (attaque, maintien, relâchement) peuvent proposer des corrections polyphoniques à la référence S829 sur des fragments PCM256 sans modifier aucun des cinq autres axes ni le MIDI officiel, et sans nouvelle attente ou lookahead.

## Hypothèse
Une sélection locale des propositions selon K initial, K proposé, niveau de preuve musicale et risque empirique sur pistes de calibration peut récupérer les bons changements K2–K6 sans sacrifier massivement K0/K1. Des seuils élevés de risque (Wilson, nombre minimal de corrections, majorité par piste, validation sur pistes CAL distinctes) doivent bloquer les règles non reproductibles.

## Modifications
`src/causal_note/s832_guarded_private_selector256.py` : 20 candidats propres à Exact-K (= 4 modes × 5 seuils), segmentation des risques en cellules (mode, seuil, K source, K proposé, bande de confiance positive), apprentissage par sous-CAL (12 pistes) puis sélection/veto sur 4 pistes CAL, avec repli strict S829. `scripts/s832_fit80_guarded_local256.py` rejoue les 80 pistes et audite 306 536 frontières, K0–K6. `test/test_s832_guarded_local256.py` vérifie fallback, signaux GT absents de `choose()`, vraisemblance et candidat shape.

## Protocole
S800 FIT80 authentique, 80 WAV GuitarSet et checksum validés, cinq folds disjoints 48 TRAIN/16 inner-CAL/16 OUTER, 306 536 décisions PCM256. Trois têtes musicales S831 et deux têtes K S829 propres à l'axe réentraînées. Aucun entraînement sur les pistes OUTER ; scores appariés S832/S829/H0.

## Audits obligatoires
- Données : PASS, 80 WAV S800 signés.
- Acoustique : PASS synthétique, test causal PCM256 ; aucune mesure matérielle iRig.
- Architecture : PASS import/fichiers des cinq autres axes inchangés ; aucun partage.
- Apprentissage : PASS, apprentissage des cellules CAL puis veto sur pistes CAL distinctes.
- Métriques : PASS, 80 pistes, 306 536 fenêtres, comparaison exacte S829.
- LIVE : UNKNOWN, pas de p95/p99 physique ni sérialisation déployable.
- Agents NVIDIA/Work : non sollicités.

## Statut
**TERMINÉ, NON PROMU.** Run [#39644879595](https://github.com/Andriamarosoa/note/actions/runs/39644879595) PASS ; artefact `S832_FIT80_GUARDED_LOCAL_PCM256` id `12367129426`. S832 **57,5247279 % = 176 334/306 536**, identique au S829 apparié, sans régression ni gain. H0 : 61,6971579 %.

Les **cinq folds choisissent zéro règle** (`validation_veto`). La sous-validation CAL gagne parfois malgré une instabilité sur certaines pistes : Funk2 max +44 décisions (16 536 fenêtres), Rock3 max +939 ; BN1, Jazz1 et SS1 ont des scores négatifs pour toutes les règles testées. Cela prouve le fonctionnement du veto, pas un progrès de précision. Prochaine piste : contrôle de risque plus local/généralisable, sans régler les paramètres à partir des étiquettes OUTER déjà consultées ni attribuer un score validé au futur candidat.
