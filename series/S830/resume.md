---
serie: S830
statut: termine
type: experience
reference: S708
branche: codex/note-s830-private-musical-evidence256
run_id: 39632626042
commit_sha: 088e28e122bfc1f04c981716bf8db73faa615ca7
promotion: false
---

# S830 — Trois preuves musicales indépendantes pour Exact-K à PCM256

## Objectif
Tester si des experts **Onset / Sustain / Release** entièrement privés à la caractéristique Exact-K améliorent le S829 57,5247 % sur FIT80 sans fusion ni interdépendance des six axes. Rester à 256 échantillons par décision (44,1 kHz, cadence 5,80 ms), sans lookahead ni attente ajoutée.

## Code et architecture
Les fichiers ajoutés dans S830 sont `src/causal_note/s830_private_musical_evidence256.py`, `scripts/s830_train_private_musical_evidence256.py`, `test/test_s830_private_musical_evidence256.py`, le workflow, et ces rapports. Le code existant des cinq autres caractéristiques et le MIDI officiel n'ont pas été modifiés.

Chaîne isolée Exact-K : PCM → DSP privé S829 causal [49,72] par hauteur → deux modèles de cardinalité S829 réentraînés + trois classifieurs privés estimant Onset, Sustain et Release → sélection sur inner-CAL de propositions `s829`, `sustain`, `joint`, `router`, `blend` pour des seuils et décalages candidats. Les trois nouvelles têtes n'empruntent ni poids ni postérieurs aux véritables axes Onset/Offset.

## Protocole
Cinq folds de compositions disjointes : 3 compositions TRAIN (48 pistes), 1 inner-CAL (16 pistes), 1 OUTER (16 pistes). 80 pistes GuitarSet FIT signées S800, total 306 536 décisions sur frontières PCM256. On compare Exact-K K0–K6 toutes trames S830 contre S829 entièrement réentraîné dans chaque fold et H0 historique sur la même grille. Les labels n'interviennent que pour entraîner/calibrer/noter, jamais dans le calcul PCM en production.

## Validation et résultats
GitHub Actions [#39632626042](https://github.com/Andriamarosoa/note/actions/runs/39632626042) : `smoke` SUCCESS et `signed_fit80` SUCCESS ; artefact `S830_FIT80_INDEPENDENT_MUSICAL_HEADS_PCM256` id `12360813096`, `report.json complete=true`. L'exécution précédente #39632269301 avait échoué sur le traitement de la dernière tranche PCM incomplète (<256), corrigé avec test dédié.

- Exact-K S830 : **57,5247279 % = 176 334/306 536**.
- S829 apparié : **57,5247279 % = 176 334/306 536** ; gain S830 = **0**.
- H0 historique apparié : **61,6971579 % = 189 124/306 536** ; S830 = **−4,1724300 points**.
- Sous-comptages S830 : **83 523** ; surcomptages : **46 679**.
- **Sur les cinq folds, le mode retenu est toujours `s829`, seuil indicatif 0,30, shift 0.** Le choix empêche un mauvais expert privé de faire baisser le score, mais la nouvelle représentation musicale ne montre aucune amélioration mesurée.
- Les faiblesses persistantes concernent les vraies classes K2, K3, K5 et K6.

## Audits et limites
- Données : PASS, 80 pistes, groupes disjoints, sources/denominators signés.
- Acoustique : tests synthétiques de causalité PCM256, de fin de bloc incomplète PASS ; pas de test matériel iRig.
- Architecture : contrôles CI anti-import et fichiers protégés PASS ; aucun effet sur les cinq autres chemins puisque ceux-ci sont inchangés.
- Apprentissage : cinq vrais classifieurs privés par fold ; TRAIN/CAL/OUTER disjoints.
- Métriques : PASS, score apparié et détail K0–K6 du résultat final.
- LIVE : UNKNOWN, aucune preuve de p95/p99 CPU ni résultat matériel ; le calcul FIT80 est un replay hors ligne sur trames causales.
- Six agents Work et trois avis NVIDIA distincts : NON EXÉCUTÉS ; aucun consensus revendiqué.

## Verdict
**TERMINÉ, NON PROMU.** Les trois nouvelles têtes ne sont pas adoptées par le sélecteur sur les données CAL ; conserver la référence S828 pour Active Exact-K, S708 officielle et les cinq autres axes intacts. Ne pas considérer ce test comme preuve qu'une influence entre axes apporterait un gain : commencer par tester une transmission de *suggestions unidirectionnelles en lecture seule*, avec sélection calibrée, ablation et veto de régression, après constitution des candidats et métriques comparables.
