---
serie: S837
statut: termine
type: experience
reference: S708
branche: codex/note-s837-six-axis-80pct-independent-loops256
run_id: 39998150139
commit_sha: 6216833aaeeec2880c804c4c0d04d4b9b3d34b1a
promotion: false
---

# S837 — Boucles six axes indépendant PCM256, comparaison CAL→OUTER terminée

## Objectif
Tendre vers >80 % chacun pour Exact-K, Active Exact-K, Onset F1, Offset F1, Frame Pitch F1 et Full Note F1 sur la même cohorte, sans attendre de future donnée et sans que l'amélioration de l'un influence les cinq autres. La première série de boucles de politiques individuelles était un diagnostic, pas une promesse d'obtenir 80 % par simple réglage de seuil.

## Hypothèse
Les six décodeurs S834 n'étaient pas calibrés pour les nouveaux postérieurs acoustiques obtenus strictement tous les 256 échantillons. Des variantes indépendantes de seuils K, attack/sustain, release et réarmement pouvaient regagner des notes, mais risquaient des erreurs de transfert entre compositions. Les erreurs de décision sont examinées après évaluation pour bâtir une nouvelle famille de sélecteurs par fragment.

## Modifications
- `scripts/s837_six_axis_policy_loops256.py` : source S708/S836 congelée, une seule inférence acoustique réelle par hop; déclenchement de banques privées de candidats sans sortie MIDI.
- `test/test_s837_six_axis_policy_loops256.py` : 30 tests dans les suites CI de la série, préservation de la référence, preuve d'exclusion de l'OUTER de la calibration.
- `series/S837/audit_loop1.md` : échec initial et diagnostic des contre-exemples Exact-K/Active-K. Fix orthographique tp vs TP, qui n'était pas une vraie erreur musicale.
- Workflow `.github/workflows/note-s837-six-axis-policy-loops256.yml`, run final [#39998150139](https://github.com/Andriamarosoa/note/actions/runs/39998150139), succès.
- Aucun modèle d'origine, aucune sortie MIDI, aucune autre tête, ni aucun pipeline microphone Live n'a été modifié.

## Protocole
- 5 vrais WAV mono GuitarSet accompagnement, 5 compositions, 20 062 frontières PCM256, source et provenance SHA gelées. 4 compositions choisissent une politique, la 5e OUTER ne sert qu'à l'évaluation. Pour chaque OUTER, re-sélectionner sur les autres compositions.
- Aucun audio futur, lookahead ou hold ajouté ; calcul postérieur S708 chaque 256 et décodeurs isolés.
- Attention : modèle S708 source entraîné historiquement à 1024 et FIT80 déjà exploré. Ce protocole **ne prouve aucune généralisation externe** ; 5 WAV _comp ne représentent pas les 80 WAV originaux complets et les solos.
- Toujours comparer la métrique identique sur les mêmes traces : K en accuracy, Onset/Offset/Frame Pitch/Full Note en F1; Offset F1 ne peut pas être comparé à Offset accuracy S708.
- Si un changement gagne peu ou perd, ne pas le promouvoir. Analyser identité des TP récupérés/perdus, FP nouvelles/supprimées, sur/sous comptages K0–K6, harmonique plausible non prouvée, registre et composition.
- Conserver les six architectures et états privés ; aucune fusion en MIDI avant stricte non-régression des six axes et certification Windows iRig temps réel.

## Audits obligatoires
| Rôle | Résultat |
|---|---|
| Données | PASS 5/5 compositions FIT80 authentiques, SHA |
| Acoustique | PASS 20 062 décisions à 256, sans futur |
| Architecture | PASS 6 décodeurs privés, sources et MIDI inchangés |
| Apprentissage | PASS protocole de choix excluant la composition OUTER ; source préentraînée 1024 |
| Métriques | FAIL 5/6 axes restent sous 80, plusieurs régressions |
| LIVE | UNKNOWN, pas de validation matérielle Windows ni p99 pour ces variantes |

## Statut
TERMINÉ pour les six boucles de grille, non promu. [Run #39998150139](https://github.com/Andriamarosoa/note/actions/runs/39998150139) PASS : Exact-K 56,60→56,27 %, Active-K 46,77→45,45 %, Onset F1 68,45→64,84 %, Offset F1 **18,18→49,32 %**, Frame Pitch F1 84,97→84,90 %, Full Note F1 67,72→68,36 %, sur 5 WAV seulement. Les gains Offset sont réels sur cet échantillon mais apportent beaucoup de FP et n'atteignent pas 80 %. S838 ouvre une tête de sélection **apprise par fragment**, pour accepter les corrections locales et bloquer les régressions, avant extension FIT80.
