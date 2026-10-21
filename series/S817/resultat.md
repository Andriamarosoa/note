---
serie: S817
statut: termine
type: experience
reference: S708
branche: codex/note-s817-activek-temporal-highpoly
run_id: 39543749810
commit_sha: 817e57edec3b57421f61e3f4ab74580280ab3926
promotion: false
---
# S817 — Résultats spécialistes hautes polyphonies

## Exécution

Run final [#39543749810](https://github.com/Andriamarosoa/note/actions/runs/39543749810) SUCCESS : deux spécialistes temporels entraînés par 5 plis (10 nouveaux modèles), cinq compositions et 80/80 morceaux validés, S815 gelé et H0 reproduits. Premier run #39540692270 FAILED en sérialisant asdict non importé, corrigé ensuite. Audit séparé de toutes les décisions SHA [#39543864931](https://github.com/Andriamarosoa/note/actions/runs/39543864931) PASS.

## Métriques

| Caractéristique | Référence S708 officielle CAL32 | S817 nouveau FIT80 |
|---|---:|---|
| Onset F1 | 74,2639 % | non modifiée |
| Exact-K | 62,0455 % | non modifiée |
| Active Exact-K | 56,2762 % | **56,91248582 % FIT80** (141946 / 249411), identique S815 |
| Offset accuracy | 77,3292 % | non modifiée |
| Full Note F1 | 57,1319 % | non modifiée |
| Frame Pitch F1 | 76,4626 % | non modifiée |

Sur la même cohorte FIT80 déjà explorée, H0=57,01953803 %, S815=56,91248582 %, S816=56,42413526 %. S817 a retrouvé H0/S815 avant comparaison, puis a produit **exactement 0 changement de décision** sur 306536 trames MIDI256, donc 0 gain. Sous-comptages 65530, surcomptages 41935, contre H0 80654 et 26544. FIT80 reste exploratoire.

## Comparaison à S708

Référence historique S708 et S707 inchangées, aucune émission MIDI de production, aucune unification. Les nouvelles têtes utilisent uniquement des signaux acoustiques courants/passés, leur propre mémoire et la prédiction privée d'Active-K S815. Il n'y a pas de partage de poids avec Frame Pitch, Exact-K, Onset, Offset ni Full Note.

## Audits

| Axe | Statut | Preuve/limite |
|---|---|---|
| Données | PASS | 80 WAV/annotations + checkpoints + décisions SHA vérifiés dans runs complet et recovery |
| Acoustique | PASS | Contrat causal des échantillons 256 et fenêtres antérieures seules |
| Architecture | PASS | Deux spécialistes propres et S815 gelé sur branche de recherche |
| Apprentissage | FAIL | 10 modèles temporels privés entraînés, mais 0 décision changée ; aucune hausse de score |
| Métriques | PASS | Vrais 141946/249411, H0 142213/249411, 306536 frames authentiques |
| LIVE | UNKNOWN | Aucune latence physique iRig ou CPU total validée |

## Verdict

**REJET SCIENTIFIQUE** : S817 56,9125 % = S815 56,9125 %, sous H0 57,0195 %, aucune décision modifiée. Aucun >85 %, aucun système unifié ni LIVE promu.

## Prochaine action

Repenser les preuves acoustiques et l'architecture de cardinalité plutôt qu'ajouter des têtes peu sélectionnées. Tester un estimateur ordinal K≥1..6 indépendant, avec validation cinq compositions et preuves MIDI256, sans modifier S708.
