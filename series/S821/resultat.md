---
serie: S821
statut: termine
type: experience
reference: S708
branche: codex/note-s821-nvidia-ss1-k4-guard
run_id: 39562763898
commit_sha: c044b49badd39fef48d9785672428afd9688bb16
promotion: false
---
# S821 — Résultats de la revue et du correctif Active-K

## Exécution

Consultations NVIDIA indépendantes et test de garde-fou acoustique à exécuter, sans résultat musical inventé.

## Métriques

| Axe | Référence officielle S708 CAL32 | S821 SHADOW |
|---|---:|---|
| Onset F1 | 74,2639 % | inchangé |
| Exact-K | 62,0455 % | inchangé |
| Active Exact-K | 56,2762 % | non mesuré |
| Offset accuracy | 77,3292 % | inchangé |
| Full Note F1 | 57,1319 % | inchangé |
| Frame Pitch F1 | 76,4626 % | inchangé |

Comparaison équitable FIT80 : H0 57,019538 %, S815 56,912486 %, S817 56,912486 %, S820 max_high@.70 56,63864064 % (diagnostic, non candidat). FIT80 déjà exploré, non preuve de généralisation.

## Comparaison à S708

Aucune modification S708, aucun MIDI LIVE branché, aucun nouveau score des six axes. Seule recherche Active-K SHADOW, décisions 256 échantillons causales.

## Audits

| Domaine | Statut | Motif |
|---|---|---|
| Données | PASS | 80 sources authentifiées, SHA checkpoints, 48/16/16 × 5 plis |
| Acoustique | UNKNOWN | Preuves f0/harmoniques individuelles à auditer |
| Architecture | PASS | Deux nouveaux modèles privés entraînés, décision K4-only vérifiée |
| Apprentissage | PASS | 5 plis train/CAL/test composition exclus, sélection CAL seulement |
| Métriques | PASS | H0 142213, S815 141946, S821 141971 / 249411 authentiques |
| LIVE | UNKNOWN | Aucun test de latence matérielle ; moteur LIVE inchangé |

## Verdict

En cours, aucun candidat S821 promu ni unifié. L'avis d'un LLM ne constitue jamais une mesure d'amélioration.

## Prochaine action

Revoir indépendamment les trois réponses NVIDIA, vérifier leurs contre-arguments, construire le garde-fou conditionnel K4 et tester le nouveau modèle sur cinq compositions.


## Clôture scientifique vérifiée — 21 octobre 2026

**Deux workflows distincts authentifiés :** consultation **réelle** des trois NVIDIA NIM (Gemma 4 31B, Nemotron 3.5 Lightning 30B, GLM 5.3 Flash) [#39561989215](https://github.com/Andriamarosoa/note/actions/runs/39561989215), artefact [#12327798915](https://github.com/Andriamarosoa/note/actions/runs/39561989215/artifacts/12327798915) ; entraînement réel de 5 plis × 2 têtes privées sur GuitarSet originaux SHA MIDI256 et évaluation externe composition-disjointe [#39562763898](https://github.com/Andriamarosoa/note/actions/runs/39562763898) **SUCCESS**, artefact [#12329186784](https://github.com/Andriamarosoa/note/actions/runs/39562763898/artifacts/12329186784). Exactement 80 pistes GuitarSet, 306536 trames audio 256-sample, 249411 trames GT actives, 48 train/16 cal/16 test pour chaque composition. Checkpoints figés S815 et S817 SHA vérifiés, parité H0/S815 vérifiée, autres cinq caractéristiques et LIVE non modifiés.

**Global FIT80 :** H0 officiel 142213/249411 = **57,01953803 %** (undercount **80654**, overcount **26544**) ; S815 figé 141946/249411 = **56,91248582 %** (under **65530**, over **41935**) ; **S821 = 141971/249411 = 56,92250943 %** (under **65129**, over **42311**). Évolution S821 − S815 : **+25 corrects**, **−401 undercount**, **+376 overcount**, **+0,01002362 point** ; S821 − H0 **−242 corrects**, **−0,09702860 point**. Un workflow PASS technique n'autorise aucune promotion.

**Par composition (Δ correct vs S815, sélection inner-CAL uniquement) :** BN1 **0**, mode `verify5@0.70` sans changement effectif ; Funk2 **−85**, mode `verify5@0.40` ; Jazz1 **+110**, mode `verify56@0.40` ; Rock3 **0**, `base` ; SS1 **0**, `base`. Les cinq compteurs undercount S821 sont **BN1 12023, Funk2 15138, Jazz1 12771, Rock3 12150, SS1 13047**, dont la somme est **65129** : un nombre de ~12000 lu dans les logs est donc généralement un **compteur propre à une composition**, PAS +12000 nouvelles erreurs globales.

**Par vrai K :** K1 87731/112014 = 78,3215 %, K2 19632/46311 = 42,3917 %, K3 15789/34182 = 46,1910 %, K4 **12619/35190 = 35,8596 %**, K5 **5262/16323 = 32,2367 %**, K6 938/5391 = 17,3994 %. Relativement à S815, les scores corrects K4 **−376**, K5 **+401** ; toutes les autres classes **inchangées**. Le gain K5 est payé par une dégradation K4 — le problème de discrimination persiste. La sélection `base` sur SS1 **empêche** de reproduire la régression de S819 sur SS1, mais elle ne corrige pas les erreurs préexistantes de SS1.

**VERDICT : REJET POUR PROMOTION** malgré +25 corrects exploratoires ; H0 FIT80 reste meilleure référence que S821, FIT80 a déjà été largement exploré, aucune mesure indépendante externe ni latence matérielle complète, aucun de six enseignants >85% officiellement validé. Conserver les checkpoints et prédictions S821 pour analyse, sans les fusionner ni intégrer au MIDI LIVE. Prochain axe : analyse causale par hauteur des véritables ajouts (F0/harmoniques) et validation externe, puis envisager K4 vs K5 robuste à changement de composition. Les trois avis NVIDIA sont hypothèses, leurs erreurs sont consignées dans `nvidia_critical_review.md`.
