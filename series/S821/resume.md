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
# S821 — Revue NVIDIA approfondie et nouveau garde-fou causal contre K4→K5/K6

## Objectif

Faire trois vrais modèles NVIDIA auditer les 665 faux changements vrai K4 vers K5/K6 sur SS1, puis entraîner un vérificateur privé avec hypothèse contrôlable, sans toucher S708 ou LIVE. Tout gain doit être démontré par cinq partitions composition-excluses avant toute éventuelle évaluation externe.

## Hypothèse

Dans S820, la règle max_high@0.70 corrige 355 frames SS1 mais détériore 913, surtout les GT K4 (−555), alors qu'elle améliore GT K5 (+152) et K6 (+17). Les indices globaux harmonique/fondamentale sont très similaires pour les bons et mauvais changements. Un signal par hauteur et un vérificateur appris exclusivement sur TRAIN pourraient mieux séparer vrai K4 d'un accord de 5/6 notes ; ce n'est pas présumé vrai sans ablation.

## Modifications

Nouveau briefing neutre S821 à trois modèles NVIDIA NIM indépendants ; quatre contrôles de cohérence arithmétique, code, latence/causalité et fuites ; chaque modèle livre son propre audit. Une architecture SHADOW Active-K indépendante sera ensuite testée, jamais choisie a posteriori sur les scores outer. Les modèles S815/S817 restent gelés comme comparateurs, sans nouvelle dépendance à Frame Pitch ou aux autres têtes.

## Protocole

GuitarSet FIT80 réel et SHA signé, 80 WAV, 5 plis 48/16/16 disjoints par composition, 306536 trames MIDI256. Préserver le même compteur 249411 frames Active-K et les comptes S815/S817/S820 à l'identique. Apprentissage d'une tête conditionnelle au vrai K4/K5/K6 sur uniquement TRAIN, poids propres, décision sur inner-CAL seulement, évaluation externe distincte nécessaire au-delà de FIT80 déjà exploré.

## Audits obligatoires

Frais NVIDIA et appel réel vérifiés par logs/artefacts ; auditions indépendantes aveugles, revue de contradictions et hypothèses falsifiables. Tests DSP et absence de données futures, de hold, de poids inter-caractéristiques, de GT en inférence et de fuite des compositions. Conditions de succès : score Active-K OOF et chaque K sans dégradation, puis contrôle six axes, mesure LIVE réelle et validation source inédite avant promotion.

## Statut

En cours. Aucun nouveau score S821, aucune promotion, référence S708 et LIVE intacts.


## Clôture scientifique vérifiée — 21 octobre 2026

**Deux workflows distincts authentifiés :** consultation **réelle** des trois NVIDIA NIM (Gemma 4 31B, Nemotron 3.5 Lightning 30B, GLM 5.3 Flash) [#39561989215](https://github.com/Andriamarosoa/note/actions/runs/39561989215), artefact [#12327798915](https://github.com/Andriamarosoa/note/actions/runs/39561989215/artifacts/12327798915) ; entraînement réel de 5 plis × 2 têtes privées sur GuitarSet originaux SHA MIDI256 et évaluation externe composition-disjointe [#39562763898](https://github.com/Andriamarosoa/note/actions/runs/39562763898) **SUCCESS**, artefact [#12329186784](https://github.com/Andriamarosoa/note/actions/runs/39562763898/artifacts/12329186784). Exactement 80 pistes GuitarSet, 306536 trames audio 256-sample, 249411 trames GT actives, 48 train/16 cal/16 test pour chaque composition. Checkpoints figés S815 et S817 SHA vérifiés, parité H0/S815 vérifiée, autres cinq caractéristiques et LIVE non modifiés.

**Global FIT80 :** H0 officiel 142213/249411 = **57,01953803 %** (undercount **80654**, overcount **26544**) ; S815 figé 141946/249411 = **56,91248582 %** (under **65530**, over **41935**) ; **S821 = 141971/249411 = 56,92250943 %** (under **65129**, over **42311**). Évolution S821 − S815 : **+25 corrects**, **−401 undercount**, **+376 overcount**, **+0,01002362 point** ; S821 − H0 **−242 corrects**, **−0,09702860 point**. Un workflow PASS technique n'autorise aucune promotion.

**Par composition (Δ correct vs S815, sélection inner-CAL uniquement) :** BN1 **0**, mode `verify5@0.70` sans changement effectif ; Funk2 **−85**, mode `verify5@0.40` ; Jazz1 **+110**, mode `verify56@0.40` ; Rock3 **0**, `base` ; SS1 **0**, `base`. Les cinq compteurs undercount S821 sont **BN1 12023, Funk2 15138, Jazz1 12771, Rock3 12150, SS1 13047**, dont la somme est **65129** : un nombre de ~12000 lu dans les logs est donc généralement un **compteur propre à une composition**, PAS +12000 nouvelles erreurs globales.

**Par vrai K :** K1 87731/112014 = 78,3215 %, K2 19632/46311 = 42,3917 %, K3 15789/34182 = 46,1910 %, K4 **12619/35190 = 35,8596 %**, K5 **5262/16323 = 32,2367 %**, K6 938/5391 = 17,3994 %. Relativement à S815, les scores corrects K4 **−376**, K5 **+401** ; toutes les autres classes **inchangées**. Le gain K5 est payé par une dégradation K4 — le problème de discrimination persiste. La sélection `base` sur SS1 **empêche** de reproduire la régression de S819 sur SS1, mais elle ne corrige pas les erreurs préexistantes de SS1.

**VERDICT : REJET POUR PROMOTION** malgré +25 corrects exploratoires ; H0 FIT80 reste meilleure référence que S821, FIT80 a déjà été largement exploré, aucune mesure indépendante externe ni latence matérielle complète, aucun de six enseignants >85% officiellement validé. Conserver les checkpoints et prédictions S821 pour analyse, sans les fusionner ni intégrer au MIDI LIVE. Prochain axe : analyse causale par hauteur des véritables ajouts (F0/harmoniques) et validation externe, puis envisager K4 vs K5 robuste à changement de composition. Les trois avis NVIDIA sont hypothèses, leurs erreurs sont consignées dans `nvidia_critical_review.md`.
