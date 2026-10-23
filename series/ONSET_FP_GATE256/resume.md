# SÉRIE FP GATE — sélection neuronale indépendante des experts Onset 69D et 74D, strict PCM256

Branche `codex/note-onset-fp-learned-expert-gate256`. Hypothèse [#143](https://github.com/Andriamarosoa/note/issues/143) enregistrée AVANT la nouvelle expérience. Nouvelle règle scientifique de réfutation [#140](https://github.com/Andriamarosoa/note/issues/140).

## Références scientifiquement mesurées

Sur 80 vrais WAV GuitarSet, 306536 décisions à 256 nouvelles valeurs audio (44100Hz), le replay complet précédent [#39775357133](https://github.com/Andriamarosoa/note/actions/runs/39775357133) a démontré :

| Modèle | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|
| S613 native256 | 8099 | 4600 | 3168 | 67.5874155% |
| 3 têtes avec ErrorPrediction69 no feedback | 8054 | 4411 | 3213 | 67.8745997% |
| 4 têtes + FNRecovery | 8160 | 4605 | 3107 | 67.9094541% |
| 5e tête FP contrôle69 | 8127 | **4519** | 3140 | **67.9713963%** |
| Contrôle70 | 8137 | 4549 | 3130 | 67.9413852% |
| FP temporel74 systématique | 7948 | 4290 | 3319 | 67.6281642% — **rejeté** |

Le problème ne peut pas être traité par un veto harmonique fixe. Dans le précédent dataset d'événements FP69 12465 NoteOn d'origine déjà acceptés, Rock3 a **259 désaccords où 69 accepte/74 rejette** : **128 TP et 131 FP**. Parmi les 128 vrais TP rejetés par 74, **101 n'ont PAS** de parent harmonique strictement passé dans les 80ms. Cela montre que la tête temporelle peut réagir trop largement à des caractéristiques qui ne sont pas le risque qu'on cherchait.

## Hypothèse réfutable H_FP_EXPERT_GATE_02

Une **sixième tête indépendante à l'intérieur d'Onset**, `FPExpertGateHead`, devrait pouvoir apprendre à arbitrer seulement les cas où le 69D et le 74D se contredisent (éventuellement 69D rejette et 74D accepte), au lieu d'appliquer 74D uniformément. Les cas sur lesquels ils sont d'accord conservent **exactement** leur décision. Si on ne dispose pas d'un contrôle de désaccord suffisamment prouvé, retour EXACT au meilleur contrôle 69D, sans aucune modification du sélecteur.

Le modèle est **un nouveau vrai réseau supervisé `StandardScaler + MLP(16,8)` à 11 entrées strictement causales** : scores FP 69D/70D/74D actuels, ancienne prédiction de `P(correct)` 69D, différences de scores entre experts, quatre âges de parents harmoniques du Main strictement PASSÉS et hauteur MIDI normalisée. Le nombre de dimensions est 3+1+2+4+1=11. **Aucune annotation GT, classification solo/comp connue seulement par fichier, label de composition ou information d'une trame future n'entre dans le LIVE.**

## Entraînement et choix scientifique

Les quinze modèles FP 69D/70D/74D entraînés et leurs seuils CAL du run #39775357133 sont **gelés par checksum SHA256**. Les trois têtes S613/S701/ErrorPrediction ainsi que FNRecovery restent bitwise gelées.

Chaque OUTER composition possède une composition CAL originale contenant 16 pistes. Une nouvelle séparation déterministe en **8 pistes CAL-gate-TRAIN alternées** et **8 pistes CAL-gate-VALIDATION** est faite, sans aucune piste OUTER dans l'entraînement ou dans le choix de seuil. Les 11D ne sont entraînées QUE sur les désaccords des deux experts dans les huit premières pistes CAL. Si le nombre de désaccords positifs/négatifs ne suffit pas, le gate est désactivé. Sur les huit autres pistes, CAL VAL ne peut activer le gate que si le **proxy** a plus de F1 et moins d'au moins **deux FP** que 69D. L'ancien FP 69D et le modèle 74D restent des contrôles fixes.

**L'optimisation des seuils d'origine des experts 69D/74D avait déjà utilisé la CAL entière** dans l'expérience antérieure : le sous-découpage est un dispositif supplémentaire prudent, pas une preuve de validation externe. Les huit pistes gate-VALIDATION ne sont jamais utilisées pour les nouveaux poids de gate.

Le gate peut **uniquement conserver/refuser des attaques qui ont déjà été acceptées par ErrorPrediction69**. Les NoteOn FNRecovery supplémentaires sont intégralement conservées. Aucune machine d'état Main/Re-arm/Risk/FNRecovery/Offset/Exact-K, aucun modèle 1024 et aucun MIDI production ne changent. Tous les scores de neural heads et du gate sont produits **avant le sélecteur de la même décision PCM256**, pas de lookahead/hold.

## Critère préenregistré et réfutation

Référence à battre : meilleur contrôle69 **TP8127, FP4519, FN3140, F1=67.9713963%**.

Le vrai **replay original de 80 WAV** doit confirmer **F1 strictement >67.9713963% ET FP strictement <4519** à 256 échantillons, et reproduire exactement 69D, les autres experts et quatre anciennes têtes. Le proxy CAL ne vaut jamais promotion, et aucune retouche de seuil n'est autorisée après lecture de OUTER. Toute régression par piste/famille doit être publiée ; aucune promotion sur ces 80 seules pistes sans tests indépendants de généralisation.

Si le gain prédit n'apparaît pas, état `REFUTED_OR_INCOMPLETE`, audit des nouveaux TP perdus et FP conservés, pas de correction opportuniste du même diagnostic. Un test de cadence 256 (=5.805ms) ne mesure pas la latence totale microphone→MIDI, qui doit être mesurée séparément.

Les anciennes têtes S613/S701 avaient vu une partie de FIT80 lors de leur training historique : aucune preuve de généralisation externe complète.

## Fichiers

- `src/onset256/fp_expert_gate.py` : onze entrées causales, vrai checkpoint vérifié SHA256, route contextuelle ou fallback 69D.
- `src/onset256/heads.py` : tête indépendante FPExpertGate dans le réseau Onset.
- `src/onset256/runtime.py` : calcul sur chaque trame AVANT les décisions, sortie privée expérimentale `fp_gated_unified`, pas de nouveau MIDI production.
- `scripts/train_onset_fp_expert_gate256.py` : 5 petits réseaux MLP supplémentaires sur CAL, deux sous-partitions CAL disjointes, seuils figés, aucune OUTER.
- `scripts/replay_onset_fp_expert_gate256.py` : vrai replay 80 WAV strict causal, tous scores TP/FP/FN, audit de l'hypothèse.
- `tests/test_onset_fp_expert_gate256.py` : invariante des anciennes têtes, no lookahead, risque de duplication, entraînement réel, fallback.
- `.github/workflows/onset-fp-learned-gate256.yml` : smoke, provenance signé, model training, replay réel et artifact.
- `series/ONSET_FP_GATE256/resume.md` / `resultat.md` : historique et verdict scientifique.

**Aucun nouveau F1 n'est revendiqué avant un workflow scientifique entièrement exécuté.**
