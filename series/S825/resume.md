---
serie: S825
statut: termine
type: experience
reference: S708
branche: codex/note-s825-private-crossfit-pitch-midi256
run_id: 39587162104
commit_sha: d012d7e116789efa50ab88ea9b789685bfb5bc67
promotion: false
---
# S825 — Boucles de têtes Active Exact-K véritablement indépendantes sur PCM MIDI256

## Objectif

Après l'ablation S824, apprendre une nouvelle représentation probabiliste de la présence des 49 hauteurs **depuis le PCM natif MIDI256**, sans postérieurs S708, sortie Frame Pitch, ni historique d'événements MIDI. Tester **trois modèles de cardinalité entraînés sur des prédictions de hauteurs hors entraînement**, et comparer leurs scores sur les mêmes 249411 trames GT actives. Rejeter les solutions régressives.

## Hypothèse

Un head acoustique `pitch_activity` formé de zéro avec labels FIT et partagé **uniquement à l'intérieur** de l'architecture Active-K peut restaurer une partie des informations de S708. Un deuxième `cardinality` apprend des cartes de probabilité **OOF**, et les variantes d'indices harmoniques et de variations temporelles sont des ablations falsifiables. Aucune tête ne réutilise les poids ou prédictions des cinq autres caractéristiques.

## Modifications

`src/causal_note/s825_crossfit_activek.py` avec `pitch_activity` causal entraîné par `HistGradientBoostingClassifier`, puis classifieurs K0–K6 `StandardScaler+LogisticRegression` utilisant l'activité des 49 hauteurs, leur compétition fondamentale/harmonique et le changement depuis le **précédent hop 256**, sans futur. `scripts/s825_train_crossfit_midi256.py` reconstruit 80 FIT originaux à partir des traces et DSP SHA; effectue 5 splits outer de compositions (48 TRAIN/16 inner-CAL/16 outer-TEST), puis **3 sous-plis internes d'entraînement**: chaque prédiction OOF de hauteur pour une composition TRAIN provient d'un modèle entraîné sur les deux autres compositions TRAIN. Une nouvelle tête privée `pitch_activity` est enfin entraînée sur les 48 TRAIN uniquement pour les sorties CAL et OUTER. Chaque variante cardinalité a ses propres poids; sélecteur issu de CAL exclusivement.

## Protocole

80 pistes GuitarSet, PCM mono original 44,1 kHz, `feature_hop_samples=decision_hop_samples=metric_hop_samples=256`, 306536 frames, 249411 GT K>0, H0 142213 corrects et S815 141946 corrects comme comparateurs. Préenregistrement de trois modes cardinalité `pitch_stats`, `pitch_harmonic`, `pitch_temporal`. Poids d'échantillonnage par vraie classe K conservés après échantillonnage; GT n'entre jamais en inférence. Calibration sur 16 autres pistes exclusivement. Mesurer score, under, over, vrai K1..6, K0 diagnostic et chaque composition. Référence S823 76,8775% SHADOW provient de S708 1024 et n'est **pas une baseline indépendante**.

## Audits obligatoires

Signature 80 enregistrements et cache S813, 5 splits 48/16/16 disjoints + 3× OOF sans composition commune au pitch teacher, invariance préfixe/causalité, entraînement de chaque cardinalité exclusivement sur prédictions crossfit, 3 têtes aux poids distincts. Test de contribution à 0 / 1 / 2 combinaisons supplémentaires dans chaque pli ; aucune sélection à posteriori sur outer. Aucun changement S708, MIDI LIVE, Onset, Offset, Frame Pitch, Exact-K ou Full Note. La résolution 5,805ms ne garantit pas la latence hardware: mesure distincte requise. Les 80 pistes FIT80 sont historiquement explorées, jamais une preuve de généralisation indépendante.

## Statut

Terminé : exécution scientifique réelle SUCCESS [#39587162104](https://github.com/Andriamarosoa/note/actions/runs/39587162104), 3 architectures et 5 folds. Meilleur mode pitch_temporal 55,2822 % FIT80 MIDI256; −1,7373 point vs H0 57,0195 %, donc rejet. Non promu, LIVE inchangé.



## Clôture scientifique vérifiée — 21 octobre 2026

**Le vrai run [#39587162104](https://github.com/Andriamarosoa/note/actions/runs/39587162104) a terminé SUCCESS**, commit `d012d7e116789efa50ab88ea9b789685bfb5bc67`, [artefact modèles, 80 prédictions et rapport #12339688454](https://github.com/Andriamarosoa/note/actions/runs/39587162104/artifacts/12339688454). Le premier run #39586398675 a été rejeté: l'ancien appel au module S822 subsistait dans la commande CI, bien que les tests S825 aient passé; cette erreur a été corrigée avant le deuxième run, **qui exécute réellement S825**. Il est interdit de confondre ses anciens chiffres S822 avec les résultats S825.

Le protocole réel utilise 80 pistes WAV GuitarSet originales SHA, 306536 trames 256 samples à 44100Hz, 249411 trames GT actives, 5 outer compositions de 16 pistes; chacune est exclue de 48 TRAIN et de 16 CAL disjoints. Les 48 TRAIN sont divisées en 3 groupes de 16, chaque groupe reçoit les scores d'une tête pitch apprise uniquement sur les deux autres groupes TRAIN. **Aucune prédiction pitch in-sample n'alimente un entraînement de cardinalité**. Un modèle final privé pitch est entraîné sur 48 TRAIN pour CAL/OUTER. Les trois classifieurs de cardinalité ont leurs propres poids et sont sélectionnés uniquement par inner-CAL. Aucun postérieur S708 ni MIDI émis, sortie/poids d'un autre axe, futur audio ou hold post-attaque. Aucun des scores ne provient d'une reconstruction MIDI final.

### Scores S825 GuitarSet FIT80 MIDI256

| Modèle | Bonnes décisions / 249411 | Active Exact-K | Undercount | Overcount |
|---|---:|---:|---:|---:|
| H0 MIDI historique, témoin | 142213 | **57,019538 %** | 80654 | 26544 |
| S815 indépendante, historique | 141946 | 56,912486 % | 65530 | 41935 |
| S823 SHADOW S708/1024 dépendant | 191741 | 76,877523 % | 34611 | 23059 |
| S825 pitch_stats | **133351** | **53,466367 %** | 82968 | 33092 |
| S825 pitch_harmonic | **137236** | **55,024037 %** | 80231 | 31944 |
| **S825 pitch_temporal (meilleur modèle fixe)** | **137880** | **55,282245 %** | **79097** | **32434** |
| S825 sélection inner-CAL | **136617** | **54,775852 %** | **78818** | **33976** |

Le meilleur mode fixe est **−4333 décisions** / **−1,7373 point** sous H0, malgré **1557 undercounts évités**, en contrepartie de **5890 overcounts supplémentaires**. La sélection CAL baisse encore de 1263 décisions par rapport au mode fixe temporal. Aucune promotion n'est justifiée.

### Lacunes par vrai K du meilleur modèle pitch_temporal

| Vrai K | Correct / frames | Précision conditionnelle | Sous-comptages | Surcomptages |
|---|---:|---:|---:|---:|
| K1 | 92024/112014 | 82,15 % | 11236 | 8754 |
| K2 | 16894/46311 | 36,48 % | 20711 | 8706 |
| K3 | 12258/34182 | 35,86 % | 13469 | 8455 |
| K4 | 13730/35190 | 39,02 % | 16709 | 4751 |
| **K5** | **2121/16323** | **12,99 %** | **12434** | 1768 |
| **K6** | **853/5391** | **15,82 %** | **4538** | 0 |

**Falsification** : les probabilités de hauteur natives256 issues d'un modèle privé n'ont PAS retrouvé la capacité de S708, notamment K5/K6; une simple somme ou un classifieur logistique à partir de la carte OOF de hauteurs ne suffit pas. La compétition harmonique augmente certains plis mais l'erreur dominante reste le **sous-comptage des fortes polyphonies**, et son correction aveugle crée des fausses K5/K6. Par composition, la variante temporale donne BN1 +617 et Funk2 +20 décisions vs H0, mais Jazz1 **−1217**, Rock3 **−1893**, SS1 **−1860**. Une amélioration moyenne pourrait cacher des régressions : ne pas l'accepter.

### Verdict et prochain diagnostic à mener

**S825 TERMINÉ — trois boucles de modèles indépendants vérifiées; toutes régressives sur H0 ; promotion FALSE.** Six axes CAL32 S708 inchangés, aucune unification, LIVE strict inchangé. Résolution de décision MIDI256 = 5,805ms, pas une mesure de latence matérielle. FIT80 déjà exploré, ne pas transformer ses chiffres en validation externe.

Pour la suite, rechercher une architecture **vérificateur ordinal conditionnel K3→K4→K5→K6** entraînée sur des sorties OOF de hauteur mais avec preuves acoustiques privées de séparation fondamentale/harmoniques. En parallèle évaluer **les erreurs du modèle pitch privé** par K et par registre, et tester la calibration de probabilité sur CAL indépendante, sans choisir les seuils en consultant les métriques OUTER. Conserver une variante BASE et des veto par K, composition, under/over avant toute promotion; effectuer une validation externe indépendante avant d'envisager un système unifié.
