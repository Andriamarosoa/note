# S819 — Dossier commun d'audit à l'aveugle : Active Exact-K
## Contrat de neutralité

Document de données destiné séparément à plusieurs modèles NVIDIA. **Aucune solution n'est privilégiée**. Les réponses ne seront communiquées à aucun autre modèle avant la fin des appels. Les conclusions sont des hypothèses non validées jusqu'aux expériences authentiques. Ne pas envoyer de credentials, d'audio brut, de GT individuel ou de données CAL32/HELD/TEST.

## Mesures SCIENTIFIQUEMENT authentifiées (GuitarSet FIT80 historiquement exploré)

Fichier original 80 pistes mono GuitarSet, 44100 Hz, extraction nouvelle de 306536 trames au pas **256 échantillons = 5,80499 ms** (résolution de sortie, PAS latence totale). 5 compositions BN1, Funk2, Jazz1, Rock3, SS1, entraînement 48 pistes, calibration 16 pistes d'autre composition, évaluation 16 pistes d'une composition tierce ; 249411 trames vraies K>0, K0 exclu du score Active Exact-K. FIT80 a déjà été exploré : ce n'est pas une validation externe indépendante.

Version | correct / 249411 | précision | under | over
- H0 signé S708/S704 FIT80 : 142213/249411 = 57,01953803% ; under=80654 ; over=26544
- S814 classifieur/regresseur privé : 136360/249411 = 54,67280914%
- S815 classifieur + régresseur + pitch occupancy 49 experts privés : 141946/249411 = 56,91248582% ; under=65530 ; over=41935
- S816 pénalité surcomptage en calibration : 140728/249411 = 56,42413526% ; under=69540 ; over=39143
- S817 2 nouveaux experts K>=4 et K4/K5/K6 : 141946/249411 = 56,91248582% ; 0/306536 décisions changées, donc 0 gain.
- S818 6 classifieurs ordinaux indépendants pour K>=1,...,K>=6, modèles entraînés sur FIT80 : 135694/249411 = 54,40578002% ; under=73512 ; over=40205. Deux runs scientifiques GitHub SUCCESS et exactement mêmes résultats : #39546770781, #39546788233.

S818 scores OOF par composition :
BN1 27669/47770=57,9213 % ;
Funk2 27246/48827=55,8011 % ;
Jazz1 22959/40548=56,6218 % ;
Rock3 28487/54917=51,8728 % ;
SS1 29333/57349=51,1482 %.

S815 vrais K — audit de décisions exactes :
K0 43545/57125=76,23% (K0 NON dans Active-K denominator) ;
K1 87731/112014=78,32% ;
K2 19632/46311=42,39% ;
K3 15789/34182=46,19% ;
K4 12995/35190=36,93% ;
K5 4861/16323=29,78% ;
K6 938/5391=17,40%.
Les 5 folds S815 ont choisi `confidence_direct_pitch` shift 0. S817 a conservé ses décisions. S818 utilise parfois `expected`, parfois `crossing`; Rock3 et SS1 régressent.

## Code exact à analyser, sans interprétation pré-imposée

Repo public https://github.com/Andriamarosoa/note, branche `codex/note-s819-independent-model-review`.

1. `src/causal_note/s815_activek_teacher.py:45-75` :
- `_count_outputs(x,direct,regressor,pitch)` : `direct.predict_proba(summarize(x))`, `regressor.predict(summarize(x))`; `pitch.predict_proba(x.reshape(-1,222))[:, positive]` et somme des 49 probabilités.
- `_select(...)` : `kd=argmax(p)`, `kr=round(reg)`, `kp=round(pitch)` ; sélecteurs `direct`, `regression`, `pitch_sum`, `median`, `confidence_direct_pitch` (si max classe direct>=.65, autrement kp). Décalages -1,0,+1.
2. `src/causal_note/s818_activek_ordinal.py:39-94` :
- `training_samples` : max 130 exemples de chaque classe binaire 0/1 par enregistrement et par seuil K>=j ; `sample_weight=len(population_class)/n_sampled`; poids normalisés avant fit.
- `ordinal_probabilities` : six prédictions binaires distinctes sur `audio_features(X)`; projection `np.minimum.accumulate` pour garantir la décroissance des survies K>=j.
- `decide` : K=nombre de probabilités >=seuil 0.3/.45/.6/.75 OU K=round(somme(probas)-.35/0/+.35). Sélection d'une règle sur composition interne.
3. `src/causal_note/s817_highpoly_temporal.py` : contexte strictement passé aux décalages 1,4,16 hops, deux modèles privés (K>=4 puis K4/K5/K6), calibration sur composition distincte. Aucune information après l'instant actuel.
4. `scripts/s818_train_ordinal256.py` : cinq plis de source PCM SHA, revue composition-exclue 48/16/16 ; même score H0 et S815 vérifiés avant sortie des décisions ; aucune promotion.
5. `.github/workflows/note-s818-activek-ordinal.yml` : smoke tests et training vrais sur GitHub Actions.

## Contraintes indépassables de S819

- Six caractéristiques Onset F1, Exact-K, Active Exact-K, Offset accuracy, Full Note F1 et Frame Pitch F1, chacune poids propres et sorties SHADOW isolées. Toutes doivent dépasser 85% sur **validation externe** avant envisager une distillation / unification. Une amélioration d'une caractéristique ne garantit pas maintien des six métriques après unification.
- Audio entrée MIDI256 à 44,1kHz, causal sans future frame, hold, confirmation différée, retiming ou NoteOff retardé. La latence physique totale doit être mesurée, 5,80ms est seulement résolution hop.
- Interdiction d'utiliser FIT80 test composition pour entraîner/calibrer son fold, d'ouvrir CAL32/HELD/TEST réservés ou de choisir une politique après avoir consulté les scores de test. Pas de promotion si régression ; pas de confusion entre workflow PASS technique et performance.
- Six agents Work personnalisés (Données, Acoustique, Architecture, Apprentissage, Métriques, LIVE) = distincts des 6 audits déterministes de GitHub. Avis NVIDIA distincts de chacun. Aucun modèle externe n'a le pouvoir de modifier production ni de prouver ses propres résultats sans validation mesurée.

## Demande au modèle — **sans réponse suggérée**

Fournis trois ou davantage **hypothèses mutuellement vérifiables** expliquant spécifiquement pourquoi les têtes acoustiques autonomes S815–S818 plafonnent dans les morceaux inconnus de composition, notamment K4–K6 et Rock3/SS1. Pour chacune : preuves pour/contre parmi les nombres et la source, des éléments de code nommés, une hypothèse nulle, une ablation contrôlée qui pourrait la réfuter sur splits composition-disjoints, et les risques (fuites, fausses notes harmoniques/octaves, sur/sous-comptage, causalité, CPU/latence, régressions des autres sorties). Inclure critique des métriques et des erreurs éventuelles de définition des labels. Ne fixe pas d'avance architecture, famille de modèle, perte, sélection, délai, ou intervention DSP. N'invente aucun résultat de gain ni accord entre modèles.

Rapport en français avec rubriques : FAITS, HYPOTHÈSES CONCURRENTES, CONTREPREUVES, EXPÉRIENCES DE FALSIFICATION, IMPACT MIDI256, LIMITES ET INCERTITUDES.
