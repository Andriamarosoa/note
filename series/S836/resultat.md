---
serie: S836
statut: en_cours
type: experience
reference: S708
branche: codex/note-s836-live-sharp-cache256
run_id: 39989523624
commit_sha: dad2ac7ff454014a8a2896d5fac7a4b407e7a9f7
promotion: false
---

# S836 — Rapport scientifique : première optimisation du Live NOTE PCM256

## Exécution
- Branche : [S836](https://github.com/Andriamarosoa/note/tree/codex/note-s836-live-sharp-cache256).
- [Run #39986203235](https://github.com/Andriamarosoa/note/actions/runs/39986203235) **SUCCESS**, 22 tests ; benchmark synthétique initial et deux passages réels de cinq pistes GuitarSet.
- [Run #39988295532](https://github.com/Andriamarosoa/note/actions/runs/39988295532) **SUCCESS**, 23 tests ; empreintes SHA-256 frame-par-frame pour entrée 288D, spectre 96D et toutes les décisions/sélections des six axes.
- [Run #39989523624](https://github.com/Andriamarosoa/note/actions/runs/39989523624) **SUCCESS**, 23 tests ; triple replay signé référence, cache SHA, cache sans SHA, mêmes cinq pistes réelles et mêmes poids.
- [Run #39987476143](https://github.com/Andriamarosoa/note/actions/runs/39987476143) **FAIL technique analysé** : instrumentation avait commenté accidentellement l'appel temporal `nn_post`, réparée et gardée par un test d'exécution dans `test/test_s836_incremental_sharp256.py`. Voir [audit](audit_ci_failure.md).
- Vrai corpus utilisé : 5 GuitarSet accompagnements `_comp`, un pour chaque BN1/Funk2/Jazz1/Rock3/SS1, exactement **20 062 fenêtres PCM256** et SHA authentiques ; les 75 pistes restantes ne sont pas encore réévaluées ici.
- Le mode `sharp_engine=reference` demeure le défaut, `incremental` est optionnel et SHADOW, aucun MIDI officiel n'est modifié.

## Métriques
| Caractéristique S834 SHADOW, cinq mêmes pistes | F1/accuracy avec et sans S836 |
|---|---:|
| Exact-K (accuracy) | **56,599541 %**, strictement inchangé |
| Active Exact-K (accuracy active) | **46,772514 %**, strictement inchangé |
| Onset F1 | **68,447542 %**, strictement inchangé |
| Offset F1 (événements) | **18,181818 %**, strictement inchangé |
| Frame Pitch F1 | **84,969450 %**, strictement inchangé |
| Full Note F1 | **67,717172 %**, strictement inchangé |

Source H0 même cohorte : Onset F1 **71,232877 %**, Exact-K 35,639517 %, Active Exact-K 33,446122 %, Frame Pitch F1 76,772655 %, Full Note F1 44,305284 %. Offset H0 **accuracy 62,888889 % : pas comparable à Offset F1**.

### Gain spectral séparé
Le FFT S706 médian de référence est passé de 0,811 ms à 0,258 ms sur le signal synthétique du premier CI, **~3,15 fois plus rapide** pour ce composant, avec `np.array_equal` exact sur tous les features 96D. Au troisième CI, médian 1,268 → 0,390 ms (~3,25×) : les performances absolues varient selon runner.

### CPU global sur les cinq pistes
| Run | S836 p95 (ms) | S836 p99 (ms) | Interprétation |
|---|---:|---:|---|
| 39986203235, sans SHA | 5,288–5,459 | 5,500–5,792 | Sous 5,805 sur ce runner |
| 39988295532, avec SHA | 5,692–5,760 | **5,849–6,079** | **Échec du critère p99 strict** |
| 39989523624, avec SHA | 3,288–3,670 | 3,800–4,710 | Sous 5,805, runner différent |
| 39989523624, sans SHA | 3,283–3,536 | 3,757–4,346 | Sous 5,805, même runner |

Les p95 S834 de référence varient eux-mêmes : 6,427–6,762 ms dans le premier CI, 6,766–6,962 ms dans le second et **3,845–4,166 ms dans le troisième**. Les p99 et coûts instrumentés dépendent donc sensiblement des conditions d'exécution. Aucun résultat p95/p99 Linux ne démontre la latence bout-en-bout sous Windows/iRig.

## Comparaison à S708
Le modèle S708 original utilisé comme source a été **entraîné à 1 024 échantillons**, puis inféré à chaque bloc 256 en S834. Cette optimisation ne résout pas l'obstacle du vrai nouvel apprentissage 256 ; elle conserve absolument le modèle et les six décodeurs. La référence H0 FIT80 Onset F1 73,252019 % sur 80 pistes reste protégée ; le S834 F1 71,382285 % sur les mêmes 80 est inférieur. Sur les cinq pistes de ce replay, S834 Onset 68,4475 % reste inférieur à H0 71,2329 %. **Aucune unification MIDI des six références, aucun score Live complet revendiqué** ; l'accuracy Offset historique n'est pas un F1.

## Audits
| Rôle | État | Preuve / limite |
|---|---|---|
| Données | PASS | 5 WAV originaux SHA, cinq compositions, 20 062 hops |
| Acoustique | PASS | SHA chaque frame 96D et 288D identique référence/candidat |
| Architecture | PASS | six sorties/sélections SHA identiques sur chaque frame, 9 600 invariances inter-axes ; autres HEAD/SELECTOR/DECODER et MIDI protégés |
| Apprentissage | PASS | aucun réentraînement ni lecture de vérité pendant inférence |
| Métriques | PASS | six compteurs événement par événement inchangés sur cinq pistes, mais aucun nouveau score 80 WAV |
| LIVE | FAIL | p99 >5,805 ms dans au moins un runner CI, aucun iRig Windows et aucune certification indépendante |

### Pourquoi la progression CPU varie
Hypothèse H1 soutenue : le mode original refaisait **deux** groupes de trois FFT par bloc ; le cache réemploie uniquement les spectres **passés** et évite le deuxième groupe, sans audio futur. La différence de coût S706 est mesurée sur signal identique.

La variation absolue inter-runs n'est **pas expliquée de façon unique** : le témoin S834 accélère ou ralentit lui aussi, et l'audit SHA change le travail de replay après `push`. Le troisième passage avec SHA puis sans SHA **ne démontre pas de pénalité uniforme** selon les pistes : Jazz1 sans SHA a un p99 supérieur à Jazz1 avec SHA, contrairement à BN1. Des écarts de puissance du runner, ordonnanceur, fréquence CPU, charge ou effets de cache sont possibles mais pas prouvés. **Ne pas contourner p99** en sélectionnant arbitrairement un run favorable.

## Verdict
**OPTIMISATION VALIDÉE en SHADOW pour la parité et l'accélération spectrale ; NON PROMUE en Live.** Les **six décisions privées sont rigoureusement identiques** sur les 20 062 hops testés. Le budget temps réel p99 robuste n'est pas démontré, ni l'intégration complète des six têtes au microphone, ni un gain musical. Les scores ne progressent pas parce que l'intervention visait exclusivement à éliminer un calcul redondant, et ne modifiait aucune politique musicale.

## Prochaine action
Étendre aux **80/80** WAV signés, dont solos, avant de déclarer le gain universel ; profiler le temps CPU par composant et le p99 sur Windows/iRig, sans ajouter d'attente ni futur audio. Ensuite, poursuivre **Onset** indépendamment pour rattraper H0 et travailler l'Offset sur sa **même** métrique, puis les autres axes, sans fusion ni promotion d'un score venant de cohortes incompatibles. Pour toute nouvelle solution produisant un gain inférieur ou une régression, effectuer un audit causal par TP perdus/gagnés, FP ajoutés/retirés, composition, polyphonie et registre avant nouvelle hypothèse. Les agents NVIDIA/Work personnalisés n'ont pas été appelés dans S836.
