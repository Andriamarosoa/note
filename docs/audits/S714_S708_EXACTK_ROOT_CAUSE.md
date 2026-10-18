# S714 — Audit indépendant du sous-comptage S708 / Exact-K / harmoniques

Date : 18 octobre 2026. **Statut : diagnostic VALIDÉ, aucun nouveau candidat promu**.

## Provenance et protocole

- Source : [run S708 39094682419](https://github.com/Andriamarosoa/note/actions/runs/39094682419), artefact immuable `S708_K_GUIDED_S706_JOINT_CAUSAL_LOOPS`, `held_events.jsonl`, `dev_events.jsonl`, `report.json`.
- Relecteur de code : `scripts/s714_audit_s708_cardinality.py`, avec [CI S714](https://github.com/Andriamarosoa/note/actions/runs/39108384709).
- Échantillons : **32 pistes HELD**, 132 288 trames ; **12 pistes DEV**, 48 970 trames ; toutes déjà exposées dans les expériences passées. Aucun nouveau TEST n'est ouvert.
- Les métriques officielles recalculées à partir des timestamps **disponibles**, des NoteOn/NoteOff réellement émis et des JAMS sont identiques à S708 à `1e-10`.
- Les annotations sont utilisées **uniquement pour l'audit après les prédictions** ; pas d'entraînement, réglage, futur PCM ou oracle au décodeur.

## 1. Sous-comptage dominant et coût de la correction S708

| HELD | S675 témoin | S708 |
|---|---:|---:|
| Exact-K | 51,81% | **62,24%** |
| Trames sous-comptées | 55 526 | **36 323** |
| Trames surcomptées | **8 223** | 13 629 |

| DEV | S675 témoin | S708 |
|---|---:|---:|
| Exact-K | 52,84% | **58,97%** |
| Trames sous-comptées | 18 167 | **13 658** |
| Trames surcomptées | **4 926** | 6 433 |

S708 récupère 4 509 trames sous-comptées sur DEV mais introduit 1 507 trames surcomptées supplémentaires relativement au témoin. Un rescuer non gardé peut donc améliorer le rappel sans améliorer le score global assez vite.

## 2. Exact-K DEV ventilé par véritable K annoté

| Vrai K | Trames | Sous-comptages S708 | Part |
|---|---:|---:|---:|
| K1 | 21 350 | 4 134 | 19,36% |
| K2 | 5 160 | 2 781 | **53,90%** |
| K3 | 6 394 | 3 750 | **58,65%** |
| K4 | 3 810 | 2 755 | **72,31%** |
| K5 | 366 | 238 | 65,03% |

Sur DEV `comp`, **11 342 / 24 485 = 46,32%** des trames sont sous-comptées, contre **2 316 / 24 485 = 9,46%** sur `solo`. Cette différence confirme la polyphonie comme principal goulot d'étranglement. Sur HELD, les taux correspondants sont 41,99% et 12,93%.

## 3. Gravité par registre de hauteur

Mesure : fractions de **trames-note** annotées non prédites. Ce n'est pas la même métrique que F1 Onset ou Exact-K.

| Registre | Trames-note réellement présentes DEV | Trames-note manquées | Proportion manquée |
|---|---:|---:|---:|
| MIDI 40–52 (E2–E3) | 19 935 | 8 856 | **44,42%** |
| MIDI 53–64 | 36 757 | 13 367 | 36,37% |
| MIDI 65–88 | 11 012 | 2 743 | 24,91% |

Les fondamentales graves sont proportionnellement les plus difficiles ; regarder les notes E2/B2 comme vrais événements, pas seulement l'harmonique supérieure associée.

## 4. « Harmoniques » : rester prudent sur la causalité

Le rapport repère les trames où une note **faussement prédite** se trouve à +12, +19 ou +24 demi-tons au-dessus d'une note **vraiment active**.

- DEV : **1 279 / 12 994** trames-note fausses correspondent à cette coïncidence (**9,84%**).
- HELD : **6 612 / 28 496** (**23,20%**).

Ce chiffre n'est **pas** la proportion de fausses notes acoustiquement causées par les harmoniques : un alignement tonal peut être fortuit, et les harmoniques de notes mal détectées ne sont pas dans la vérité « active ». Il réfute seulement l'hypothèse trop forte « toutes les erreurs d'accords viennent de l'octave harmonique ».

## 5. Défaut de politique Exact-K confirmé, à ne PAS réparer aveuglément

Dans la politique S708 figée :
`count_confidence = 1.1` et `guide_confidence = .5`.

- Une probabilité max Softmax ne dépasse jamais 1 : `JointSelector.step()` ne peut donc **jamais** appliquer sa contrainte ordinaire sur le budget K.
- `GuidedSelector.step()` utilise encore `K` à confiance >= .5 pour ajouter **des attaques faibles appuyées par des indices acoustiques**.
- Le test `test_s708_frozen_policy_does_not_strictly_enforce_k_zero` montre qu'un K0 extrêmement confiant peut coexister avec deux NoteOn forts. Ce n'est pas un bug d'exécution : c'est la sélection historiquement calibrée. Il est **interdit** de remplacer `1.1` par `.5` directement en LIVE sans mesure appariée des six scores : la tête K se sous-estime et peut supprimer des accords corrects.

Décision : **conserver S708 inchangé**, expérimenter sur CAL un budget K souple conditionné par indices de nouvelles fondamentales, en shadow ; ne permettre une promotion qu'après les six métriques sans régression.

## 6. Incohérence de cible : unisson à la même hauteur

Sur des segments GuitarSet, **plusieurs notes annotées simultanément partagent le même pitch**. Le flux MIDI existant représente une seule note active par pitch.

- DEV : **218 trames/48 970**, soit 0,45% des trames.
- HELD : **489/132 288**, soit 0,37%.
- Si l'on calcule **en plus**, sans remplacer la mesure officielle, le K des **hauteurs distinctes**, S708 passe de 58,97% → 59,24% DEV et de 62,24% → 62,43% HELD.

La différence reste inférieure à 0,3 point et **ne saurait expliquer les 21 points qui manquent pour 80%**. Le nouveau rapport a un champ `unique_pitch_exact_k_DIAGNOSTIC_ONLY` précisément pour éviter de changer de métrique rétroactivement.

## 7. Actions suivantes de développement / règles de sûreté

1. **K / accords** : conserver le gel S708 et entraîner/calibrer uniquement sur compositions FIT/CAL un spécialiste « nouvelle fondamentale indépendante » pour K2–K6. Comparer au contrôle S708 sans agrégation harmonique ; pas de sélection sur DEV.
2. **Décodage** : apprendre une politique K souple, et tester séparément le veto K0 et le rattrapage d'attaque K2–K6 ; ne pas l'activer sans contrôle des fausses attaques, durées et réattaques.
3. **S707 offset LIVE** : modèle hazard actuellement uniquement dans la chaîne de benchmark S708. Une intégration correcte doit coupler l'état NoteOff causal au même budget actif Exact-K, et ne jamais libérer rétrospectivement les slots d'autres attaques.
4. **Vérification** : un seul flux NoteOn/NoteOff, scores Onset F1, Exact-K, active Exact-K, Offset accuracy, full-note F1 et frame-pitch F1, comparés dans le même run sur CAL puis HELD/DEV. TEST inaccessible avant gel.
5. **Latence** : garder le S713 optimisé (p95 CPU ~5.34ms sous Ubuntu runner, pas encore certifié sur iRig Windows).

**Verdict :** sous-comptage polyphonique démontré ; absence de budget K fort confirmée ; erreur d'annotation unisson faible quantifiée ; coïncidence harmonique pas suffisante pour une explication monocausale. Pas de nouveau score de reconnaissance revendiqué ni référence officielle modifiée.
