# S415 — Décodage MIDI structuré, verrou **S80 Exact-K par événement**

**13 octobre 2026.** Verdict initial mesuré : **0 régression Exact-K**, résultat MIDI complet positif par rapport à S408/S412, mais dépend d'un autre réseau préentraîné. **Aucune promotion des références sans l'accord de l'utilisateur**.

## Objectif non négociable

Le décodeur des nouvelles notes n'a aucun droit de modifier la sortie du modèle comptage S80. Pour **chacun** des **59 309 événements GuitarSet**, il reçoit `K_i=S80_i` (K0…K6) puis génère **exactement** ce nombre de nouvelles notes MIDI. K0 = aucune nouvelle attaque, pas silence.

- K0 faux NoteOn interdits.
- Si le décodeur MIDI externe produit K'>K, il ne conserve que K notes par proximité temporelle au groupe, sans choisir sur la vérité.
- Si K'<K, il complète à partir des notes existantes S412, sans jamais abaisser K.
- Le temps d'attaque et de fin sont attachés à chaque nouvelle note, plutôt qu'une unique durée par événement. Les NoteOff/NoteOn d'une hauteur réattaquée sont des événements distincts.
- Le décodeur d'événements complet est actuellement un **hybride de démo de faisabilité** utilisant YourMT3+ préentraîné comme source de notes et notre propre S80 comme verrou cardinalité. Ce n'est **PAS un nouveau modèle natif entraîné**, ni de la distillation ni une transcription YourMT3+ de remplacement.

## Réutilisation des données vérifiées

- Entrée K immuable : run original [S80 S397 #38352988801](https://github.com/Andriamarosoa/note/actions/runs/38352988801) et source originale d'horodatages [S258 #38311525987](https://github.com/Andriamarosoa/note/actions/runs/38311525987). Score 53 529/59 309 correct global et 6 054/7 385 correct poly.
- Notes de YourMT3+ : vrais événements décodés par le checkpoint `YPTF.MoE+Multi (noPS)` depuis [run officiel #37605163312](https://github.com/Andriamarosoa/note/actions/runs/37605163312), fichiers `notes.jsonl` originaux des folds 0,1,2,4, **avant** d'ouvrir la vérité JAMS. Le source possède des `global_index` natifs quand les notes sont affectées à un événement.
- Repli lorsque YourMT3 sous-compte : notes S412 (K identique à S80), mêmes données de la source `S412_continuous_MIC_note_offset_4fold_predictions.npz` du [run #38375133317](https://github.com/Andriamarosoa/note/actions/runs/38375133317).
- La vérité GuitarSet `annotation.zip` est lue seulement après la production des notes pour mesurer les indicateurs, avec 190 pistes et 34 954 notes.
- Scripts `scripts/s415_s80_exactk_locked_yourmt_structured_event_hybrid.py` et workflow `.github/workflows/v273-s415-S80-K-LOCKED-full-note-proposals.yml`.

## K immuable — par vraie classe, même nombre de bonnes réponses qu'avant

| K réel | N événements | S80 corrects | S415 corrects | Régressions K |
|---:|---:|---:|---:|---:|
| K0 | 39 652 | 38 432 | 38 432 | **0** |
| K1 | 12 272 | 9 043 | 9 043 | **0** |
| K2 | 3 445 | 2 999 | 2 999 | **0** |
| K3 | 2 289 | 1 897 | 1 897 | **0** |
| K4 | 1 207 | 806 | 806 | **0** |
| K5 | 355 | 301 | 301 | **0** |
| K6 | 89 | 51 | 51 | **0** |

Tous les K sont **identiques** ligne par ligne : aucun gain K revendiqué, aucune correction K introduite. Score global S80 = **90,2544 %** ; poly K2…K6 = **81,9770 %**.

Les 33 475 événements NoteOn produits comptent 28 319 propositions MIDI YourMT3+ et 5 156 notes de secours S412. Toutes les 59 309 lignes et les NoteOn émis sont reliés explicitement au `source_row_id`, permettant la preuve de l'invariance **par relecture indépendante des fichiers JSONL** (les événements produits, pas seulement le tableau comptage du programme).

## Notes complètes — même vérité et ±50ms

| Métrique | S408 (référence Offset utilisateur) | S412 (challenger précédent) | **S415 (hybride sous K verrouillé)** |
|---|---:|---:|---:|
| Prédictions MIDI | 33 475 | 33 475 | 33 475 |
| Pitch et Onset corrects à ±50 ms | 25 120 | 25 120 | **27 670** |
| Fin Offset juste à ±50 ms parmi notes appariées | 13 050 | 14 313 | **23 115** |
| Offset conditionnel correct | 51,95 % | 56,98 % | **83,54 %** |
| **F1 note complète** | 38,14 % | 41,83 % | **67,56 %** |

Score YourMT3+ complet **sans verrou S80** du [S414](https://github.com/Andriamarosoa/note/actions/runs/38377420107) = 75,29 % de F1 note complète. Ce score seul ne respecte pas exactement notre S80 K : nous ne l'utilisons donc pas comme sortie de notre système.

**Gain face S412 : +25,73 points de F1 note complète**, exclusivement sur les hauteurs, les attaques et durées (pas sur K).

## Limitations qui interdisent une promotion scientifique générale

1. S415 **exige YourMT3+ et ses poids préentraînés à l'inférence** ; c'est une démonstration de l'intérêt du décodage musical structuré sous contrainte K, **pas une amélioration native indépendante de YourMT3+**.
2. Les prédictions YourMT3+, comme les modèles S80, ont une exposition connue ou possible à GuitarSet pendant l'entraînement. **Aucun score n'est une preuve de généralisation sur nouveaux morceaux**.
3. Choisir les notes YourMT3+ les plus proches du groupe et compléter de S412 est une règle fixe testée sur ces mêmes pistes ; elle ne résout pas la distorsion de K quand S80 se trompe.
4. S415 ne prétend pas avoir entraîné de réseau autonome Perceiver/Transformer ; le prochain travail doit entraîner une propre tête d'événements NoteOn/NoteOff à longue mémoire, éventuellement via apprentissage sur JAMS et distillation YourMT3+ en *train seulement*.
5. Une validation sur nouvelles compositions et sur une source différente doit vérifier simultanément **absence de régression K**, F1 complet, note-off des notes >=1000ms, K0 faux NoteOn et réattaques.

**Décision actuelle :** S80 K reste verrouillé et inchangé ; **S408 demeure référence Offset active choisie par l'utilisateur** et S412 son challenger précédent ; S415 est un **nouveau challenger expérimental** à comparer en conditions honnêtes. Aucune promotion officielle.
