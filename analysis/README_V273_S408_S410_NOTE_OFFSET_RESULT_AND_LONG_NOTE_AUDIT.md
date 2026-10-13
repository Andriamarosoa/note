# S408–S410 — Onset + hauteur MIDI + Offset réellement mesurés sur GuitarSet

**13 octobre 2026 — verdict** : S408 améliore fortement la prédiction individuelle des fins de notes (**mesuré**, pas simplement proposé). Le correcteur S410 dédié aux notes longues **réduit** le nombre de notes complètes à ±50 ms ; il est **rejeté**. S80, S402 et S407 restent inchangés.

## 1. Signal, annotations, contrôles et population

- 59 309 événements acoustiques natifs issus de 190 pistes GuitarSet, 34 954 annotations de notes GuitarSet JAMS originales (onsets MIDI, pitch et **durées/offsets**).
- S80 S397, réf active **development count** `S80_S397_K5_GUARDED` : K0–K6, 33 475 nouvelles notes proposées au total ; 81,976980 % Exact-K polyphonique (GuitarSet développement avec données déjà exposées).
- S402, réf active **development note pitch** `S402_K_CONDITIONED_352D_MLP` : 33 475 prédictions MIDI avec composantes musicales séparées par composition lors de l'apprentissage.
- S407, réf temporelle **expérimentale** : un instant estimé par groupe, 84,8266 % F1 des attaques à ±20 ms sur cette population, avec apprentissage de la tête S407 séparé par composition.
- Les modèles Offsets ne voient ni le K réel, ni le MIDI réel, ni la durée réelle du morceau maintenu à l'écart ; seuls les éléments prédits, les activations de 42×49 et l'information temporelle acoustique sont des entrées d'inférence.
- Un seul début partagé par groupe S407 : les notes qui entrent dans le même K2..K6 n'ont **pas encore un début individuel**. Des notes de même hauteur simultanées sur deux cordes restent non distinguées dans la sortie S402.
- **Evaluation** : même piste + **même hauteur MIDI + onset prédit à ≤50 ms** ; appariement un-à-un ; Offset noté correct si son erreur absolue est ≤50 ms, puis autre métrique avec tolérance `max(50ms, 20 % de la durée)`.
- **Limites de généralisation** : les quatre folds des têtes S402, S407, S408 séparés selon les **compositions**, mais les modèles S80 en amont ont un recoupement connu avec GuitarSet. S408 est conçu après consultation des résultats précédents ; il n'y a **aucun nouveau corpus vierge** ni résultat officiel indépendant de la chaîne.

## 2. S408 — code, run, preuves

- Code `scripts/s408_train_pitch_conditioned_held_song_note_offsets.py`.
- [GitHub run S408 #38371603686](https://github.com/Andriamarosoa/note/actions/runs/38371603686), **success**. Artefact `11800020249`, `s408-real-note-offsets-pitch-conditioned-S80-S402-S407-songheld-NOPROMOTION`.
- **4 régressions HistGradientBoosting entraînées sur d'autres compositions** ; aucuns croisements de compositions entre apprentissage et test pour la tête S408.
- Chaque note candidate S402 reçoit sa hauteur MIDI, son Onset S407, le K prédit, la morphologie de ses fondamentales/harmoniques à l'attaque et les activations acoustiques intermittentes des événements suivants, jusqu'à 2,2 secondes. Ce n'est **pas** l'analyse continue des WAV : les instants sans événement sont partiellement observés.
- Modèle de référence : durée médiane globale / médiane par hauteur calculée **uniquement sur les morceaux d'entraînement**.
- Ablation : même classe de régresseur sans observations acoustiques futures.
- Aucune sortie K ni hauteur MIDI n'est modifiée ; seul le NoteOff de chaque note candidate est estimé.

### Mesures

Parmi **33 475 notes candidates**, **25 120** sont appariées par hauteur MIDI et Onset ±50 ms aux **34 954 notes de référence**, d'où **75,04 % de précision Onset+MIDI**, **71,87 % de rappel** ; ces deux valeurs **ne sont pas** les précisions Onset temporelles sans pitch de S407.

| Variante | Offset correct ±50 ms, conditionné sur Onset+MIDI correct | Taux conditionnel | F1 note complète (MIDI+Onset+Offset ±50ms) | MAE Offset apparié |
|---|---:|---:|---:|---:|
| Durée globale médiane | 7 543 | 30,03 % | 22,05 % | 153,49 ms |
| Durée médiane par MIDI | 7 806 | 31,07 % | 22,81 % | 152,97 ms |
| S408 sans acoustique future | 11 124 | 44,28 % | 32,51 % | 127,88 ms |
| **S408 avec acoustique future** | **13 050** | **51,95 %** | **38,14 %** | **105,43 ms** |

L'apport du futur est **4 034 cas corrigés / 2 108 régressions = +1 926 notes exactes nettes** face à l'ablation sans futur.

Tolérance Offset `max(50 ms, 20 % durée)` : S408 **14 332/25 120 = 57,05 % conditionnel**, F1 note complète **41,89 %**.

**S408 améliore concrètement la localisation des fins de notes**, mais le 51,95 % n'est pas la précision globale sur toutes les notes : le score complet est **38,14 % F1**.

## 3. S409 — audit des causes, sans nouvel apprentissage

- `scripts/s409_audit_S408_note_offset_errors_by_duration_and_K.py`
- [GitHub run #38372114698](https://github.com/Andriamarosoa/note/actions/runs/38372114698), **success**. Audit à nouveau sur les mêmes 25 120 notes et reproduction stricte 13 050 / 11 124 / 7 806.

| Durée réelle | Notes appariées | S408 correct ±50ms | Taux | Déficit |
|---|---:|---:|---:|---|
| <100 ms | 1 247 | 503 | 40,34 % | Surprédiction de durée fréquente |
| 100–249 ms | 13 085 | 9 097 | **69,52 %** | Meilleure classe |
| 250–499 ms | 7 300 | 3 208 | 43,95 % | Sous-prédiction de certaines durées |
| 500–999 ms | 2 751 | 242 | **8,80 %** | 2 441 Offsets >50ms trop tôt |
| 1000–1999 ms | 678 | **0** | 0 % | Durée systématiquement raccourcie |
| >=2000 ms | 59 | **0** | 0 % | Durée systématiquement raccourcie |

Par K **prédit par S80** sur les mêmes notes correctement appariées par pitch et Onset :

| K prédit | Nombre de notes | Offset correct ±50ms |
|---|---:|---:|
| K1 | 8 414 | 57,57 % |
| K2 | 6 002 | 47,23 % |
| K3 | 6 250 | 53,34 % |
| K4 | 2 814 | 50,43 % |
| K5 | 1 381 | 39,61 % |
| K6 | 259 | 27,41 % |

**K0 n'a aucun NoteOn** et donc aucune nouvelle note à laquelle affecter un Offset. Les notes déclenchées avant K0 peuvent toutefois se terminer durant ce K0 ; il faut un état temporel suivi en continu.

## 4. S410 — correction longue dédiée, rejetée après mesure

- Script `scripts/s410_long_note_offset_expert_train_only_gate.py`.
- [GitHub run #38372660830](https://github.com/Andriamarosoa/note/actions/runs/38372660830), **success**, mais résultat **négatif**.
- Le modèle n'utilise en inférence que K et les durées/scores de sortie déjà prédits par S80/S402/S407/S408. Pour chaque fold externe, un **second split entre compositions d'entraînement** choisit le seuil du classificateur de « note >=500 ms » sans consulter la composition du fold externe.
- Le classificateur et le modèle de durées longues tentent d'allonger les Offsets S408 ; la prédiction S410 ne raccourcit jamais une note déjà estimée par S408.
- Deux des quatre folds ont interdit les corrections car leur validation interne ne montrait pas de gain. Les autres ont corrigé 395 et 595 notes respectivement. Les vrais morceaux tenus à l'écart restent sous-échantillonnés et hétérogènes.

| Variante | Offsets corrects ±50ms / 25 120 | Précision conditionnelle | F1 note complète | MAE |
|---|---:|---:|---:|---:|
| **S408** | **13 050** | **51,95 %** | **38,14 %** | 105,43 ms |
| S410 | 12 999 | 51,75 % | 37,99 % | 104,07 ms |

S410 améliore légèrement la MAE, sauve **6** notes de durée ≥1s auparavant systématiquement ratées, mais **perd 51 notes exactes nettes** à ±50 ms et retire 49 succès sur les notes 100–499ms. **Rejeté : S408 demeure le meilleur décodeur Offset expérimental selon le critère principal.**

## 5. Décision de conception suivante — vraie survie temporelle, pas une correction statique

Les Offsets des notes longues ne peuvent pas être déduits robustement des **seules observations intermittentes centrées sur de nouveaux événements**, même en leur ajoutant un correcteur long S410.

La prochaine architecture devrait :
1. Suivre chaque note `(piste, pitch, onset, état actif)` sur les *WAV complets*, **y compris entre attaques** ; ne pas confondre K0 et silence.
2. Réutiliser le **Frame** et le vecteur harmonique de S404 ainsi que les caractéristiques de flux déjà présentes dans S80, comme **entrées de suivi temporel**. Ne pas réinventer un autre système de comptage K.
3. Ajouter une tête `P(NoteOff dans ce bloc | pitch, note active, âge, trajectoire énergétique)` (hazard/survival) et une régression résiduelle en ms autour de la fermeture ; tests de réattaque et d'étouffement.
4. Sur les annotations de notes, séparer **cibles de note réellement active** et **l'énergie des harmoniques** qui peuvent subsister ; ne pas étiqueter les harmoniques comme nouvelles notes.
5. Entraîner sur compositions distinctes et comparer note Offset ±50ms, MAE, F1 complet, recall des notes ≥500ms et ≥1000ms ; refuser toute amélioration entraînant trop de régressions K1..K6.
6. Réaliser ensuite une validation complète extérieure à GuitarSet déjà exposé par S80.

**References restent inchangées :** S80 S397 **81,98 %** Exact-K poly sur développement ; S402 **36,76 %** accords poly complets ; S407 **84,83 %** Onset F1 ±20 ms ; S408 **38,14 %** F1 notes complètes ±50 ms (GuitarSet développement conditionnel à toutes les limites listées). **Aucune promotion officielle.**
