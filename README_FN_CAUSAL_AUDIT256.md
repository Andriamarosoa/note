# NOTE — audit FNs Onset PCM256 falsifiable (avant toute correction)

**Branche expérimentale :** `codex/note-fn-causal-diagnosis-256`

## Question précise

La version native S613 entraînée à 256 prend **8 099 bonnes attaques**, génère **4 600 fausses attaques** et manque **3 168 attaques** sur GuitarSet FIT80 (80 pistes ; 306 536 trames 256). La nouvelle tête de doute MLP69, correctement câblée sans réarmement, élimine **189 FP** mais aussi **45 TP**, soit une sélection finale TP8054 / FP4411 / FN3213, F1 **67,8746 %**. Il faut distinguer deux causes : **lacune de reconnaissance de nouvelles attaques par le modèle** versus **blocage par la politique d'émission**.

**La troisième tête ErrorPrediction n'est pas une tête de récupération : elle ne peut qu'accepter ou contester un NoteOn déjà proposé par le Main.** Elle ne peut donc pas retrouver les 3 168 FN du Main. Une éventuelle nouvelle tête FNRecovery sera seulement justifiée après audit de son information disponible, de sa capacité d'identification et de ses faux positifs.

## Hypothèse PRÉ-ENREGISTRÉE avant accès aux 80 pistes de l'expérience

`H_FN_MODEL_01` : **au moins 50 % des 3 168 FN** présentent, dans les trames de ±80 ms autour de l'attaque annotée, une probabilité Onset inférieure au seuil natif S613 (frontal lui-même faible), plutôt qu'une attaque forte bloquée uniquement par la machine d'état. Ce seuil de 50 % est figé dans le code *avant* le replay. Si moins de 50 % tombent dans ces catégories, **hypothèse réfutée ou incomplète**, et il faut rouvrir l'audit avant de corriger la tête principale.

Ce test établit uniquement une **association descriptive** entre FN et scores faibles ; il ne prouve pas une causalité au sens d'une intervention. Pour identifier la cause logicielle, il faudrait ensuite tester une correction isolée avec prédiction quantitative préalable sur un replay causal réel, conformément à la règle générale [issue #140](https://github.com/Andriamarosoa/note/issues/140).

## Expérience d'identifiabilité : peut-on même proposer les FN sans regarder le futur ?

Trois détecteurs **indépendants de propositions Onset uniquement**, strictement causaux, sont figés avant le test. Ils reçoivent la même sortie S613 native 256 et chacun possède sa propre machine d'état (aucun état/poids de S613/S701/MLP69 modifié) :

| Générateur de candidats | Minimum P(onset) | Minimum P(sustain) | Re-arm |
|---|---:|---:|---:|
| `sensitive` | 0,06 | 0,10 | 0,025 |
| `moderate` | 0,12 | 0,20 | 0,05 |
| `cautious` | 0,20 | 0,30 | 0,08 |

Les autres paramètres temporels sont conservés : une trame de confirmation, période réfractaire 10 trames et aucune attente après une attaque. Chaque proposition est une hypothèse d'événement à l'instant actuel, et **n'est jamais émise dans le MIDI officiel**. Après *la fin de l'inférence audio*, les propositions sont comparées uniquement aux **GT réellement non retrouvées par S613**, selon l'appariement exact 80 ms historique. Cela mesure la **couverture maximale des FN par un oracle connaissant les annotations**, pas la performance du système. Ce n'est pas un score de modèle et il est interdit d'en annoncer le F1 comme un gain réel.

Le programme fournit, pour les FN restants : le maximum de P(onset) et P(sustain) sur la fenêtre évaluée, la qualité S701, l'état de réarmement et de réfractaire de la tête Onset avant la décision, la couverture candidate par variante, la hauteur et la texture solo/comp. On classe chaque FN **une seule fois** en catégories descriptives :

1. `acoustic_onset_below_0_12` ;
2. `main_onset_below_calibrated_threshold` ;
3. `sustain_below_main_threshold` ;
4. `onset_and_sustain_not_simultaneous` ;
5. `selector_not_armed` ;
6. `selector_refractory` ;
7. `selector_budget_timing_or_assignment` ;
8. `no_frame_within_80ms`.

Ces classes peuvent être liées à une cause, mais leur nom **n'est pas une preuve de causalité**. La différence entre FN à probabilités faibles et FN bloqués par la sélection évite de proposer un mauvais correctif.

## Annotations et protocole

Le moteur réel `Stream` est appelé exactement sur **256 nouveaux échantillons PCM44 100 Hz**, soit une cadence de 5,805 ms, avec les poids d'origine **réellement entraînés nativement à 256** (SHA-pinned `S613`, `S701`). Le frontal utilise les fenêtres **passées** 2048/4096/8192 pour la fréquence, sans accès aux trames futures. Un observateur `FNExplorer` entoure les têtes S613/S701 sans modifier ni les probabilités ni les machines d'état principales. Il enregistre, pour chaque proposition permissive, le vecteur pré-décision original **69D** sans étiquette, et les états pré-décision pour chaque hauteur. **La GT n'entre qu'après `close_proposals()` et la fin du WAV**, pour produire les annotations FN et GT-FN-possible.

La cohorte FIT80 contient une **fuite historique de training** pour les têtes de référence : S613 avait vu 20/80 en apprentissage, S701 48/80. Le résultat teste nos hypothèses sur la cohorte signée, pas la généralisation du pipeline à des sessions jamais vues.

## Fichiers

- `scripts/audit_onset_fn_candidates256.py` : audit falsifiable, SHA, 80 vrais WAV, comptes et catégories exclusives, propositions phonétiques indépendantes.
- `tests/test_audit_onset_fn_candidates256.py` : smoke de causalité, invariance de S613, annotation a posteriori, candidats et vecteur de future tête 69D.
- `.github/workflows/onset-fn-causal-audit256.yml` : CI technique puis replay signé et diffusion d'artefacts.
- `series/ONSET_FN_CAUSAL_AUDIT256/resume.md` et `resultat.md` : chronologie sans confusion d'une hypothèse, des preuves et du résultat.

Un apprentissage FNRecoveryHead neuronal distinct n'est justifié **qu'après** le résultat quantitatif de la couverture et des causes. Pour la construire, on exploitera seulement les vecteurs de candidats 69D causaux et les étiquettes supervisées attachées *après* l'inférence, avec TRAIN/CAL/OUTER disjoints par composition. Les résultats devront être évalués en **vrai flux MIDI256**, contre les mêmes FN/FP, sans modifier les autres têtes.

**Statut : diagnostic en attente du replay, aucune promotion, aucun gain inventé.**
