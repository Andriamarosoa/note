# S737–S738 — architecture Exact-K dynamique pour toutes les fonctions

**R&D, non promu.** S708, S707, chemin LIVE et TEST intacts.

## Contrat commun
Pour chaque fragment de quatre hops de 256 échantillons à 44,1 kHz (23,22 ms), le routeur neuronal voit les 32 combinaisons H0-inclusives de H0 à H5, toutes les hypothèses K0–K6, leurs compatibilités et des audits de risques calculés exclusivement hors-fold d'entraînement. Il décide KEEP ou une combinaison pour tout le fragment. Les probabilités Onset de chaque hauteur MIDI restent indépendantes dans le décodeur. La combinaison peut changer au fragment suivant.

Le réseau ne reçoit ni annotation d'inférence, ni vrai K de son fragment, ni audio futur. Aucun masque ne peut être choisi a posteriori sur CAL. La meilleure combinaison moyenne sur GuitarSet n'est pas une politique dynamique.

## Phases réellement distinctes

| Fonction | État |
|---|---|
| Onset | S737 : entraînement FIT80 en cours, vrai H5 entraîné S734, 32 combinaisons neuronales |
| Sustain | Adaptateur causal six têtes S738 créé ; pas encore de poids entraînés ni de mesure |
| Release | Adaptateur causal six têtes S738 créé ; pas encore de poids entraînés ni de mesure |
| Offset | Nécessite un véritable adaptateur de décision NoteOff et S707 hazard ; ne pas le simuler en changeant Release |
| Identité des notes | Nécessite ADD/REMOVE/CHANGE_PITCH/REATTACK et un note-set, pas un simple classifieur Sustain |
| Exact-K | Distribution source S708 figée ; sélecteur historique K-conditionnel repris et dynamisé à l’échelle des fragments |

Les six indicateurs officiels (Onset F1, Exact-K, Exact-K actif, Offset accuracy, F1 notes complètes et Frame Pitch F1) sont des mesures de performance du système MIDI ; ce ne sont pas six cibles neuronales interchangeables.

## Protocole itératif S737
- FIT80 : deux initialisations, budgets 1/3/6 epochs, audit par composition exclue, cibles correction/régression du fragment.
- CAL32 : trois seuils prédéclarés par budget, soit 18 évaluations réelles ; un seul flux NoteOn/NoteOff et six scores.
- Gate : aucune baisse parmi six métriques, aucun ancien Offset correct ni Onset long perdu ; comparaison par piste et sous-groupes.
- HELD32 et DEV12 : seulement après gel de la sélection CAL ; leur exposition historique interdit d'appeler ces données « test aveugle ».
- TEST fermé ; LIVE non branché ; S708 reste officiel.
- L'objectif est strictement supérieur à 80 % pour chacun des six indicateurs, mais aucun apprentissage ne garantit ce résultat. Une boucle s'arrête à son budget maximal GitHub Actions et rapporte honnêtement ses métriques même si l'objectif est manqué.

## Limites scientifiques
Un choix de masque apprenable ne peut retrouver une hauteur jamais proposée. L'entraînement d'Onset utilise une cible locale par bloc (surrogate), qui ne garantit pas une meilleure note complète ni un meilleur Offset. Une correction proposée par une tête peut perturber des NotesOff futurs. Une validation honnête demande le vrai décodeur causal et toutes les métriques, jamais uniquement les sorties réseau.

Les prochains spécialistes sont conçus pour être entraînés séparément : Onset, puis Sustain, Release, Offset, identité des notes et Exact-K si une modification est pertinente, toujours en gardant des sources signées et des validations distinctes.

## Code et workflows
- src/causal_note/s736_fragment_mask.py : réseau neuronal class-conditionnel avec risques et KEEP.
- scripts/s737_train_fragment_onset_loops.py : apprentissage FIT, boucles CAL et garde-fous.
- src/causal_note/s738_multitarget_routing_contract.py : adaptateurs Sustain/Release et protections explicites.
- test/test_s737_fragment_training_loops.py et test/test_s738_multitarget_routing_contract.py.
- .github/workflows/note-s737-dynamic-fragment-onset-loops.yml et note-s738-multitarget-contract.yml.

**Ne pas promouvoir et ne pas annoncer >80 % sans rapport réel terminé.**
