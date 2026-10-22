# Résultat — série Onset ErrorPrediction 69D, hard negatives et réattaques

**Statut : expérimentation non promue, rapport numérique à obtenir par workflow.**

## Baseline vérifiée sur le même FIT80

S613 vrai 256 : TP8099, FP4600, FN3168, F1 67,5874155 %. Tête ErrorPrediction initiale 63D : TP8090, FP4553, FN3177, F1 67,6704308 % (+0,0830 point), 124 faux positifs contestés et 41 bonnes attaques contestées à tort. Référence historique S708 entraînée 1024 : 73,2520 % (non admissible comme poids natifs 256).

## Nouvelle expérience

Trois spécialistes indépendants `balanced`, `harmonic`, `restrike` (tous MLP neuronal 69→64→32→2 entraînés avec des événements audio PCM256 réels) dans chacun des cinq folds composition-disjoints ; spécialiste et seuil sélectionnés sur CAL uniquement. Les 6 caractéristiques additionnelles sont exclusivement l'état PRE-sélection Onset. L'entraînement utilise les labels TP/FP ajoutés **après** les prédictions, jamais GT dans le signal LIVE. Un second replay scientifiquement requis mesure en temps réel causal les 80 WAV/306536 décisions, vrais TP/FP/FN/F1, doutes corrects/erronés, par piste et par composition.

**Pas encore de nouveau F1.** Si le nouveau F1 régresse S613, CI scientifique FAIL, résultat conservé, aucune promotion. Même s'il augmente, la cohorte FIT80 n'est pas externe aux anciennes têtes : ce n'est qu'un gain expérimental, pas une validation industrielle.

Les cinq autres axes et le MIDI officiel sont gelés, aucun allongement de délai, aucune tête 1024.
