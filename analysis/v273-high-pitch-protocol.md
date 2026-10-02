# High Pitch Rescue v2 — protocole avant entraînement

## Question

Tester si les surcomptages observés avec le HPR v1 viennent de la construction
du flux compressé et de sa fusion directe avec la représentation normale.

HPR-v2 reprend le principe de la branche `dual-stream-bass` du dépôt
`Andriamarosoa/midi` sans copier son objectif de hauteur : la branche rescue
observe deux fois plus de passé réel, le compresse par moyenne de paires
(`AveragePooling(2,2)`) et reste séparée de la branche principale.

## Population

Partition interne inchangée : 43 357 groupes d'apprentissage et 15 952 groupes
de validation, 190 pistes internes au total. Aucun fold externe 3 ni Locked12.
Une graine, 12 époques, checkpoints descriptifs 4/8/12, époque 12 primaire.

Les six anciens cas K=3 de surcomptage HPR-v1 restent des cas sentinelles :
40904, 48187, 67182, 67297, 67316 et 67320. Ils ne servent à aucune sélection
de modèle ni correction de sortie.

## Représentation

Branche normale inchangée :
- fenêtres source 256 et 2048 échantillons ;
- mêmes 31 fins de trame et 64 fréquences ;
- aucun futur supplémentaire.

Branche rescue HPR-v2 :
- fenêtres source 512 et 4096 échantillons ;
- compression par moyenne de chaque paire d'échantillons ;
- sorties respectives 256 et 2048 échantillons ;
- spectres évalués au taux effectif 22 050 Hz afin de préserver la hauteur
  physique au lieu de fabriquer une transposition +12 ;
- historique réel supplémentaire maximal : 2048 échantillons (~46,4 ms) ;
- anticipation supplémentaire : zéro.

Le FIR 127 taps, la décimation impaire et l'extension par zéros de HPR-v1 sont
supprimés.

## Fusion

Les canaux compressés ne sont plus concaténés dans l'encodeur principal.

Le contrôle et le traitement possèdent la même architecture et les mêmes
paramètres. La branche rescue produit un résidu de sept logits K=0..6. Sa
dernière couche est initialisée exactement à zéro.

- `observed_only` : gate rescue fixe à 0 ;
- `with_high_pitch` : gate rescue fixe à 1.

À l'initialisation, les deux bras doivent reproduire exactement la prédiction du
contrôle quatre-canaux normal. Le contrôle ne doit recevoir aucun gradient dans
la tête rescue ; le traitement doit pouvoir apprendre cette contribution.

## Entraînement

- seed 16164, dropout seed 46164 ;
- TensorFlow 2.15.1, NumPy 1.26.4, opérations déterministes ;
- Adam 0,0002, batch 128, cross-entropy non pondérée ;
- même ordre d'exemples par époque ;
- aucun correcteur de sortie ;
- décodage argmax P(K=0..6).

## Mesures

Publier Exact-K global, polyphonique, par vrai K, sur/sous-comptages,
surcomptages K<4, strates de registre et comparaison appariée.

Le critère principal reste conservateur : gain polyphonique sans régression
globale ni sur K=1/2/3, diminution des surcomptages K<4, et amélioration de
K=3 avec grave. Les six sentinelles servent uniquement à expliquer le mécanisme.

Aucune promotion automatique. V27.3 reste la référence officielle.
