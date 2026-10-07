# Informations spectrales brutes dans le réseau de comptage

L'audit `ac808794` a trouvé un signal utile avant normalisation, sans forte
amélioration du comptage. Ce test intègre cette information dans le CNN natif.
Il prédit uniquement K0–K6, sans identifier les notes.

## Comparaison fixée avant les résultats

- Population : 59 309 groupes, dont 7 385 polyphoniques ; folds 0,1,2,4.
- Pour chaque fold, apprentissage sur les compositions des trois autres folds ;
  fold 3 et player05 exclus de l'apprentissage et de l'évaluation.
- Référence : checkpoints `freeze_local_combo` archivés, rejoués avant le test.
- Les deux bras repartent du **même checkpoint uniforme propre au fold**, avant
  son ajustement pondéré, afin de ne pas ajouter des époques au candidat seul.
- La normalisation existante et les coordonnées spectrales sont conservées.
- Témoin : trois canaux supplémentaires répètent les valeurs normalisées.
- Traitement : trois canaux supplémentaires conservent les valeurs brutes / 12.
  Ces valeurs sont déjà relatives au fond acoustique ; ce ne sont pas des
  amplitudes physiques absolues.
- Même CNN et **864 paramètres supplémentaires** dans les deux bras. Les poids
  des nouveaux canaux sont nuls au départ ; tous les autres poids sont copiés.
  Les probabilités initiales reproduisent donc l'ancre uniforme.
- Huit époques d'ajustement, Adam 2e-4, pondération historique
  `targeted_k23_half_k4_quarter`, mêmes ordres de lots et état aléatoire initial.
  `candidate_norm`, `candidate_hidden2` et `cardinality` restent gelées.
- Même entrée spectrale 31×64×3 et même fenêtre de 4 096 échantillons à 44,1 kHz.
  Aucun échantillon audio ni résultat YourMT3+ supplémentaire.

Les huit époques reprennent le budget historique ; elles ne sont pas choisies
après lecture des scores de cet essai. Aucun meilleur checkpoint intermédiaire
n'est sélectionné. Le témoin supplémentaire contrôle la capacité et l'interface
ajoutées ; il peut lui-même différer de la référence historique après entraînement.

## Mesures et portée

Recompter les prédictions K0–K6, global/poly, sous/sur-comptages, corrections et
régressions face à la référence et au témoin de même capacité. Publier chaque
fold, les poids et les prédictions, même si le gain est nul ou négatif.

Un gain face au seul témoin ne suffit pas : la comparaison avec
`freeze_local_combo` et les pertes K0/K1 doivent être rapportées. Les folds ont
déjà servi à plusieurs recherches : résultat exploratoire, sans promotion
automatique ni prétention de validation sur un nouveau jeu indépendant.

L'analyse FFT multirésolution reste une intervention distincte ; elle n'est pas
mélangée à ce test pour permettre d'attribuer l'effet observé.
