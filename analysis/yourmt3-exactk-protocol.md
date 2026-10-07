# YourMT3+ — comparaison exploratoire Exact-K

Protocole fixé le 7 octobre 2026, avant toute observation des prédictions.

- Référence : `freeze_local_combo`, checkpoints du run `37271262794`, sans
  transplantation de neurones ni correcteur.
- Population : les 59 309 lignes des folds 0, 1, 2 et 4 du bundle natif
  `v273-window-pair-36351028493`. Fold 3 et player 05 exclus des évaluations.
- Contrôle préalable obligatoire : reproduire les annotations de chaque groupe
  exactement, puis les Exact-K de la référence (48 454 corrects au total,
  2 530 sur 7 385 groupes polyphoniques).
- Modèle : checkpoint préentraîné `YPTF.MoE+Multi (noPS)`, configuration par
  défaut de la démo officielle YourMT3+, précision float32 sur CPU. Révision
  source `5e66c1ea173a8186e0d20432b841d3180cc015b5`, poids SHA256
  `ae38e415c79efd5592dcb9b658cdb99ddb11d4c4e1eaa364cab04a052473fc25`.
- Aucun entraînement, recherche de seuil, décalage temporel ajusté, filtre de
  hauteur ou sélection d'instrument selon les résultats.
- Audio : même mix pickup mono GuitarSet, rééchantillonné de 44 100 à 16 000 Hz
  avec torchaudio, puis segments successifs de 32 767 échantillons selon la
  démo. Décodage et fusion des notes officiels, conservation des tokens bruts.
- Pour K : compter les nouvelles attaques décodées affectées à chaque groupe,
  et non les notes tenues. Tolérance d'affectation native de 882 échantillons
  (20 ms). Une attaque est affectée une fois au groupe candidat le plus proche,
  avec priorité au premier groupe en cas d'égalité. Tous les instruments et
  événements de note décodés sont conservés. Les événements hors groupe sont
  rapportés séparément. La classe 6 est plafonnée à 6 comme la cible existante ;
  les comptages bruts supérieurs à six restent exportés.
- Les bornes originales des groupes proviennent des horodatages entiers avant
  troncature, conservés dans les dix archives `training-part-*.zip`. Chaque shard
  doit correspondre à l'empreinte SHA256 du manifeste du bundle. Les features
  du bundle sont en float16 et ne servent pas à reconstruire le temps. Comme les
  groupes sont disjoints et larges d'au plus 40 ms, l'affectation au groupe par
  distance à l'intervalle équivaut à celle par candidat le plus proche pour le
  rayon natif de 20 ms. Cette propriété est testée contre les listes complètes
  de candidats synthétiques et vérifiée sur toutes les annotations réelles.
- Sorties : Exact-K0 à K6, global, poly K2–K6, sous-/sur-comptage, matrices de
  confusion, corrections, régressions, net, détail par fold et par ligne,
  notes et attaques non affectées, erreurs de décodage, temps d'inférence.

Correction technique avant mesure : le run `37604653992` s'est arrêté au
contrôle de précision des coordonnées float16. Aucun score YourMT3+ n'a été
observé dans ce run. Le lancement 2 récupère les horodatages intégraux des
caches originaux ; le protocole d'affectation et les critères sont inchangés.

## Portée de la conclusion

GuitarSet figure dans l'entraînement publié de YourMT3+. L'absence de recoupement
entre ces poids et nos enregistrements n'est pas établie. Il s'agit donc d'une
comparaison descriptive préentraînée, pas d'une validation indépendante sur
des données garanties inédites. Le modèle reçoit aussi davantage de contexte
audio que la référence causale : aucun résultat ne démontre une amélioration
à latence égale. Aucune promotion automatique de la référence.

Sources officielles :
- https://huggingface.co/spaces/mimbres/YourMT3/tree/5e66c1ea173a8186e0d20432b841d3180cc015b5
- https://arxiv.org/abs/2407.04822
