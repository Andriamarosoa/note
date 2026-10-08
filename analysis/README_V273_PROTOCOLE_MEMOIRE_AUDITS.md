# V27.3 — Conservation des variantes et audits contextuels

## Contrat demandé

Chaque correction ou variante conserve une identité, ses propositions et ses
cas corrigés/régressés. Un faible gain, un bilan global négatif ou l'absence de
promotion ne suppriment pas une sélection candidate. Les variantes initiales
et calibrées restent distinctes. Une amélioration face à une variante et une
correction face à `freeze_local_combo` sont deux preuves différentes, à conserver.

Le réseau doit pouvoir exploiter les corrections, régressions et neutres du
**groupe complet**, avec les caractéristiques observables du fragment, le K
initial, le K proposé, les effectifs et la similarité des références. Aucune
mauvaise performance individuelle ne supprime les combinaisons de ses membres.

Une régression persistante rouvre l'audit de sa catégorie. Les possibilités
à départager sont une catégorie trop large, une mauvaise utilisation de la
preuve, un support insuffisant ou une caractéristique manquante. La seule
présence d'une erreur ne prouve pas automatiquement une nouvelle catégorie.
Les labels du fragment évalué ne doivent pas servir à inventer sa catégorie
puis à présenter le résultat comme une prédiction indépendante.

## Écarts constatés dans le code

1. Le catalogue S8 mesuré n'activait que les audits globaux. Les colonnes
   locales 3/4/5/7/8 des neuf descripteurs d'audit étaient mises à zéro : taux
   locaux de correction/régression/neutre, support local et distance.
2. Le registre historique ne couvrait pas tous les scripts : sa liste de
   préfixes omettait notamment `learn_` et les modules `v273_*`. Il ne
   conservait pas non plus les CSV/NPZ de preuve comme entrées d'inventaire.
3. La variante calibrée était évaluée comme remplacement possible ; ses
   apports doivent aussi être enregistrés comme candidature distincte,
   avec ses 251 améliorations et 232 dégradations face au S8 initial.

## Expérience fixée avant résultats

Réentraîner deux variantes locales du catalogue, à sept et huit sélections,
avec les producteurs immuables du run 37789646292 et le cache des spécialistes
du run 37777844182. Comparer aux modèles globaux archivés ; aucune modification
des propositions, de la perte catégorielle, des paramètres, des seeds ou des
30 époques. Les 127/255 groupes restent présents.

Activer toutes les statistiques locales calculées uniquement sur les références
autorisées. Elles utilisent les 64 voisins les plus proches de même K initial,
et pour chaque groupe le même K proposé, avec le lissage déjà fixé à 12. Ce
voisinage est un contexte de preuve ; ce n'est pas une nouvelle catégorie
définie après lecture des erreurs évaluées.

Vérifier l'influence effective à poids figés : après entraînement local,
neutraliser uniquement les cinq canaux locaux et comparer scores et décisions.
Publier les deux résultats ; cette intervention ne constitue pas un modèle
réentraîné sans audits.

Conserver dans une mémoire commune les sorties initiales, calibrées et locales,
leurs corrections, régressions, gains spécifiques et synergies. Relier chaque
événement à ses caractéristiques et à sa provenance. Le registre doit accepter
une variante même moins bonne globalement ; il ne doit pas substituer une
variante à son parent ni utiliser la vérité pour choisir la meilleure sortie.

Les versions calibrées du run 37803433204 utilisent des labels d'autres morceaux
du fold exclu. Elles restent des candidates expérimentales avec cette portée
explicite ; leurs sorties ne seront pas injectées comme nouvelle tête dans
une évaluation qui interdit tous les labels de ce fold. Un adaptateur et des
prédictions imbriquées compatibles seront nécessaires pour cette intégration.

## Preuves attendues

- Tests de conservation des données d'audit, de l'inventaire et des variantes,
  exclusions des labels et maintien des synergies.
- Identité des votes, propositions et audits bruts avec les archives ; preuve
  des cinq canaux effectivement présents dans les entrées du modèle local.
- Corrections/régressions par vrai K et fold, comparaisons au parent, cases
  complémentaires entre variantes et union descriptive de leurs corrections.
- Erreurs persistantes reliées à leur support local, taux M/N/neutre et distance,
  sans supposer qu'elles relèvent toutes d'une catégorie différente.
- Toutes les variantes restent candidates documentées, sans sélection de la
  meilleure sur les folds évalués et sans promotion automatique.

Folds 0/1/2/4 déjà exposés seulement. Fold3/player05 exclus. Aucune validation
sur données inédites n'est revendiquée. Les anciennes expériences restent
inchangées et leurs inventaires sont conservés ; le nouvel inventaire décrit
les références Git effectivement disponibles, pas une totalité invérifiable.
