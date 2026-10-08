# Audit de la surestimation — protocole de rejeu figé

Question : pourquoi le sélecteur S8 annonce-t-il +397,42 corrections nettes
alors qu'il n'en réalise que +38 ? Le contrôle à sept sélections présentait
déjà +383,20 annoncé contre +60 réalisé.

Cet audit conserve tous les modèles et leurs poids du run 37789646292.
**Aucun entraînement, optimiseur, ajustement de calibration, choix de seuil
ou sélection de modèle.** Les archives et les entrées originales sont
contrôlées par SHA-256 ; le rejeu des sorties externes doit reproduire les
scores archivés à une tolérance numérique maximale de 0,00002 et le même net.

## Vérifications

1. La perte est une BCE de justesse du verdict initial + une CE conditionnelle
   lorsque le verdict initial est faux. Elle doit correspondre à la
   log-vraisemblance de la distribution jointe K disponible/OTHER. Une seule
   contribution par événement, sans compter les 127/255 groupes comme des
   observations indépendantes.
2. Décomposer l'écart des probabilités de correction et de régression, puis
   comparer la calibration des décisions retenues aux autres décisions.
3. Comparer, sur la **même population** ayant une alternative, un choix
   uniforme parmi les K distincts disponibles au choix du meilleur score.
   Examiner ensuite la restriction aux scores positifs. Ce diagnostic
   permet de distinguer maximisation entre alternatives et sélection des
   scores les plus optimistes. Aucun de ces choix ne devient une nouvelle règle.
4. Reconstituer les entrées d'entraînement, hors fold pour leurs producteurs,
   et les entrées des folds externes. Rejouer les poids finaux sur les deux.
   Les premières sont vues par le sélecteur : leurs bons résultats ne sont
   pas une évaluation indépendante. Comparer les pertes probabilistes
   finales et les taux de correction réellement associés aux scores.

## Écart de régime des producteurs

Pendant l'apprentissage du sélecteur, les votes proviennent de producteurs
entraînés sur deux folds. Les historiques d'un récepteur reposent sur les
deux autres folds d'entraînement ; les producteurs de ces références ont
été entraînés sur un seul fold.

À l'évaluation externe normale, les votes proviennent de trois folds ; les
historiques utilisent trois folds de référence, dont les producteurs ont
été entraînés sur deux folds. Ces exclusions évitent l'usage des labels du
récepteur, mais les tailles d'entraînement et de référence diffèrent.

Pour examiner la sensibilité à cet écart avec les **mêmes poids**, rejouer :

| Diagnostic | Folds ajustant les producteurs des votes | Folds de référence des historiques |
|---|---:|---:|
| Normal | 3 | 3, références produites hors fold avec 2 |
| Votes seulement alignés | 2 | 3, références produites hors fold avec 2 |
| Historiques seulement alignés | 3 | 2, références produites hors fold avec 1 |
| Deux régimes alignés | 2 | 2, références produites hors fold avec 1 |

Pour chaque fold externe, appliquer chacune des trois omissions possibles
d'un fold d'entraînement. Publier tous les résultats et leur moyenne,
sans choisir la meilleure omission ni fusionner les prédictions. Chaque
variante exclut le fold évalué de tous ses producteurs. La normalisation
du contexte reste celle du modèle figé, ajustée sur ses trois folds
d'entraînement. Les historiques locaux restent désactivés comme dans le
modèle initial ; l'audit porte ici sur la confiance, pas sur une variante locale.

Ces interventions peuvent établir une sensibilité aux entrées. Elles ne
permettent pas d'attribuer toute la surestimation à une cause unique et ne
constituent pas des modèles réentraînés selon un protocole homogène.

## Limites d'interprétation

Les folds 0/1/2/4 ont déjà servi au développement. Fold3/player05 restent
exclus. Les lignes d'entraînement apparaissent dans trois modèles externes,
contre un pour les lignes évaluées : les comparaisons agrégées portent sur
des **moyennes par événement/modèle**, pas sur un nombre indépendant accru.

Seuls les poids de fin d'entraînement sont disponibles. Le rejeu peut
mesurer l'écart entraînement/évaluation final ; il ne peut pas dire à quelle
époque un éventuel surapprentissage apparaît. La baisse de perte enregistrée
aux époques 1/16/30 ne remplace pas cette mesure.

La présence d'une normalisation correcte ne démontre pas la calibration.
Inversement, une surestimation ne prouve pas à elle seule une erreur de
formule, une fuite de labels ou une absence de signal acoustique.
