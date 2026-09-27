# Apprentissage natif : 23 contre 31 trames, évaluation externe fold 3

## Question et périmètre figés

Le défaut de couverture est corrigé et audité. Il reste à mesurer si le réseau
qui compte K apprend mieux avec l'audio supplémentaire. L'expérience compare
le **réseau natif V26 à sept classes employé dans V27.3**, avec sortie directe
`argmax P(K=0..6)`, sur 23 et 31 trames. Aucun indice ln, nouveau correcteur,
seuil de sortie ou fusion supplémentaire n'est ajouté.

Ce test isole le réseau de comptage. Il **ne reproduit pas la chaîne V27.3
complète**, son ancre V10.4, ses fusions ou ses transitions. Son score ne doit
donc pas être présenté comme un remplacement de la référence 42,6019 %.

Comparaison principale : pondération uniforme. Comparaison secondaire prévue
avant résultat : pondération inverse de la racine de la fréquence des classes,
déjà utilisée dans V26. Chaque comparaison oppose seulement les fenêtres,
avec exactement les mêmes poids de classes. Aucun choix entre ces variantes
ne sera fait à partir du résultat externe.

## Entrées et partitions

Les 50 morceaux du fold 3 viennent de l'archive exacte à 31 trames déjà auditée,
SHA-256 `e885a2e100296e2ce257690b704432097869c2c7883654214bd4f9859ede4742`.
Ils ne sont pas recalculés : 15 279 groupes, dont 1 969 polyphoniques.

Les 190 autres morceaux sont préparés **une seule fois**, avec la même chaîne
de proposition figée et les positions entières exactes. Dix lots de 19 morceaux
partagent cette préparation entre tous les entraînements. Les étiquettes K
sont vérifiées indépendamment depuis les annotations ; les cas de plusieurs
attaques sur une même corde sont comptabilisés séparément de son occupation.
Les 23 premières trames et le calcul runtime sont comparés au format étendu.

Les 24 compositions conservent leurs folds historiques. Apprentissage interne :
folds 1, 2 et 4 ; validation interne : fold 0 ; réentraînement final : folds 0,
1, 2 et 4 ; **seule évaluation externe : fold 3**. Ni les pistes historiques
réservées, ni les autres folds comme tests externes ne sont évalués.

Un bundle commun contient une seule carte à 31 trames par ligne. Le témoin lit
simplement ses 23 premières trames. Caractéristiques, masques, statistiques,
positions, cibles, ordre des lignes et fichiers sources sont identiques.
Les lectures se font par lots depuis des fichiers mappés en mémoire.

## Apprentissage

Pour chaque pondération et chaque fenêtre : **8 époques internes puis 8 finales**,
budgets historiques du fold 3, figés avant cette comparaison. Pas d'arrêt anticipé
ni de choix d'époque sur le fold externe. Les réseaux repartent de zéro pour
chaque phase ; dernière époque conservée.

TensorFlow 2.15.1, NumPy 1.26.4, Adam 2e-4, lots de 128. Graines historiques V26 :
16164 en phase interne, 17064 en phase finale. Les deux durées conservent les
mêmes paramètres initiaux, mêmes poids de classes et même permutation explicite
à chaque époque. Les empreintes de ces éléments sont contrôlées après exécution.
Les poids de classes sont calculés exclusivement sur la partition d'apprentissage.

Les paramètres et poids terminés sont enregistrés avant la prédiction externe.
Chaque époque sauvegarde un checkpoint et un état de progression. Les fichiers
et rapports sont archivés dans une préversion GitHub propre à l'exécution.

## Audit prévu après entraînement

- Rejouer indépendamment l'argmax des probabilités enregistrées.
- Vérifier les partitions, paramètres initiaux, ordres de lots, poids de classes,
  huit époques complètes, empreintes des modèles et alignement des prédictions.
- Rapporter Exact K global et polyphonique, sous-comptages, surcomptages,
  matrice de confusion et résultats pour chaque vrai K et composition.
- Séparer corrections, dégradations et changements d'une erreur vers une autre.
- Examiner séparément les 87 groupes dont une attaque était hors fenêtre,
  dont 42 groupes polyphoniques, et les erreurs résiduelles.
- Produire un intervalle descriptif par rééchantillonnage des compositions ;
  ne pas présenter une graine et cinq compositions comme preuve définitive.

Le fold 3 a déjà servi à des diagnostics : ceci est une comparaison de
développement. La chaîne de proposition commune historique n'a pas été
réentraînée séparément dans chaque partition. Aucune promotion automatique
ni revendication d'avoir résolu toutes les causes d'erreur n'est prévue.
