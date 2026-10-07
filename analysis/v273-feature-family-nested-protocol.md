# Sélection des familles de features sur FIT exclusivement

Préenregistré le 7 octobre 2026, avant cette nouvelle évaluation, depuis
`a39dfbb8ff357d2c27b3909c28d2f3fe456512cb`.

## Défaut d'évaluation confirmé

Le script `audit_v273_residual_feature_family_lofo.py` entraîne chaque modèle
sur FIT puis choisit sa famille et son seuil sur les prédictions VAL réunies.
Le bilan +20 (53 corrections / 33 régressions à 0,59) du run `37547761255` est
donc un bilan de sélection interne, pas une mesure après sélection sur FIT.
Le résultat à 0,50 reste une description de chaque famille fixée.

Cette étude corrige cette séparation. Elle ne modifie ni les gabarits ni le
réseau et ne prétend pas rendre vierges les folds déjà explorés. Aucune donnée
ni aucun résultat du fold 3 ou de player 05 ne sera chargé pour cette étude.

## Données et extraction figées

- Exports vérifiés du run `37356100423`, folds 0, 1, 2, 4 exclusivement.
- Mêmes FIT/VAL, B_low, base_K3, masques valides ; 1 666 lignes physiques,
  122 pistes, 845 lignes VAL. Autres vrais K inclus dans le bilan.
- Résidus historiques conservés par apparition FIT/VAL, sans les remplacer
  par une réextraction stable ; réajustement témoin reproduisant les probabilités
  d'origine à 1e-12 près.
- Les huit features supplémentaires sont exactement celles du run `37547761255`,
  extraites par `inference_features` actuel : géométrie (3), attaque (5).
- Utiliser les horodatages et identités déjà exportés, vérifiés contre la config.
  **Ne pas charger le bundle global** : il inclut des partitions hors périmètre
  et n'est pas nécessaire à cette extraction. Vérifier le fold avant décodage.
- Archiver une fois les features supplémentaires et les entrées réduites,
  puis travailler uniquement sur ces fichiers. Aucune fréquence annotée en entrée.
- Contrôler les MD5 GuitarSet, les SHA des exports et des dépendances.

## Sélection imbriquée

Pour chaque fold VAL externe autorisé, employer exclusivement son FIT exporté.
Effectuer trois rotations internes par fold de provenance : chaque ligne FIT
reçoit une probabilité produite par un modèle entraîné sur les deux autres
folds. StandardScaler et LR sont ajustés uniquement sur leurs K2/K3 d'entraînement.
Même LR équilibrée, C=1, random_state=28431, max_iter=3000.

Quatre familles figées : `base`, `base_geom`, `base_attack`, `base_geom_attack`.
Les définitions et l'ordre sont repris tels quels du script existant.

### Politique principale : seuil 0,50 inchangé

Choisir la famille au meilleur net Exact-K sur les prédictions internes FIT ;
égalité : moins de régressions, moins d'actions, ordre des familles. Si son net
n'est pas strictement positif, s'abstenir. C'est la règle des précédents audits.
La décision VAL utilise uniquement cette famille choisie sur FIT. Les quatre
modèles finaux sont aussi ajustés sur FIT pour les contrôles descriptifs figés ;
leurs résultats VAL ne peuvent pas modifier le choix.

### Contrôle secondaire : grille robuste existante sur FIT

Reprendre la grille déjà publiée 0,30..0,80 par 0,01, et ses contraintes : net
total positif ; aucun net négatif par fold interne, interprète ou style. Par
famille, égalité départagée par net, corrections, puis seuil plus haut.
Parmi les familles enrichies, départager par net, AUC FIT K2/K3, puis moins de
features, puis ordre. Utiliser une famille enrichie seulement si son net FIT
dépasse strictement celui du témoin `base` robuste (ou zéro s'il s'abstient).
Sinon garder le témoin ; si aucun candidat admissible, s'abstenir.

Publier aussi ce témoin robuste choisi sur FIT seul. Ne pas choisir après coup
entre politique principale et contrôle secondaire à partir de VAL.

## Contrôles et décision

Reproduire d'abord les résultats descriptifs à 0,50 des quatre familles et le
bilan fixe `base_geom_attack@0,59`, sans s'en servir pour sélectionner. Exiger
la reproduction des décisions et comptes du rapport source. Archiver features,
états des 16 modèles finaux et 48 modèles internes, probabilités, choix et SHA.
Le rejeu doit recalculer les scores, choix et bilans sans audio ni réajustement.

Tests : exclusion du fold 3, séparation des enregistrements FIT/VAL, rotations
internes, abstention, frontières de seuil, et invariance du choix quand les
labels VAL changent. Conserver la base en cas de régression. Un éventuel gain
interne reste exploratoire ; aucune promotion automatique et aucun nouvel
entraînement neuronal. Chemin normal uniquement ; aucune conclusion sur le
chemin compressé. Aucun résultat externe n'entre dans cette décision.
