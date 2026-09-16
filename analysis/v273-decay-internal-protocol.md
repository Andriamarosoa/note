# V27.3 — diagnostic interne apparié de décroissance exponentielle

Protocole figé le 16 septembre 2026, avant tout résultat de ce diagnostic.

## Hypothèse et portée

Une attaque faible peut apporter de l'énergie sans produire une hausse entre
deux trames : le son précédent diminue plus vite que la nouvelle contribution.
La différence positive entre l'amplitude observée et une décroissance prévue
pourrait donc fournir une information utile au comptage des nouvelles notes.
Le logarithme sert à ajuster cette décroissance, pas à calculer directement K.

Le test compare **V27.3 gelée**, **V27.3 + correcteur acoustique témoin**, et
**V27.3 + le même correcteur avec des indices de décroissance**. Il ne modifie
pas l'entraînement V28.0-I, la référence officielle, les règles de ranking ou
les fichiers scientifiques des expériences précédentes.

Les caches et prédictions intermédiaires nécessaires à une reconstruction
exacte du pipeline interne V27.3 ne sont plus disponibles : les artefacts
V10.4 ont expiré le 9 septembre. Le premier diagnostic utilise donc les
prédictions V27.3 archivées du fold canonique 1. Les modèles et calibrages
qui les ont produites n'ont pas appris ce fold ; ils ont utilisé les folds
0, 2, 3, 4. **Il serait incorrect de dire que le fold 0 n'a jamais participé
à l'apprentissage de cette référence.**

Ces prédictions et résultats avaient déjà été étudiés. C'est une réanalyse
de développement, pas une nouvelle validation externe, ni un remplacement
du score agrégé V27.3 de 42,6019 %. Aucun checkpoint V28 n'est utilisé.

## Sources figées

- V27.3 : run `34297767492`, commit `dd43b4f92b234dc7c84377e18389fd34350cd1b4`,
  artefact `v273-selective-transition-fold-1` / `10086186883` ; ZIP SHA-256
  `7fe5060ca6b3e901c41855f19209dadd52710cf2ae9d4acc50278a69ad600771`.
- Cache acoustique causal : artefact `v280-i-prepared` / `10432567858`,
  run `35061192743`, commit `a9630c929b40cd4ba8ce3853ad620576211d18f5` ;
  ZIP SHA-256 `2931345bf7570c2d9501407728e1ce3830f2cfc39f2ef2bf10f24627b4c154a4`.
  La préparation `104681672224` a réussi ; le reste de V28-I peut continuer.
  Ce cache est une copie vérifiée du cache F, conservée 90 jours.
- Manifeste scientifique F : SHA-256
  `e43567e3bd0945911915b4b11d88f960e4d95c0519809045dc9c7b056f0e5226`.

Les empreintes brutes NPZ/JSON de référence et celles du cache sont contrôlées
avant tout apprentissage. L'identité des lignes, des pistes, des folds et des
cibles doit être strictement égale. Les autres folds ne sont pas indexés dans
le tableau acoustique pour ce diagnostic. La vérification d'intégrité lit les
fichiers complets ; elle ne transforme pas leurs autres lignes en exemples.

## Partitions et comparaison

Le fold 1 comprend 14 001 lignes, 1 776 polyphoniques, 40 pistes et seulement
**quatre compositions** : `BN3-154-E`, `Funk2-119-G`, `Funk3-112-C#`, `SS3-98-C`.
Le groupement retire le joueur et les suffixes comp/solo : les variantes d'une
même composition restent ensemble.

Chaque composition est exclue à son tour. Les deux correcteurs apprennent sur
les trois autres compositions et prédisent uniquement la composition exclue.
Chaque ligne reçoit donc une prédiction de correcteur hors apprentissage.
Les statistiques de normalisation sont calculées sur les trois compositions
d'apprentissage seulement. Aucun seuil ni hyperparamètre n'est choisi à partir
de la composition exclue. Aucun résultat historique/Locked12 n'est ouvert.

Le contrôle et la variante utilisent la même régression logistique multinomiale
L2, `C=1`, solveur `lbfgs`, maximum 10 000 itérations, tolérance `1e-6`, seed
27341, sans rééquilibrage. Un échec de convergence est un échec technique,
pas un verdict scientifique. Il n'y a aucun balayage de réglages.

Les actions sont conserver K, K-1, K+1. La cible d'apprentissage est la correction
exacte lorsque l'écart vaut un ; dans les autres cas, conserver K. Les actions
hors 0..6 sont interdites. On choisit la probabilité d'action maximale ; en cas
d'égalité, conserver K. Les erreurs plus éloignées ne peuvent donc pas être
corrigées entièrement : c'est une limite volontaire de ce premier test.

Les entrées communes sont le compte V27.3 encodé en sept colonnes, les cinq
probabilités archivées de son spécialiste V27.2, et six résumés spectraux :
moyenne et dernière amplitude logarithmique pré-cluster, moyenne et maximum
récents, moyenne de la hausse au pré-cluster et maximum du flux positif.
Chaque résumé conserve 80 groupes fréquentiels, par moyenne de trois bins
CQT adjacents (le dernier groupe est partiel). Il y a 492 entrées communes.

La variante ajoute 160 entrées de décroissance. Le contrôle reçoit 160 zéros
aux mêmes positions, soit 652 colonnes et la même configuration de modèle.
Ce contrôle teste l'utilité de la transformation explicite pour un correcteur
léger ; il ne prouve pas que l'information est absente du signal d'origine.

## Extraction causale de la décroissance

Le crop existant comporte 24 trames, espacées de 256/44100 secondes, dont la
dernière ne dépasse jamais `cluster_start + 1764` échantillons (40 ms).
Les 17 premières trames servent à ajuster la décroissance : leur dernier
endpoint est au moins 7×256 échantillons avant la fin du crop, donc strictement
antérieur au cluster. Cela reste vrai lorsque la piste se termine tôt.

On reconstruit l'amplitude positive à partir de `log1p(100*A)` et on ajuste
`ln(A) = intercept + pente*t`, indépendamment par bin. Le plancher est le
maximum de `1e-6` et de 2 % du maximum pré-cluster du bin. Les observations
sous le plancher sont ignorées ; il faut au moins huit trames valides, une
pente comprise entre -60 et -0,25 par seconde, et une RMSE logarithmique
au plus égale à 0,30. Ces constantes techniques sont fixées avant mesure.

La prédiction est extrapolée vers les sept dernières trames, déjà disponibles
dans le crop causal. L'indice est
`max(log1p(100*A_observée) - log1p(100*A_prévue), 0)`.
Les bins dont l'ajustement n'est pas fiable donnent zéro. On fournit au
correcteur la moyenne et le maximum temporels, regroupés par trois bins.
La prédiction de décroissance n'utilise jamais les sept trames qu'elle évalue.

Limites : historique d'environ 93 ms entre les endpoints des trames ajustées,
magnitudes quantifiées float16 et fenêtres CQT longues dans les graves,
décroissance exponentielle locale approximative, bruit, interférences et
harmoniques. Une bande contenant plusieurs sons ne suit pas nécessairement une
exponentielle. Aucun logarithme du mélange ne sépare automatiquement les notes.

## Critères préenregistrés et livrables

Les trois variantes sont comparées sur les mêmes 14 001 lignes. On rapporte
Exact-K polyphonique/global, matrices de confusion, sous/sur-comptage, fausses
polyphonies, corrections/régressions et notamment les vrais K=2 prédits K=1.

Le diagnostic justifie une réplication séparée seulement si la variante :

1. gagne au moins **2 points polyphoniques** face à V27.3 et face au contrôle ;
2. ne perd aucun correct global et n'ajoute aucune fausse polyphonie face aux deux ;
3. présente une borne bootstrap 95 % strictement positive face aux deux ;
4. gagne en polyphonie face au contrôle dans au moins trois des quatre compositions.

Le bootstrap est apparié par **composition**, 10 000 réplications, seed 27341.
Avec quatre compositions et des ensembles d'apprentissage qui se recouvrent,
ses intervalles sont **descriptifs** ; ils ne constituent pas une confirmation
indépendante ni une estimation complète de l'incertitude d'entraînement.
Les 2 points sont le seuil de ce diagnostic léger, pas le critère de promotion
de V28-I ou de remplacement de V27.3.

En cas d'échec, conclusion : pas de gain matériel avec cette formulation et
ce correcteur, V27.3 conservée. Cela ne réfute pas toutes les modélisations de
décroissance possibles. Aucune autre variante n'est lancée automatiquement.

Les artefacts à 90 jours conservent les résumés acoustiques, normalisations,
coefficients, partitions, probabilités, prédictions appariées, empreintes,
versions des dépendances, rapport JSON et synthèse Markdown. Les tests couvrent
une décroissance pure, une attaque masquée sans hausse totale, l'absence de
futur, le silence, les partitions et la comparaison appariée complète.

## Correction technique avant tout résultat réel

Le run `35064823827` (commit `570de36db1ed5592b6a8c822091b35e1e252e5af`)
a passé les 24 tests et vérifié les données, puis son tout premier ajustement
témoin a atteint le plafond initial de 1 000 itérations. Le programme a échoué
sur `ConvergenceWarning` dans `model.fit`, avant `predict_proba` et avant toute
mesure de performance sur les compositions exclues. Aucun résultat de ce run
n'est admissible comme verdict.

Le plafond est porté à 10 000 pour les deux variantes. L'objectif L2 (`C=1`),
le solveur, la tolérance, les caractéristiques, les partitions, les actions et
tous les critères de décision restent identiques. C'est une correction du
budget numérique de convergence, pas une sélection à partir des scores.
