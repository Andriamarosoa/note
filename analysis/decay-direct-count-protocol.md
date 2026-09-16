# Décroissance dans un compteur direct, sans correcteur ajouté

Protocole du 16 septembre 2026, figé avant les nouveaux scores.

Le diagnostic précédent ajoutait un correcteur à V27.3. Ce nouvel essai
évalue une autre formulation : **caractéristiques audio → probabilités de
K=0..6 → argmax**. Les prédictions V27.3 et les probabilités V27.2 ne sont
jamais des entrées du modèle. Il n'existe ni action conserver/−1/+1, ni
correction après la sortie. V27.3 reste la référence de comparaison.

Deux compteurs directs utilisent une régression logistique multinomiale L2 :
`C=1`, `lbfgs`, tolérance `1e-6`, maximum 10 000 itérations, seed 27341,
sans pondération ni recherche d'hyperparamètres. Chaque normalisation est
ajustée sur les compositions d'apprentissage uniquement. La cible est le
vrai K, y compris lorsque la référence s'en éloignait de plus d'une note.
L'argmax choisit le plus petit K en cas d'égalité exacte. Les classes absentes
de l'apprentissage reçoivent une probabilité nulle.

Le témoin reçoit 480 résumés acoustiques et 160 zéros. La variante reçoit les
mêmes 480 résumés et 160 indices de décroissance. La configuration et les
partitions sont identiques ; l'ajout de colonnes actives augmente les degrés
de liberté effectifs de la variante.

Les caractéristiques sont reprises sans modification de l'artefact
`10434200550`, run `35065190353`, commit
`6823a421017671d50a5498fd55f9424533e8a336`, ZIP SHA-256
`444bfb51d140c69c9dad3937f130a6ca0916dfe0ae0cde2ef478e64b3244e8ee`.
Les empreintes des deux NPZ et les identités des lignes sont vérifiées.

Il s'agit des mêmes 14 001 lignes du fold canonique 1, dont 1 776
polyphoniques, et des mêmes quatre compositions. Chaque modèle apprend sur
trois compositions et prédit la quatrième ; toutes les pistes comp/solo et
tous les joueurs d'une composition restent ensemble. V27.3 n'avait pas
appris ce fold, mais les données et résultats de développement ont déjà été
étudiés. Cet essai découle de cet audit : **aucune indépendance statistique
nouvelle n'est revendiquée**.

Les critères du diagnostic précédent restent inchangés : au moins +2 points
d'Exact-K polyphonique face au témoin et à V27.3, aucun correct global perdu,
aucune fausse polyphonie supplémentaire, borne bootstrap descriptive 95 %
positive face aux deux, et gain face au témoin sur au moins trois compositions.
Le bootstrap rééchantillonne les quatre compositions 10 000 fois, seed 27341.
Un succès justifierait seulement une réplication séparée. V27.3 n'est jamais
remplacée automatiquement ; aucun entraînement suivant n'est déclenché.

La formulation directe isole l'effet du mode de prédiction. Elle conserve
les limites acoustiques identifiées : fenêtre longue dans les graves,
quantification float16 du cache, décroissance d'un mélange spectral et
seulement 40 ms après le cluster. Elle conserve aussi le déséquilibre des
cibles. Un résultat négatif concerne ce compteur léger et ces caractéristiques.
Ce n'est pas un réentraînement complet de l'architecture historique V27.3.

Les coefficients, partitions et prédictions sont enregistrés et rejoués
dans le job pour vérifier les calculs. Le workflow publie uniquement les
synthèses agrégées JSON/Markdown, sans lignes individuelles ni coefficients.
Le premier lancement est déclenché par la publication de ce protocole.
À la publication du résultat, le workflow passera en réexécution manuelle.
