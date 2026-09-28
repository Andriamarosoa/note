# Test causal du budget : huit époques sont-elles limitantes ?

Comparer les checkpoints internes à **8, 12 et 16 époques cumulées** pour
les deux objectifs déjà testés, uniforme et pondéré. Comparaison principale :
16 contre 8 ; 12 est un point intermédiaire déclaré avant les résultats.

Les données et partitions sont celles de l'expérience `36351028493` :
43 357 groupes d'apprentissage interne et 15 952 de validation interne.
**Aucune ligne du fold externe 3 ne passe dans le réseau lors de ce test.**
L'architecture, les 31 trames, les poids de classes, Adam à 0,0002 et les lots
de 128 restent ceux des modèles natifs. Aucun correcteur de sortie.

Reprendre les checkpoints internes de huit époques. Exiger la restauration
exacte des variables d'Adam et de son compteur de 2 712 mises à jour ; arrêter
si le checkpoint ne le permet pas. Rejouer les 15 952 décisions internes et
vérifier l'identité des poids avant/après ce contrôle. Les prédictions
d'apprentissage du point 8 proviennent de l'audit figé `36375624830`, dont
l'identité du checkpoint est vérifiée.

L'ordre des lots poursuit les indices d'époque 8 à 15 du protocole initial.
Le flux aléatoire du dropout reprend avec la graine initiale +50 000, identique
entre objectifs. **Il s'agit d'une prolongation auditée, pas d'une affirmation
d'identité bit à bit avec un entraînement de 16 époques sans interruption.**
La remise à zéro d'Adam est interdite.

Mesurer à chaque point l'Exact K global et polyphonique, les comptes exacts,
sous/surcomptages et pertes par K, sur les exemples vus et la validation.
Conserver les probabilités et poids de 8, 12 et 16. Publier l'état intermédiaire
à 12 pendant que la suite s'exécute. Le rapport final recalcule les transitions
de décisions et les gains par composition.

Interprétation fixée avant résultats : une hausse sur apprentissage et
validation démontre que des mises à jour supplémentaires améliorent le compte
dans cette expérience. Une hausse sur les seuls exemples vus ne valide pas
la prolongation comme solution à la faible performance de validation. Une
absence de progrès ne prouve pas que toute durée supplémentaire serait inutile.
Rapporter les régressions par K, notamment K=1, 2 et 3, même si le score agrégé
progresse. Aucune sélection de checkpoint ni promotion officielle automatique.

Les résultats portent sur une graine de reprise et une partition interne.
Les checkpoints finaux évalués précédemment sur le fold 3 restent inchangés.
