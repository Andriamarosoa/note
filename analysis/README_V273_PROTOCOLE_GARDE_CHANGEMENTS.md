# V27.3 — Quatrième série : risque conditionnel au changement effectivement proposé

Déclaré après les trois premières séries. Le mélange non linéaire léger des
audits de groupe apporte 569 corrections / 463 régressions (+106) en
conservant 523 des 566 anciennes corrections. Le routage plus agressif reste
à 506 /354 (+152). La majorité des événements ne sont pas changés, alors que
les pertes précédentes étaient ajustées sur tous les événements ; tester
directement la population des changements constitue une intervention sur
l'objectif d'apprentissage, sans interprétation physique automatique.

Garder les actions originales du croisement à +48. Un garde peut seulement
les autoriser ou revenir à KEEP. Ainsi aucune nouvelle régression ni
nouvelle correction n'est possible face à ce parent ; le coût réel est
entièrement visible dans les corrections abandonnées. Une combinaison
complète reste une action autorisable même si ses constituants sont mauvais.

Pour chaque morceau test, entraîner uniquement sur les changements proposés
par ce réseau figé dans les autres morceaux du même fold exclu. Cibles :
correction, régression, changement neutre. Deux contextes : probabilités
initiales + K initial/proposé + audits de tous les groupes donnant ce K ;
puis ajouter les 58 observables acoustiques. Arbres de gradient fixés :
80 itérations, learning_rate 0,05, 7 feuilles, profondeur 3, feuille minimale
20, L2=10, seed=27402, sans arrêt anticipé.

Mélanger les trois probabilités d'issue avec leur estimation originale aux
poids 0,25 /0,50 /0,75 /1. Les coûts restent 1 /1,15 /1,3 /1,5 /2 /3 /5.
Conserver les 56 politiques ainsi que deux choix imbriqués : global par
morceau et par K initial, sur ces 56 politiques + les sept coûts de la
distribution d'origine. La contrainte interne conserve au moins 90 % des
anciennes corrections et un net >=original. Aucun label du morceau test
dans l'entraînement ou le choix ; mêmes limites de supervision que précédemment.

Ces probabilités décrivent l'issue d'une action fixée, pas une distribution
complète sur tous les K. Publier la calibration des trois issues uniquement
sur les changements proposés, et toutes les pertes de corrections. Conserver
modèles, données, prédictions et choix sans promotion automatique.
