# Pourquoi l'Exact K reste faible : audit d'apprentissage

Le périmètre reste le fold externe 3. Les deux modèles natifs à 31 trames
(uniforme et pondéré), leurs checkpoints internes et finaux, leurs données
et leurs huit époques restent figés. Aucun nouvel entraînement ni correcteur.

Mesurer l'Exact K, les sous/surcomptages, les matrices de confusion et la
perte par vrai K sur les lignes d'apprentissage et sur la partition non vue
initialement prévue. Rejouer les décisions non vues et vérifier l'identité
des poids avant et après l'inférence. La perte d'inférence est distinguée
de la perte historique d'entraînement, calculée avec dropout et poids évolutifs.

Ces mesures doivent distinguer un comptage mal appris même sur les exemples
vus d'un écart de généralisation. Les classes fréquentes et rares sont
rapportées séparément ; une bonne précision globale dominée par K=0 n'est
pas une bonne performance polyphonique.

Diagnostiquer séparément les vrais K>=2 prédits 0/1, et les confusions entre
2, 3, 4, 5 et 6. Le classement conditionné par le vrai statut polyphonique
est un oracle : il utilise une vérité inaccessible en production. Il sert
uniquement à localiser une perte de score et ne sera pas publié comme un
modèle amélioré ni comme un correcteur.

Compléter avec les expériences contrôlées déjà terminées (pondération,
fenêtre) et les étapes du contrôle V27.3 complet, sans comparer directement
les scores du composant isolé à la référence officielle. Les observations
sur la concurrence entre groupes ne sont pas présentées comme une cause
universelle. Ni une origine harmonique ni un nombre d'époques insuffisant
ne seront déclarés démontrés sans expérience les isolant.

Le diagnostic doit préciser pour chaque explication : preuve disponible,
part du déficit concernée, limites et correction native à tester. Aucun
seuil, checkpoint ou architecture n'est sélectionné sur le fold externe.
