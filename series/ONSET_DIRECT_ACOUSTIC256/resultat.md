# Résultat — apprentissage acoustique direct PCM256

**Entraînement en cours ; aucun gain revendiqué avant mesure.**

Les 80 WAV originaux ont été parcourus, soit 306536 décisions PCM256.
Les nouvelles caractéristiques 195D sont calculées sur chaque bloc courant.
88 tests passent, y compris les contrôles du sélecteur immédiat, des réattaques,
du score apparié un-à-un, de l'équivalence cache/flux et des entrées acoustiques.

La référence vérifiée reste **68,3709869203 % F1**, TP8050 / FP4231 / FN3217.
Le score des nouvelles prédictions sera distingué de sa confirmation par un
replay audio complet. Aucun modèle n'est promu à ce stade.
