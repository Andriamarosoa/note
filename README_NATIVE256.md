# Onset — modèle à poids réellement entraînés à 256

Cette branche est indépendante des cinq autres caractéristiques et n'écrit pas le MIDI officiel.

**Poids :** S613 natif entraîné à 256 (SHA256 `7ccf046e3a1870eaca3a108e6f9565298aeeadf43732a2f83559542c03f5f17f`), et S701 spécifique Onset entraîné à 256 (SHA256 `cc72cf81f477c6b734e253df5fb04d1789ab707dc160f06bdc5cf9ffdff2fb0f`). S701 est un HistGradientBoosting de 39 variables, **pas** un réseau neuronal. Le checkpoint S701 est récupéré du [run #39085408073](https://github.com/Andriamarosoa/note/actions/runs/39085408073) et vérifié avant tout chargement.

L'architecture Onset exécute à chaque hop 256 (5,805 ms) une nouvelle analyse PCM, un modèle S613 et la tête S701, avec deux sélecteurs privés isolés. Elle **ne charge pas** le réseau temporel S708 historiquement entraîné à 1024.

L'expérience conserve le mode `native` sélectionné précédemment par la calibration S701. Le `head_0.75_floor_0.20` est testé comme ombre sans promotion ni sélection après lecture des résultats. L'évaluation SIGNÉE est fondée sur les 80 pistes GuitarSet (306536 frontières 256), tolérance d'attaque 80 ms, avec seuil Onset historique S708/H0 **73,25201855% F1**. Aucun score ne sera inventé avant exécution complète; les résultats FIT80 déjà explorés ne sont pas une preuve de généralisation externe. Le coût CPU et la latence physique du matériel restent inconnus.

Code : `src/onset256/`, `scripts/replay_native256.py`, `.github/workflows/onset-s701-weights-256.yml`. Aucune autre caractéristique modifiée.
