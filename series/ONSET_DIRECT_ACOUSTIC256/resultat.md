# Résultat reproduit — acoustique directe Onset PCM256

Le replay des 80 WAV signés et des 306536 decisions PCM256 confirme **TP 7585 / FP 3068 / FN 3682 / F1 69,206204 %**. La référence HGB est reproduite à TP 8050 / FP 4231 / FN 3217 / F1 68,370987 %.

Gain de F1 : +0,835217 point, avec -1163 FP et +465 FN nets. L'audit paire à paire identifie 825 anciens TP perdus et 360 nouvelles récupérations ; 802 pertes sont en accompagnement. Les 821 pertes de Rock3 et SS1 se concentrent dans les deux politiques qui désactivent la référence HGB.

Les 80 replays vérifient chaque bloc : mêmes caractéristiques, mêmes événements de référence et mêmes événements candidats que les predictions scellées, sans futur ni attente ajoutee. 96 tests passent. L'audio complet à été exécuté localement ; la CI de tests est distincte.

Voir [le rapport de l'audit](../ONSET_POLYPHONY_LOSS_AUDIT256/résultat.md) et l'archive experimentale fournie avec cette serie. Aucune promotion dans le MIDI officiel ni validation externe indépendante.
