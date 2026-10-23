# Résultat — arbitre HGB/acoustique appris PCM256

**Statut actuel : implémentation expérimentale ; aucun nouveau F1 mesuré et aucune promotion.**

| Version | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|
| HGB historique | 8050 | 4231 | 3217 | 68,370987 % |
| Acoustique directe 195D protégée | 7585 | 3068 | 3682 | **69,206204 %** |
| HGB réactivé partout, rejeté | 8376 | 4578 | 2891 | 69,163123 % |

Anciennes pertes : 825. Nouvelles récupérations de référence : 360. FN persistants : 2857. Les 724 cas sous seuil acoustique et les régressions Rock3/SS1 guident l'apprentissage.

**À remplir uniquement après le replay complet des 80 WAV originaux** : scores CAL, OUTER et audio frais, TP retrouvés, TP perdus, FP supplémentaires, ventilation Rock3/SS1 et risques résiduels.

F1 de la nouvelle tête : NON MESURÉ. Règles : zéro audio futur, zéro attente, 256 samples, autres axes et MIDI inchangés, aucune promotion automatique. Résultat FIT80 non indépendant.
