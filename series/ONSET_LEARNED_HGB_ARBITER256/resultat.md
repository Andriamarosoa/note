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

## Run #39950605428 — résultat nouveau et distinct (23 octobre 2026)

- 80 WAV signés et replay direct natif PCM256 terminés avec parité d'événements et de caractéristiques.
- Une **nouvelle** reproduction par réentraînement a obtenu TP 7607 / FP 3093 / FN 3660, soit **69,2584331 % F1**.
- Elle diffère de la référence archivée (TP 7585 / FP 3068 / FN 3682, 69,206204 %). La cause de l'écart de réentraînement n'est pas démontrée.
- Le garde-fou original a arrêté l'arbitre avant son entraînement. Ceci n'est pas une dégradation mesurée de l'arbitre.
- La suite traite cette variante réentraînée comme **distincte** ; les **825 anciennes pertes du modèle archivé ne lui sont pas attribuées sans nouvel appariement**. Aucune promotion.
- Provenance : https://github.com/Andriamarosoa/note/actions/runs/39950605428
