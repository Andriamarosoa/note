# Règle : auditer les effets de chaque solution Onset PCM256

L'audit ne doit PAS expliquer un écart de reproduction, sauf si c'est sa question. Il doit expliquer pourquoi une solution nouvellement testée donne une faible progression ou fait régresser le F1.

## Protocole obligatoire, que la solution gagne ou perde

1. Comparer les mêmes 80 pistes, mêmes annotations, même OUTER et même résolution PCM256, avec TP, FP, FN, F1, précision et rappel.
2. Associer individuellement les annotations et séparer TP conservés, récupérés, détruits et FN persistants. Vérifier les comptes.
3. Analyser les faux positifs ajoutés et retirés, les doublons, les réattaques, et la compétition d'appariement hongrois à 80 ms.
4. Pour chaque proposition HGB, inspecter le score acoustique, la probabilité d'arbitrage, le seuil de CAL figé, et les raisons de rejet (seuil, réfractaire, doublon).
5. Ventiler composition, texture solo/accompagnement, registre et polyphonie K, sans les confondre.
6. Comparer le gain sur CAL au gain sur OUTER et chercher pourquoi une sélection réputée bonne généralise mal.
7. Examiner les indices spectro-harmoniques, la fondamentale et la réattaque dans les exemples gagnés et perdus ; ne pas appeler ces associations des causes physiques prouvées.
8. Conserver chaque nouvelle famille de régression comme preuve ; une correction qui sauve un événement mais régresse ailleurs ne prouve pas encore la bonne cause.
9. Préenregistrer la nouvelle hypothèse puis entraîner sur TRAIN, calibrer sur CAL, tester sur OUTER sans sélectionner la correction après ouverture des étiquettes OUTER.
10. Ne pas ajouter d'attente, de futur audio, ou modifier les autres têtes et le MIDI officiel.

## Code

Le post-mortem lecture seule se trouve dans scripts/audit_onset_solution_effect256.py.
Il sort report.json, per_track.csv, hgb_candidates.csv et ground_truth_changes.csv.
Il exige le replay audio complet et sa parité, avant d'accepter des conclusions.

Commande quand les artefacts HGB sont disponibles :

    python -m scripts.audit_onset_solution_effect256 --experiment "$ONSET_EXPERIMENT_DIR" --gate "$GATE_DIR" --out "$EFFECT_AUDIT_DIR"

Cette branche d'audit est volontairement distincte de celle du run audio actuellement actif, afin de ne pas interrompre son entraînement.
