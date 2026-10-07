# Périmètre GuitarSet courant — players 00 à 04 uniquement

Décision confirmée le 7 octobre 2026.

## Périmètre autorisé

Le pipeline V27.3 courant conserve exclusivement les joueurs GuitarSet :

- 00
- 01
- 02
- 03
- 04

Le joueur `05` reste explicitement exclu.

Cette règle est déjà imposée dans `src/causal_note/guitarset.py` par
`ALLOWED_PLAYERS = {"00","01","02","03","04"}` et couverte par les tests
`test/test_guitarset.py`, qui vérifient que player 05 est ignoré par l'index,
rejeté par `GuitarSetTrack` et refusé par `load_boundary_slots`.

## Statut des expériences player 05

Les runs historiques concernant player 05 restent conservés uniquement pour
traçabilité. Ils sont **hors périmètre scientifique courant** et ne doivent pas :

- modifier un seuil ;
- sélectionner une famille de features ;
- servir de validation ;
- entrer dans les métriques de référence ;
- être utilisés pour promouvoir ou rejeter une variante dans le pipeline 00–04.

En particulier, les runs `37552019921` et `37568502612` sont archivés mais
non autoritatifs pour la poursuite actuelle.

## Référence méthodologique courante

La dernière évaluation valide dans le périmètre 00–04 est l'audit imbriqué
`v273-feature-family-nested`, vérifié par CI `37570591433`.

Il corrige le défaut de sélection du run exploratoire `37547761255` :
le gain descriptif `+20` de `base_geom_attack@0.59` avait été choisi sur les
VAL réunis. Une fois famille et seuil choisis exclusivement sur FIT, le bilan
VAL sélectionné est net `0`. La décision actuelle est donc de **conserver la
référence**.

Toute nouvelle étude doit rester sur folds 0/1/2/4, players 00–04, sans fold 3
et sans player 05, sauf décision explicite contraire ultérieure.
