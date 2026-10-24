---
serie: S851
statut: termine
type: experience
reference: S708_historique
branche: codex/note-s851-identical-guitar-standard-ab256
run_id: 40098180183
commit_sha_execution: a80f2a0b6879d02e517778d84a6d9eaddf808c50
promotion: false
---

# S851 — Résumé de boucle : quatre IA contre la même transposition standard

## Objectif et réalisation
Comparer véritablement deux modèles S850 (BCE et RANK), chacun avec GRU continue et reset32, à la même prise de guitare réelle et à la même transposition standard +12 de S848, sans entraîner sur le WAV de contrôle et sans altérer les têtes officielles NOTE.

**Run achevé** : [#40098180183](https://github.com/Andriamarosoa/note/actions/runs/40098180183), statut d'exécution SUCCESS. Quatre fichiers audio produits ; artefact S851_REAL_GUITAR_STANDARD_BCE_RANK_PCM256 (ID 12562115617). Résultats complets dans resultat.md et resultat.json de l'artefact.

## Conclusion mesurée
- Ancienne IA S847 : **0,1164** de corrélation enveloppe vs standard, une seule note MIDI73 sur les 603 blocs.
- BCE continue (modèle choisi sur CAL, sans choix sur le WAV test) : **0,2567**, trois notes distinctes ; **rejetée**.
- BCE reset32 (audit uniquement) : **0,3825**, quatre notes distinctes ; **rejetée**.
- RANK continue : **-0,0867**, MIDI68 sur 601/603 ; **rejetée**.
- RANK reset32 : **0,1190**, MIDI68 sur 573/603 ; **rejetée**.
- Le témoin standard comparé à l'entrée réelle conserve l'enveloppe à **0,9038** (comparaison différente). Aucune qualité de note ne se déduit de cette valeur.

## Audit des limites
Le progrès de diversité de BCE ne suffit pas : S850 BCE top1 +12 contient une note GT sur seulement **8,4935 %** des décisions OUTER indépendantes, et E2→E3 n'atteint que **26,8071 % F1**. Le rejeu de clips physiques GuitarSet n'est pas une transposition fidèle de la prise : attaque, timbre, vibrato et phase ne sont pas conservés. Ne pas confondre signal audible et transcription juste.

## Invariants
Pas de lookahead, 256 nouveaux PCM à 44,1 kHz par hop ; mémoire causale autorisée, pas de délai artificiel. Le standard DSP n'est que le témoin offline. Six axes NOTE/MIDI inchangés ; référence Onset 80 WAV **69,2584331 % F1**. Aucun modèle promu ; latence p99 Windows/iRig non mesurée.

## Suite justifiée par les preuves
Auditer les erreurs de hauteur GT par note et par fragment, puis entraîner une voie neuronale transformant **la performance originale** plutôt qu'une banque de clips, avec A/B contre le standard et séparation TRAIN/CAL/OUTER. Ne promouvoir qu'après démonstration musicale et métriques objectives, pas après une CI verte.
