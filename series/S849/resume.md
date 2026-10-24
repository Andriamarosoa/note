---
serie: S849
statut: en_cours
type: audit
reference: S708
branche: codex/note-s849-transpose-constant-note-audit256
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S849 — Boucle 1 : audit du neurone de sélection de note figé

## Objectif
Diagnostiquer la note bloquée MIDI73 de S848 (603/603 décisions +12) avant de lancer une nouvelle architecture. Mesurer les scores du réseau S847 réel avec état GRU continu, réinitialisé toutes les 8 décisions et toutes les 32 décisions. Aucune politique arbitraire de diversification, aucune modification de hauteur DSP dans l'inférence.

## Hypothèse
S847 a été entraîné sur huit blocs (8×256=2048 échantillons au total dans un entraînement causal, *un hop de 256 en entrée à chaque décision*) mais utilisé en Live pendant 603 blocs successifs. Le décalage d'horizon entre TRAIN et LIVE peut réduire sa sensibilité temporelle ; cependant la préférence pour MIDI73 peut provenir du biais de classe / perte BCE et persister même à horizon 8. Seul l'audit peut distinguer ces phénomènes.

## Modifications
- `scripts/s849_audit_neural_note_saturation256.py` : réutilise le même checkpoint S847, la banque authentique des 39 notes et le même WAV de guitare S848. Audite la classe choisie, les probabilités par classe et leurs écart-types, avec resets GRU {0,8,32}. Aucun reset ne veut dire « attendre » ; il s'agit d'un diagnostic sans lookahead.
- `.github/workflows/note-s849-neural-saturation256.yml` : workflow autonome, SHA du checkpoint conservé, deux artefacts GitHub antérieurs comme références vérifiables ; sauvegarde un JSON mesuré.
- Aucune modification des sources S847, de S848, des six têtes NOTE ou du MIDI.

## Protocole
Première boucle de recherche : si les 603 decisions restent MIDI73 en reset8, abandonner toute politique de seuils et de normalisation artificielle et corriger le réseau / données (durée de training causale, perte séparant vrai pitch des harmoniques, distribution multi-note). Tout score de présence sur les 80 WAV Onset nécessite de vraies sorties NoteOn et reste distinct du proxy S847.

## Audits obligatoires
| Axe | Attendu |
|---|---|
| Données | modèle et vrai son S848 identiques |
| Architecture | sortie 256 samples et GRU passée, sans futur |
| Apprentissage | aucun checkpoint promu ni poids modifiés |
| Métriques | top1 et variance des probabilités sur 603 hops, aucune F1 inventée |
| LIVE | vrai p99 Windows/iRig non testé |

## Statut
Audit en cours. Référence Onset expérimental PCM256 **69,2584331 % F1** (7607 TP, 3093 FP, 3660 FN), inchangée. Standard offline +12 S848 corrélation enveloppe **90,4 %**, IA S847 **11,6 %**, et MIDI73 figé : échec.
