---
serie: S843
statut: en_cours
type: experience
reference: S708
branche: codex/note-s843-real-audio-transpose-reconstruction256
run_id: 40086388674
commit_sha: 52e06ebbcdd37f38721c974628af6d5339e33c53
promotion: false
---

# S843 — Reconstruction IA de vrais sons de corde + transposition causale PCM256

## Exécution
Nouvelle tête neuronale `real_audio_transpose_head256.py` et entraîneur indépendant `s843_train_real_hex_transpose256.py`. Elle consomme un unique nouveau bloc micro 256 à 44 100 Hz à la fois, conserve la seule mémoire GRU du **passé**, et retourne six formes d'onde **estimées** de cordes acoustiques, six scores de présence de corde et 49 hypothèses de notes avec ablation de transposition ±12.

Les six sorties ne sont ni des oscillateurs synthétiques ni des extraits stockés. Elles sont des valeurs calculées par le réseau qui doit être entraîné contre de **vrais enregistrements hexaphoniques d'origine**. Avant un apprentissage validé, ces sorties sont sans valeur musicale. Le son de micro et le son hex proviennent de capteurs différents ; aucune garantie d'identité acoustique exacte, de séparation parfaite ou d'alignement de phase parfait.

La vérification technique CI et l'entraînement réel doivent être rapportés distinctement, sans scores inventés.

### Actions réelles du 24 octobre 2026

- [Contrôle logiciel S843 #40086388674](https://github.com/Andriamarosoa/note/actions/runs/40086388674) : **SUCCESS**, test de blocs 256, causality/état GRU, gradients, vraie provenance exigée, ablations et absence de MIDI.
- [Entraînement hex réel #40086388680](https://github.com/Andriamarosoa/note/actions/runs/40086388680) : **en cours à la dernière vérification** ; téléchargement des trois archives GuitarSet originales, total de plusieurs Go. Aucune archive hex n'a encore été validée par ce run au moment de la rédaction, et aucun nouveau poids entraîné sur de vraies cordes n'est encore confirmé.
- Ces tests unitaires utilisent des tenseurs de contrôle **artificiels uniquement pour éprouver le logiciel** ; ils ne sont PAS des exemples musicaux d'entraînement, ni un résultat scientifique sur guitare.


## Métriques
| Axe | Résultat actuellement prouvé | Statut |
|---|---:|---|
| Référence de travail Onset PCM256 | **69,2584331 % F1**, 7607 TP / 3093 FP / 3660 FN | Intact, ancien run 39957354552 |
| S842 transposition latente E2 SS1 | **0 %** vs témoin 12,22 % F1 de présence | Rejeté |
| S842 transposition latente E3 SS1 | **66,85 %** vs 77,86 % F1 de présence | Rejeté |
| S842 superposition E2+E3 SS1 | **0 %** vs 13,48 % F1 de présence | Rejeté |
| Nouvelle reconstruction S843 | **NON MESURÉ** sur vraies cordes | Prototype code |
| Nouvelle transposition S843 | **NON MESURÉ** | Validation à faire |
| Exact-K | Intact (aucune mesure nouvelle) | Tête indépendante |
| Active Exact-K | Intact (aucune mesure nouvelle) | Tête indépendante |
| Offset | Intact (aucune mesure nouvelle) | Tête indépendante |
| Frame Pitch F1 | Intact (aucune mesure nouvelle) | Tête indépendante |
| Full Note F1 | Intact (aucune mesure nouvelle) | Tête indépendante |

## Comparaison à S708
S708/H0 historique donne 73,252019 % de F1 mais avec modèle entraîné à 1024. La tête S843 ne charge aucun de ses poids. Elle n'utilise ni DSP de transposition ni anciennes fenêtres FFT de 8192 échantillons, mais peut mémoriser les blocs passés via GRU sans retarder la décision. Aucun F1 S843 ne peut être comparé à Onset 69,2584 % avant création de vrais NoteOn et nouveau replay intégral de tous les WAV.

## Audits
| Rôle | Statut | Interprétation |
|---|---|---|
| Données | UNKNOWN | trois archives officielles GuitarSet et mapping hex/JAMS à vérifier |
| Acoustique | PASS tests | 256 nouveaux PCM, reconstruction par réseau contre vrais signaux hex |
| Architecture | PASS tests | CNN+GRU causal, transporteur +12/−12 et sans-vues, six autres têtes inchangées |
| Apprentissage | UNKNOWN | aucune prétention de vraies poids tant que le run réel n'a pas réussi |
| Métriques | UNKNOWN | pas d'évaluation de vraies attaques Onset ni de vraies séparations hex |
| LIVE | UNKNOWN | aucun timing CPU Windows/iRig validé, aucun MIDI |

## Verdict
**NON PROMU** : code de tête et apprentissage supervisé de sons réels, à valider en CI et contre les archives d'origine. Les reconstructions et transpositions audio sont des **estimations neuronales**, pas la garantie de restituer un son réellement présent. La tête ne donne pas directement de NoteOn et n'a aucun effet sur la référence Onset.

## Prochaine action
Faire réussir les tests de conformité et l'apprentissage hex à partir de véritables prises GuitarSet. Auditer de manière systématique la correspondance des six cordes et le décalage physique entre micro/hex, puis comparer sans-vues contre ±12 sur E2, E3 et leurs recouvrements. Uniquement après de vrais gains au niveau des notes, créer une tête Onset qui exploite ces preuves, garder les six autres têtes gelées et évaluer les 80 WAV sans retard.
