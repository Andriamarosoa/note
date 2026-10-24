---
serie: S843
statut: termine
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

### Résultat scientifique final du run S843 sur vrais enregistrements

[GitHub Actions #40086388680](https://github.com/Andriamarosoa/note/actions/runs/40086388680) est **SUCCESS** (terminé à 10:44 UTC le 24 octobre 2026). **Les trois archives originales microphone, hexaphoniques, JAMS ont été vérifiées** (micro SHA256 237cdc..., hex MD5 c31d..., annotation SHA256 8daa...). Le réseau a été entraîné 4 époques sur **10 prises TRAIN guitariste 04**, calibré sur **10 prises guitariste 05** et évalué sur **40 prises OUTER guitaristes 00–03**. Il a vu **640 paires d'octave de vraies notes indépendamment enregistrées**.

| Sortie S843 | Sans transpose | Avec IA transpose ±12 |
|---|---:|---:|
| Présence E2, GT 73 positifs sur 8960 blocs OUTER | **1,6163 % F1** (TP 73 / FP 8887 / FN 0) | **0 % F1** (TP 0 / FP 0 / FN 73) |
| Présence E3, GT 95 positifs | **2,0983 % F1** (TP 95 / FP 8865 / FN 0) | **0 % F1** (TP 0 / FP 0 / FN 95) |
| E2+E3 | **Non évaluable** | **Non évaluable** |

**Il y a 0 exemple GT positif E2+E3 dans les 8960 blocs sélectionnés pour OUTER.** Le script original a calculé numériquement F1=0 même sans positifs, mais ce n'est PAS une mesure valide de détection de l'accord. Il y avait 8960 FP sur ce pseudo-test pour le contrôle sans transpose et 0 FP pour l'IA transposée.

L'erreur de reconstruction hex estimée est **MAE=0,028967995** dans le domaine capteur hex ; sans comparaison de cette erreur à une estimation silence, rien ne prouve que les six sorties reconstruisent mieux le son réel. Le seul calcul CPU local Linux est sur **8 pas** (p99=2,3893362 ms) : il ne valide ni p99 représentatif ni latence complète Windows/iRig.

**Verdict : apprentissage réel effectué, architecture S843 non performante sur E2/E3 et non promue.** Aucun nouveau F1 Onset événement n'a été mesuré. Les poids expérimentaux sont dans l'artefact GitHub `S843_REAL_GUITARSET_SIX_STRING_NEURAL_RECONSTRUCTION_PCM256` (#12557477491), mais le MIDI officiel et les six têtes existantes restent intacts. La suite S844 augmente la représentativité du TRAIN, pondère les vraies notes graves et distingue F1 non évaluable avant de réapprendre sur les mêmes archives signées.

### Actions réelles du 24 octobre 2026

- [Contrôle logiciel S843 #40086388674](https://github.com/Andriamarosoa/note/actions/runs/40086388674) : **SUCCESS**, test de blocs 256, causality/état GRU, gradients, vraie provenance exigée, ablations et absence de MIDI.
- [Entraînement hex réel #40086388680](https://github.com/Andriamarosoa/note/actions/runs/40086388680) : **en cours à la dernière vérification** ; téléchargement des trois archives GuitarSet originales, total de plusieurs Go. Les trois archives ont été validées par leurs empreintes et un checkpoint de recherche a réellement été entraîné ; ses scores musicaux sont insuffisants.
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
| Données | PASS source / mapping UNKNOWN | les trois archives signées vérifiées, la correspondance fine des six canaux aux cordes reste à prouver |
| Acoustique | PASS tests | 256 nouveaux PCM, reconstruction par réseau contre vrais signaux hex |
| Architecture | PASS tests | CNN+GRU causal, transporteur +12/−12 et sans-vues, six autres têtes inchangées |
| Apprentissage | PASS vrai entraînement | 640 couples d'octaves, poids PyTorch appris en quatre époques sur prises réelles |
| Métriques | FAIL notes / Onset UNKNOWN | 0 TP E2/E3 transposé sur évaluation par bloc ; aucun NoteOn nouveau évalué ; hex MAE sans baseline silence |
| LIVE | UNKNOWN | aucun timing CPU Windows/iRig validé, aucun MIDI |

## Verdict
**TERMINÉ, NON PROMU** : vraie tête formée à partir de sons hex réels, mais aucun E2/E3 correctement détecté par la version transposée dans l'évaluation S843. Les reconstructions et transpositions audio sont des **estimations neuronales**, pas la garantie de restituer un son réellement présent. La tête ne donne pas directement de NoteOn et n'a aucun effet sur la référence Onset.

## Prochaine action
Faire réussir les tests de conformité et l'apprentissage hex à partir de véritables prises GuitarSet. Auditer de manière systématique la correspondance des six cordes et le décalage physique entre micro/hex, puis comparer sans-vues contre ±12 sur E2, E3 et leurs recouvrements. Uniquement après de vrais gains au niveau des notes, créer une tête Onset qui exploite ces preuves, garder les six autres têtes gelées et évaluer les 80 WAV sans retard.
