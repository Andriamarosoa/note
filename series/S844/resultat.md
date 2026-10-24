---
serie: S844
statut: termine
type: experience
reference: S708
branche: codex/note-s844-real-hex-rare-note-coverage256
run_id: 40087751049
commit_sha: f5b82d620e72db706e7ed64bae9d5d567a8c82f7
promotion: false
---

# S844 — Apprentissage acoustique réel orienté E2/E3, résultats

## Exécution
Branche de correction indépendante du modèle S843, entraînement à nouveau sur archives GuitarSet signées et captation hex à six cordes réelles. Entrée mono **256 échantillons** au moment présent, mémoire GRU des décisions antérieures, aucun audio du futur ni calcul fixe de pitch-shift, aucune modification des six axes MIDI.

### Contrôle historique reproductible S843
[GitHub Actions #40086388680](https://github.com/Andriamarosoa/note/actions/runs/40086388680) a terminé **SUCCESS** : les trois archives originales ont été vérifiées par checksum, 60 vraies pistes utilisées (10 TRAIN guitariste 04, 10 CAL guitariste 05, 40 OUTER guitaristes 00..03), 4 époques, 640 paires d'octave réelles, 8960 blocs PCM256 OUTER.

| Reconnaissance par bloc | Sans transpose S843 | Transposition IA S843 |
|---|---:|---:|
| E2 | 1,6163 % (TP 73 / FP 8887 / FN 0) | 0 % (TP 0 / FP 0 / FN 73) |
| E3 | 2,0983 % (TP 95 / FP 8865 / FN 0) | 0 % (TP 0 / FP 0 / FN 95) |
| E2+E3 | **Non évaluable** (0 positif OUTER, 8960 FP) | **Non évaluable** (0 positif OUTER, 0 FP) |

Le run antérieur reportait naïvement 0 % E2+E3, mais il n'existait aucune vérité positive de superposition dans cette sélection. La reconstruction hex moyenne avait MAE **0,028968** dans le domaine de pickup ; aucun témoin zéro n'avait été mesuré. Le timing CPU Linux observé sur huit pas uniquement (p99 2,389 ms) n'est pas un certificat temps réel.

## Expérience S844 achevée sur les vraies prises GuitarSet

[GitHub Actions #40087751049](https://github.com/Andriamarosoa/note/actions/runs/40087751049) **SUCCESS**. Tests initiaux **11+5 PASS**, trois archives originales signées, 16 pistes TRAIN / 16 CAL / 64 OUTER. Échantillons GT E2/E3/joint :
- TRAIN : **1172 / 1347 / 718** ; poids E2=3,0797, E3=2,85 calculés sur TRAIN seulement ;
- CAL : **684 / 1313 / 376** ;
- OUTER : **3889 / 4953 / 1921** sur 49 152 décisions strictement PCM256.

Réseau entraîné huit époques, checkpoint au meilleur CAL (époque 8), **1 424 paires d'octaves réelles** pour la perte de transport. Comparaison sur les **mêmes** blocs OUTER SS1/etc. :

| Cible GT active | Sans transposition (TP/FP/FN) | Sans transposition F1 | Avec IA transposition (TP/FP/FN) | Avec IA F1 |
|---|---:|---:|---:|---:|
| E2 | 1396 / 10609 / 2493 | **17,5664 %** | 1147 / 8190 / 2742 | **17,3446 %** |
| E3 | 2913 / 18375 / 2040 | 22,2019 % | 2913 / 18366 / 2040 | **22,2095 %** |
| E2+E3 | 1094 / 16845 / 827 | **11,0171 %** | 1068 / 16437 / 853 | 10,9956 % |

**Pas de gain significatif et régulier des vues latentes de transposition.** Les scores restent des **F1 de note active**, pas de NoteOn Onset. Comparer S843 et S844 directement en points de F1 est impossible car les sélections de fenêtres diffèrent.

**Auditeur réel de forme d'onde** depuis l'artefact `S844_REAL_GUITARSET_RARE_E2_E3_NEURAL_RECONSTRUCTION_PCM256` (#12557298859) : prédicteur silence **MAE=0,029205378**, modèle neuronal **MAE=0,029785044**, sortie moyenne absolue **0,001868151** contre vraie cible hex **0,029205378**. **Le modèle S844 reconstruit moins bien les vraies pistes hex que la simple prédiction zéro.** Le code de décodeur global avec atténuation par confiance peut avoir favorisé la quasi-extinction du signal, mais ce mécanisme exact n'est pas expérimentalement prouvé.

Le F1 GT de présence des six cordes (après threshold CAL) est **57,2035 %**, pas une preuve de séparation acoustique. CPU Linux p99 **2,7235 ms sur 8 pas seulement** ne prouve rien sur le PC iRig ou la chaîne complète.

**Verdict : entraînement réussi, reconstruction audio et transposition insuffisantes, NON PROMU.** Référence Onset PCM256 reste **69,2584331 % F1**. La nouvelle S845 essaie de préserver la forme d'onde vraie microphone dans le décodeur neural ; ce n'est pas un engagement de gain.

## Métriques S844
| Axe | Résultat |
|---|---|
| S844 256 E2 F1 GT par bloc | **17,3446 %** avec IA, **17,5664 %** sans vues latentes |
| S844 256 E3 F1 GT par bloc | **22,2095 %** avec IA, **22,2019 %** sans vues |
| S844 256 E2+E3 GT par bloc | **10,9956 %** avec IA, **11,0171 %** sans vues |
| S844 reconstruction hex vs silence | **0,029785** MAE modèle vs **0,029205** silence : échec |
| Référence Onset 80 WAV 256 | **69,2584331 % F1** ; non modifiée |
| Exact-K | inchangé / non réévalué |
| Active Exact-K | inchangé / non réévalué |
| Offset | inchangé / non réévalué |
| Frame Pitch F1 | inchangé / non réévalué |
| Full Note F1 | inchangé / non réévalué |

## Comparaison à S708
S708/H0 est 73,252019 % Onset FIT80 à entraînement historique 1024, **non** notre témoin PCM256 indépendant. Aucun résultat S844 de classification de notes actives n'est une métrique Onset.

## Audits
| Contrôle | État initial | Attendu |
|---|---|---|
| Données | PASS | 3 archives vérifiées, classes E2/E3 et chevauchement présentes, trains par joueur indépendants |
| Acoustique | PASS code | nouveau bloc 256, aucune attente ni audio du futur |
| Architecture | PASS code | perte note pondérée TRAIN uniquement, autres têtes et MIDI protégés |
| Apprentissage | PASS | entraînement réel et preuve de positifs par split |
| Métriques | FAIL | CAL seulement pour seuil, OUTER jamais pour choix, silence MAE baseline |
| LIVE | UNKNOWN | aucun p99 Windows/iRig end-to-end |

## Verdict
**TERMINÉ, NON PROMU**. Les vrais résultats E2/E3 et la reconstruction n'ont pas fourni les preuves nécessaires. Les faux positifs massifs du témoin S843 et l'échec du transport ne doivent pas être cachés ou réétiquetés comme des gains.

## Prochaine action
Lire l'artefact complet du run ; s'il existe des positifs E2/E3/ou chevauchement, comparer les mêmes fragments avec/sans transpose, les erreurs FP/FN, la reconstruction acoustique avec prédiction silence et les poids de vues. En cas de gain vérifié, intégrer plus tard comme **propositions d'Onset indépendantes**, puis replay 80 WAV et appareil Windows avant toute promotion.
