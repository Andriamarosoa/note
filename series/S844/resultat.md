---
serie: S844
statut: en_cours
type: experience
reference: S708
branche: codex/note-s844-real-hex-rare-note-coverage256
run_id: en_attente
commit_sha: en_attente
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

## Métriques S844
| Axe | Résultat |
|---|---|
| S844 256 E2 F1 GT par bloc | **NON MESURÉ**, attendre RUN scientifique |
| S844 256 E3 F1 GT par bloc | **NON MESURÉ** |
| S844 256 E2+E3 GT par bloc | **NON MESURÉ** et doit avoir des positifs |
| S844 reconstruction hex vs silence | **NON MESURÉ** |
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
| Données | UNKNOWN | 3 archives vérifiées, classes E2/E3 et chevauchement présentes, trains par joueur indépendants |
| Acoustique | PASS code | nouveau bloc 256, aucune attente ni audio du futur |
| Architecture | PASS code | perte note pondérée TRAIN uniquement, autres têtes et MIDI protégés |
| Apprentissage | UNKNOWN | entraînement réel et preuve de positifs par split |
| Métriques | UNKNOWN | CAL seulement pour seuil, OUTER jamais pour choix, silence MAE baseline |
| LIVE | UNKNOWN | aucun p99 Windows/iRig end-to-end |

## Verdict
Non promu. La validité de l'hypothèse E2/E3 n'est déterminable qu'avec les vrais résultats S844 sur audio originaux. Les faux positifs massifs du témoin S843 et l'échec du transport ne doivent pas être cachés ou réétiquetés comme des gains.

## Prochaine action
Lire l'artefact complet du run ; s'il existe des positifs E2/E3/ou chevauchement, comparer les mêmes fragments avec/sans transpose, les erreurs FP/FN, la reconstruction acoustique avec prédiction silence et les poids de vues. En cas de gain vérifié, intégrer plus tard comme **propositions d'Onset indépendantes**, puis replay 80 WAV et appareil Windows avant toute promotion.
