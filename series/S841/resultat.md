---
serie: S841
statut: termine
type: experience
reference: S708
branche: codex/note-s841-real-e2-e3-octave256
run_id: 40083001531
commit_sha: afed37989d7592f8298e305dd30f294f9485e66b
promotion: false
---

# S841 — Résultats du diagnostic GuitarSet octave E2 / E3

## Exécution
[GitHub Actions #40083001531](https://github.com/Andriamarosoa/note/actions/runs/40083001531) **SUCCESS**, 3 tests, audit signé et 80 WAV d'origine contrôlés, 306 536 décisions PCM256. Cette série ne modifiera aucun modèle ou application. Les vrais WAV sont contrôlés par SHA à partir des sources S800 et des traces FIT80. Les groupes de composition sont BN1/Funk2/Jazz1/Rock3/SS1, 16 WAV chacun, total attendu de 306536 blocs PCM256.

## Métriques
| Signal | Score actuel |
|---|---:|
| Onset F1 témoin PCM256 | **69,2584331 %**, TP7607 / FP3093 / FN3660 |
| HGB ajouté | 69,2489759 % ; rejeté |
| Nouvelle transposition neuronale E2 / E3 / E2+E3 | **NON MESURÉ** |
| Exact-K | Inchangé, indépendant |
| Active Exact-K | Inchangé, indépendant |
| Offset | Inchangé, indépendant |
| Frame Pitch F1 | Inchangé, indépendant |
| Full Note F1 | Inchangé, indépendant |

Les métriques de présence E2/E3 dans la vérité terrain sont des **comptes de GT**, pas des métriques d'amélioration. Onset F1 s'évalue sur NoteOn avec correspondance par hauteur et temps, et non sur F1 par trame ou simplement présence de l'harmonique.

## Comparaison à S708
S708/H0 historique, modèle entraîné à 1024, est une référence administrative protégée : **73,252019 %** non comparable comme architecture native S840. Le témoin actuel 69,2584331 % correspond à la branche Onset PCM256 déjà replayée. La mesure E2/E3 de cette série n'est pas un nouveau score d'Onset.

### Répartition vérifiée S841 par composition

| Composition | Attaques E2 | Attaques E3 | Superpositions de durées E2+E3 |
|---|---:|---:|---:|
| BN1 | 9 | 7 | 0 |
| Funk2 | 4 | 11 | 0 |
| Jazz1 | 15 | 53 | 13 |
| Rock3 | 51 | 153 | 62 |
| SS1 | 74 | 54 | 40 |
| **Total** | **153** | **278** | **115** |

**39** paires d'attaques GT sont proches de 80 ms, mais **seulement 10** à distance <= 256/44100 secondes. Ces deux nombres de proximité ne démontrent pas le sustain. Les **115 intersections** reposent sur les débuts et durées des notes GT et ne correspondent pas forcément à 115 accords distincts. Ce sont des **comptes de GT** et non le résultat d'un modèle musical. Les notes MIDI 40/52 représentent les hauteurs E2 et E3.

## Audits
| Rôle | État | Précaution |
|---|---|---|
| Données | PASS | 80 WAV originaux, 80 traces SHA et manifest S800 signés |
| Acoustique | PASS code | analyse des dates d'attaque, zéro traitement de transposition DSP |
| Architecture | PASS code | S840 original et cinq autres têtes inchangés |
| Apprentissage | UNKNOWN | paires de notes physiquement appariées absentes |
| Métriques | UNKNOWN | GT E2/E3 et E2+E3 mesurés, aucun F1 Onset nouveau |
| LIVE | UNKNOWN | aucune mesure CPU Windows/iRig |

## Verdict
**AUDIT TERMINÉ ET CONFORME** à l'objet GT, aucun Onset F1 nouveau. L'expérimentation S842 porte séparément sur l'entraînement IA. Deux NoteOn proches à 80ms ne constituent pas à eux seuls un vrai chevauchement E2+E3 ; les durées de notes doivent être vérifiées. Ne pas fabriquer d'exemples positifs, ni sélectionner sur OUTER. Aucun score nouvellement promu.

## Prochaine action
Utiliser la provenance du rapport signé GT pour chercher de vraies prises distinctes de E2, E3 et E2+E3. Constituer les paires et entraînements neuronaux à 256 uniquement si les conditions physiques et de vérité terrain sont démontrées, comparer avec/sans vues en gardant le Live public figé et évaluer finalement 80 WAV et données externes sans fuite.
