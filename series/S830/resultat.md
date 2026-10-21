---
serie: S830
statut: termine
type: experience
reference: S708
branche: codex/note-s830-private-musical-evidence256
run_id: 39632626042
commit_sha: 088e28e122bfc1f04c981716bf8db73faa615ca7
promotion: false
---

# S830 — Résultats scientifiques sur 80 pistes FIT80 natif256

## Exécution
**COMPLETED SUCCESS** — [GitHub Actions #39632626042](https://github.com/Andriamarosoa/note/actions/runs/39632626042), commit scientifique `088e28e122bfc1f04c981716bf8db73faa615ca7`. Deux jobs `smoke` et `signed_fit80` ont réussi. L'artefact `S830_FIT80_INDEPENDENT_MUSICAL_HEADS_PCM256` (id `12360813096`) contient `report.json` complet avec 80 pistes, cinq folds et 306 536 trames. Source : S800 FIT80 avec métadonnées et WAV originaux vérifiés. La première exécution a échoué sur la fin PCM partielle, bug corrigé avant la validation finale.

## Métriques
Le dénominateur Exact-K est constitué de **toutes les trames K0–K6**, y compris K0 et non uniquement des trames actives.

| Source | Correctes / 306 536 | Exact-K | Sous-comptages | Surcomptages |
|---|---:|---:|---:|---:|
| H0 original (même FIT80) | 189 124 | **61,6972 %** | 80 654 | 36 758 |
| S829 réentraîné (même FIT80) | 176 334 | **57,5247 %** | 83 523 | 46 679 |
| **S830 (5 têtes privées)** | **176 334** | **57,5247 %** | **83 523** | **46 679** |

**Différence S830 – S829 : 0,0000 point**. S830 est à −4,1724 points du témoin H0. Parité parfaite et régression nulle du chemin S829 dans l'expérience. Aucune amélioration des classes polyphoniques démontrée.

| Composition OUTER | Exact-K S830 | Mode retenu inner-CAL | Seuil | Shift |
|---|---:|---|---:|---:|
| BN1 | 61,31 % | `s829` | 0,30 | 0 |
| Funk2 | 61,11 % | `s829` | 0,30 | 0 |
| Jazz1 | 57,48 % | `s829` | 0,30 | 0 |
| Rock3 | 54,83 % | `s829` | 0,30 | 0 |
| SS1 | 53,37 % | `s829` | 0,30 | 0 |

Le choix `s829` aux cinq folds signifie que les sorties du modèle musical sont **ignorées pour le K final** ; elles ont été entraînées et évaluées en CAL mais le sélecteur a retenu le témoin. Le rapport courant enregistre seulement le score CAL du gagnant, **pas** toutes les métriques des modes alternatifs, ce qui empêche de quantifier leur déficit par rapport à S829 sans nouvel audit ciblé.

| Vrai K | Trames | Exact-K S830 | Sous-comptages |
|---|---:|---:|---:|
| K0 | 57 125 | 80,30 % | 0 |
| K1 | 112 014 | 83,37 % | 9 160 |
| K2 | 46 311 | 22,02 % | 24 957 |
| K3 | 34 182 | 26,29 % | 14 612 |
| K4 | 35 190 | 43,61 % | 16 132 |
| K5 | 16 323 | 12,85 % | 13 722 |
| K6 | 5 391 | 8,37 % | 4 940 |

## Six métriques officielles du MIDI unifié
| Caractéristique | Résultat S830 (MIDI final) |
|---|---|
| Onset F1 | NON MESURÉ — axe intact |
| Exact-K | 57,5247 % SHADOW K uniquement ; le MIDI final n'a pas été reconstitué |
| Active Exact-K | NON MESURÉ — S828 reste la référence de développement séparée |
| Offset | NON MESURÉ — axe intact |
| Full Note F1 | NON MESURÉ — pas d'intégration |
| Frame Pitch F1 | NON MESURÉ — axe intact |

La référence officielle S708 CAL32 est protégée. S809 84,24 % SHADOW à 1 024 échantillons et S828 73,83 % Active Exact-K à 256 ont des protocoles différents, donc ne peuvent pas être comparés directement à cette métrique.

## Audits
| Catégorie | Statut | Preuve |
|---|---|---|
| Données | PASS | 80 WAV GuitarSet d'origine, SHA provenance, composition-disjoint |
| Acoustique | UNKNOWN | Tests synthétiques causalité/PCM256 PASS, mais généralisation acoustique et matériel iRig non validés |
| Architecture | UNKNOWN | Isolation de code PASS et cinq autres axes inchangés ; audit adversarial de toutes les interactions non réalisé |
| Apprentissage | PASS | Deux experts K + trois preuves musicales appris sur TRAIN, selector inner-CAL |
| Métriques | PASS | 306 536 trames K0–K6 et comparaison S829/H0 appariée |
| LIVE | UNKNOWN | p95/p99/latence matérielle iRig non mesurée |

Les six agents Work et trois avis NVIDIA authentiques n'ont pas été invoqués. Aucun certificat de généralisation HELD, de latence physique ou de non-régression du MIDI unifié.

## Comparaison à S708
Aucun changement de référence S708, S828, MIDI officiel ou des cinq autres axes. S830 s'exécute en SHADOW, indépendant. Son sélecteur conservateur a systématiquement retenu S829 ; il n'y a ni gain ni perte.

## Verdict
**Expérience TERMINÉE — NON PROMUE.** L'indépendance est possible mais les preuves musicales nouvelles, dans leur forme actuelle, n'apportent pas de gain Exact-K en sélection CAL. Avant d'étudier une influence entre axes, instrumenter les performances par candidat et par classe K, puis comparer les corrections conditionnelles sur CAL et OUTER sans modifier les producteurs de caractéristiques. La contrainte zéro lookahead/hold reste obligatoire.

## Prochaine action
Ne pas promouvoir S830. Instrumenter le score des sorties de chaque expert sur CAL et par vrai K, puis étudier une communication en lecture seule et sens unique entre architectures avec ablations et veto de régression, sans fusion des états et sans ajouter de latence.
