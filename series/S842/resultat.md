---
serie: S842
statut: en_cours
type: experience
reference: S708
branche: codex/note-s842-real-guitarset-neural-transpose256
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S842 — Résultats de l'entraînement appris de transposition sur GuitarSet

## Exécution
Expérience de comparaison **avec vs sans vues transposées latentes** sur des vraies paires octaves de notes guitar et un test ciblé E2, E3 et E2+E3. Cette branche n'alimente pas le MIDI final, ne modifie pas les cinq autres caractéristiques, ni Onset officiel. Le code réutilise S840 mais entraîne aussi sa fusion de vues sur de vraies annotations, ce que la première version ne faisait pas.

## Métriques
| Référence | Score | Protocole |
|---|---:|---|
| Onset F1 69,2584331 % | 7 607 TP / 3 093 FP / 3 660 FN | 80 WAV GuitarSet PCM256, témoin expérimental |
| HGB Onset | 69,2489759 % | rejeté |
| S841 GT E2 | 153 attaques | réelle GT, pas un score |
| S841 GT E3 | 278 attaques | réelle GT, pas un score |
| S841 GT E2+E3 overlap | 115 intersections / 8 enregistrements | GT durées, pas un score |
| Réseau S842, E2, E3, E2+E3 | **NON MESURÉ** | en cours sur vraies paires et probe |
| Exact-K | inchangé | indépendant |
| Active Exact-K | inchangé | indépendant |
| Offset | inchangé | indépendant |
| Frame Pitch F1 | inchangé, proxy S842 non encore mesuré | indépendant |
| Full Note F1 | inchangé | indépendant |

## Comparaison à S708
S708/H0 historique FIT80 a un Onset F1 de 73,252019 % mais son apprentissage à cadence 1024 est hors de notre étude. Notre témoin PCM256 indépendant est 69,2584331 % ; **aucune valeur de cette expérimentation ne doit être assimilée à un gain de F1 Onset** tant que les NoteOn ne sont pas générés et appariés.

## Audits
| Rôle | État | Contrôle |
|---|---|---|
| Données | UNKNOWN | extraction des paires réelles, vérification SHA et split |
| Acoustique | PASS code | blocs bruts de 256, aucune modification par DSP |
| Architecture | PASS code | encodeur/tansporteur/vues, témoin nu, 5 autres axes protégés |
| Apprentissage | UNKNOWN | nécessité du vrai train GuitarSet et checksum |
| Métriques | UNKNOWN | E2/E3/JOIN F1 proxy et Onset F1 vrai distincts |
| LIVE | UNKNOWN | aucune mesure CPU Windows/iRig, pas de MIDI |

## Verdict
Non promu. En attendant les preuves d'apprentissage GuitarSet et de généralisation sur SS1, la transposition ne peut être qualifiée de meilleure pour identifier E2/E3. Même un éventuel bon proxy de notes actives ne suffit pas à conclure sur les erreurs d'Onset.

## Prochaine action
Lire les artefacts GitHub des paires originales et du modèle S842 ; analyser les exemples TP/FP/FN du E2/E3 simultané et les sources des éventuelles régressions. Ensuite, seulement si l'ablation le justifie, créer un générateur de candidats de NoteOn indépendant et rejouer intégralement 80 WAV à 256 avant d'envisager une promotion.
