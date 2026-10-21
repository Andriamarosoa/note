---
serie: S825
statut: en_cours
type: experience
reference: S708
branche: codex/note-s825-private-crossfit-pitch-midi256
run_id: en_attente
commit_sha: en_attente
promotion: false
---
# S825 — Résultats des boucles de cardinalité apprises sur pseudo-postérieurs indépendants PCM256

## Exécution

En attente du run scientifique : 80 pistes FIT80, 5 folds composition-outer, chaque fold 3 sous-folds de pitch OOF, puis trois classifieurs Active-K distincts. Les métriques proviendront des 249411 trames actives MIDI256 originales.

## Métriques

| Expérience | Active Exact-K FIT80 | Statut |
|---|---:|---|
| H0 source émis MIDI256 | 57,019538 % | Référence historique |
| S815 3 têtes Active-K indépendantes MIDI256 | 56,912486 % | Comparateur |
| S823 S795 dépendant S708, snapshot 1024 | 76,877523 % | Diagnostic non admissible |
| S825 pitch_stats | NON MESURÉ | Modèle indépendant |
| S825 pitch_harmonic | NON MESURÉ | Modèle indépendant |
| S825 pitch_temporal | NON MESURÉ | Modèle indépendant |
| S825 calibration-only | NON MESURÉ | Sélection sur CAL uniquement |

## Six caractéristiques officielles S708 CAL32

| Caractéristique | S708 CAL32 | S825 |
|---|---:|---|
| Onset F1 | 74,2639 % | Non modifié |
| Exact-K | 62,0455 % | Non modifié |
| Active Exact-K | 56,2762 % | Recherche SHADOW seule |
| Offset accuracy | 77,3292 % | Non modifié |
| Full Note F1 | 57,1319 % | Non modifié |
| Frame Pitch F1 | 76,4626 % | Non modifié |

## Comparaison à S708

S708 officiel et LIVE inchangés. S825 peut produire une décision à chaque hop de 256 échantillons, mais le temps d'extraction et d'inférence réel doit encore être mesuré; aucun échantillon futur ni hold supplémentaire.

## Audits

| Domaine | Statut | Justification |
|---|---|---|
| Données | UNKNOWN | SHA archives 80 sources non encore exécutées |
| Acoustique | UNKNOWN | Entrée exclusivement DSP native256 |
| Architecture | UNKNOWN | Têtes propres et aucune feature S708/MIDI |
| Apprentissage | UNKNOWN | 3 sous-plis OOF et 48/16/16 à authentifier |
| Métriques | UNKNOWN | 249411 active frames et sous/sur par K |
| LIVE | UNKNOWN | Pas de futur/hold ; latence microphone non mesurée |

## Verdict

Aucune promotion ni unification : métriques réelles S825 non encore disponibles.

## Prochaine action

Comparer les trois ablations sur les mêmes partitions; conserver le meilleur **uniquement** s'il n'aggrave pas K1–K6 et les cinq autres caractéristiques, puis demander validation hors FIT80 et test matériel LIVE.
