---
serie: S823
statut: en_cours
type: experience
reference: S708
branche: codex/note-s823-legacy-s795-features-causal256
run_id: en_attente
commit_sha: en_attente
promotion: false
---
# S823 — Vraies caractéristiques S795 reportées causalement à MIDI256

## Exécution

Replay diagnostique de 80 snapshots OOF S800 signés, avec leur cadence d'observation S795 1024, puis décisions et métriques MIDI256. Un tel replay ne constitue PAS une ré-extraction acoustique 256.

## Métriques

| Active Exact-K | Score FIT80 |
|---|---:|
| S810 1024 SHADOW, 62310 frames actives | 76,5656 % (protocole différent) |
| H0 MIDI256, 249411 frames actives | 57,019538 % |
| S822 RAW23 PCM256 indépendant | 54,454695 % |
| S822 AGG61 PCM256 indépendant | 54,385733 % |
| S823 vrais S795 RAW23 reportés 256 | NON MESURÉ |
| S823 vrais S795 AGG61 reportés 256 | NON MESURÉ |
| S823 CAL-selected | NON MESURÉ |

## Six caractéristiques S708 CAL32

| Axe | S708 officiel |
|---|---:|
| Onset F1 | 74,2639 % |
| Exact-K | 62,0455 % |
| Active Exact-K | 56,2762 % |
| Offset accuracy | 77,3292 % |
| Full Note F1 | 57,1319 % |
| Frame Pitch F1 | 76,4626 % |

## Comparaison à S708

Aucune modification S708, MIDI final ou LIVE. Inputs S795 contiennent S708 postérieurs, K et notes émises : **dépendance inter-axes**. L'observation n'est actualisée qu'à 1024 même si chaque décision est rendue à 256.

## Audits

| Rôle | Statut | Critère |
|---|---|---|
| Données | UNKNOWN | 80 JSON/NPZ SHA, H0 MIDI256 142213 |
| Acoustique | UNKNOWN | Vrais canaux mixtes historiques, pas DSP 256 nouvelle |
| Architecture | UNKNOWN | 2 nouveaux modèles, inputs historiquement liés à S708 |
| Apprentissage | UNKNOWN | 48/16/16 composition exclue |
| Métriques | UNKNOWN | 249411 frames actifs |
| LIVE | UNKNOWN | Non intégré et latence micro non mesurée |

## Verdict

Non promouvable quelle que soit la précision; aucun score inventé.

## Prochaine action

Comparer aux 23/61 variables S822 natives, identifier l'information additionnelle contenue dans les sorties S708 et tracer la voie d'un modèle indépendant qui ne reçoit que le son présent ou passé.
