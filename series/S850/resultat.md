---
serie: S850
statut: en_cours
type: experience
reference: S708
branche: codex/note-s850-causal-multinote-rank256
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S850 — Deux entraînements indépendants BCE vs RANK sur les vraies notes de guitare

## Exécution
Les deux variantes apprennent **à partir de la véritable guitare GuitarSet et des annotations JAMS** — pas d'échantillons de notes synthétisés, ni de DSP pitch-shift dans la tête. TRAIN guitaristes 00–03 / CAL 04 / OUTER 05, aucune fuite entre joueurs. Séquences d'apprentissage 32 blocs PCM256 passés, mais nouvelle entrée Live toujours exactement **256** ; aucune attente ni MIDI officiel.

## Métriques
| Contrôle | Mesure |
|---|---|
| S848 original, standard +12 corrélation d'enveloppe | **90,38 %** |
| S848 original, IA S847 +12 corrélation d'enveloppe | **11,64 %** |
| S848 note MIDI 73 choisie | **603/603** : échec |
| S849 GRU reset tous les 8 hops | **603/603** MIDI 73 : échec |
| S849 GRU reset tous les 32 hops | **603/603** MIDI 73 : échec |
| S850 variante BCE, note GT OUTER +12 | **NON MESURÉ** avant run |
| S850 variante RANK, note GT OUTER +12 | **NON MESURÉ** avant run |
| S850 top1 GT vs validation | **NON MESURÉ** |
| S850 audio vs standard | **NON MESURÉ** avant test final à source égale |
| Onset F1 PCM256 référence | **69,2584331 %**, intact |
| Les six têtes NOTE et MIDI | indépendants, non modifiés |

## Comparaison à S708
Le modèle historique S708 avait été entraîné à 1024 échantillons et ne sert pas à former les nouvelles têtes. S850 est une architecture neuronale brute native PCM256 indépendante ; aucun score de présence de notes ne peut remplacer le Onset F1 officiel.

## Audits
| Axe | État | Vérification |
|---|---|---|
| Données | UNKNOWN | archives authentiques SHA et classe E2/E3 GT, 4/1/1 joueurs disjoints |
| Acoustique | PASS code | entrée 256, GRU passée, pas de délai algorithmique |
| Architecture | PASS code | BCE témoin et RANK séparés avec activité note et polyphonie |
| Apprentissage | UNKNOWN | vrais modèles, checkpoints CAL uniquement |
| Métriques | UNKNOWN | vraie F1 par note OUTER et base standard après training |
| LIVE | UNKNOWN | p99 CPU appareil Windows/iRig et MIDI non mesurés |

## Verdict
**NON PROMU** tant que les sorties ne sont pas validées. Si la sélection devient diverse mais incorrecte, c'est toujours un échec. Une entrée audio réelle en banque ne garantit pas à elle seule une vraie transposition de la performance originale.

## Prochaine action
Consulter le run S850 et comparer BCE vs RANK sur guitariste OUTER 05 sans sélectionner par OUTER. Vérifier les FP et FN par note. Ensuite comparer le meilleur CAL sur le même WAV et standard S848. Si les deux échouent, réaliser un audit de l'information acoustique disponible dans PCM256 et réviser l'objectif, pas les seuils à l'aveugle.
