---
serie: S837
statut: en_cours
type: experience
reference: S708
branche: codex/note-s837-six-axis-80pct-independent-loops256
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S837 — Boucles multi-axes indépendantes, résultat contrôlé

## Exécution
Code S837 et grille indépendante par axe créés, CI et cinq enregistrements réels à lancer. Les résultats sont **en attente de la première exécution**. Aucun résultat Onset ou autre n'est déclaré à partir de cette modification tant que les traces et l'audit des régressions ne sont pas disponibles.

## Métriques
Référence scientifiquement disponible, 80 GuitarSet FIT80 SHADOW S834 :
| Axe | Score de référence | Objectif |
|---|---:|---:|
| Exact-K (accuracy toutes trames) | 75,780659 % | >80 % |
| Active Exact-K (accuracy trames actives) | 65,193596 % | >80 % |
| Onset F1 | 71,382285 % | >80 % |
| Offset F1 événement | 38,677437 % | >80 % |
| Frame Pitch F1 | **84,998850 %** | préserver >80 % |
| Full Note F1 | 71,108037 % | >80 % |

Référence protégée S708/H0 FIT80 : Onset F1 73,252019 %. Offset accuracy 69,886229 % n'est **pas comparable** à Offset F1. Pas de nouveau score natif256 prêt pour le Live et pas de modèle unifié revendiqué. Les scores sous 80 % sont le cœur du problème scientifique.

## Comparaison à S708
Les modèles acoustiques S708/H0 historiques sont gelés, entraînés originellement à un pas de décision 1024 et inférés sur des blocs nouveaux de 256. Les références CAL32 officielles demeurent protégées. Le code S837 n'écrit ni le MIDI final, ni les checkpoints originaux et ne sélectionne aucune politique en regardant sa composition OUTER.

## Audits
| Rôle | État | Preuve / limite |
|---|---|---|
| Données | UNKNOWN | original GuitarSet signé à vérifier |
| Acoustique | UNKNOWN | PCM256 réel à vérifier |
| Architecture | UNKNOWN | invariances multi-axes en CI |
| Apprentissage | PASS | aucun checkpoint modifié |
| Métriques | UNKNOWN | 6 évaluations appariées à exécuter |
| LIVE | UNKNOWN | p95/p99 Windows/iRig non mesuré |

## Verdict
**NON PROMU.** L'optimisation CPU S836 ne donne pas de gain musical et la grille des nouvelles politiques n'a pas encore de score vérifié. Régression Onset S834, faiblesse Offset et performance des accords K3/K4 à auditer avant amélioration.

## Prochaine action
Exécuter la grille indépendante sur cinq vrais WAV, analyser chaque TP récupéré/perdu et chaque FP introduit avant une deuxième expérience. Étendre à 80/80 dès que le protocole est prouvé stable. Toute progression isolée doit être rejetée si l'un des cinq autres axes régresse. Si aucune politique Onset ou Offset n'atteint 80 %, documenter le plafond de cette famille de stratégies puis tester de nouvelles propositions acoustiques entraînées séparément, sans attendre de futur audio.
