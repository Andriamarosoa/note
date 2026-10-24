---
serie: S851
statut: en_cours
type: experience
reference: S708
branche: codex/note-s851-identical-guitar-standard-ab256
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S851 — Comparatif même son réel, même référence standard, quatre décodeurs IA

## Exécution
La preuve externe requise est le même WAV de guitare et le même pitch shift standard +12 que S848 ; quatre sorties S850 (BCE/RANK × état continu/reset32) seront mesurées mais **aucune ne sera sélectionnée sur la sortie standard**. Le réseau promu n'existe pas ; code shadow only.

## Métriques préexistantes
| Référence | Mesure |
|---|---|
| S848 standard +12 vs original, dynamique | **90,38 %** |
| S848 S847 IA vs standard, dynamique | **11,64 %** |
| S848 IA S847 note MIDI73 choisie | **603/603** |
| S849 reset8 ou reset32 sur ancien modèle | **603/603 MIDI73** |
| S850 BCE E2→E3 sur vrai OUTER guitariste05 | **26,8071 % F1** de présence (TP1463, FP7782, FN207) |
| S850 BCE E3→E4 sur vrai OUTER | **39,9559 % F1** (TP1993, FP4977, FN1013) |
| S850 RANK E2→E3 | **12,7794 % F1** |
| S850 RANK E3→E4 | **21,9486 % F1** |
| S850 sélection sur CAL | **BCE**, RANK rejeté |
| Nouveau WAV BCE vs standard | **NON MESURÉ** |
| Nouveau WAV RANK vs standard | **NON MESURÉ** |
| Onset F1 80 WAV travail | **69,2584331 %**, inchangé |
| Autres cinq axes et MIDI | indépendants, pas de mise à jour |

## Audits
| Domaine | État |
|---|---|
| Données | à vérifier par artefacts GitHub S847/S848/S850 |
| Architecture | code raw256/GRU, standard offline à part |
| Apprentissage | S850 terminé, BCE vainqueur CAL |
| Métriques | audio/choix notes à évaluer, aucune note correcte inventée |
| LIVE | aucune intégration ni p99 Windows |

## Verdict
Non promu en attendant le A/B audio. Même une meilleure note active GT ne prouve pas que l'on a un vrai transposeur de performance ; une sortie bloquée ou artificiellement diversifiée sera rejetée.

## Prochaine action
Comparer les quatre sorties au standard original, auditer si la mémoire causale 32 blocs est réellement utile. Si insuffisant, revoir représentation temporelle et polyphonie plutôt que masquer FP par seuil.
