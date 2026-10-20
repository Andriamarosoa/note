---
serie: S806
statut: en_cours
type: experience
reference: S708
branche: codex/note-s806-physical-fundamental-contrast-oof
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S806 — résultats de la nouvelle représentation acoustique

## Exécution

Résultats S806 : **NON MESURÉS** tant que le run scientifique sur 80 traces signées S800 avec extraction des 16 caractéristiques et décodeur MIDI256 inchangé n'est pas fini. Sources de vérité S800 [run 39352880035](https://github.com/Andriamarosoa/note/actions/runs/39352880035), S803 [run 39363062723](https://github.com/Andriamarosoa/note/actions/runs/39363062723). Série démarrée avec les deux rapports obligatoires avant tout résultat ; les scores ne peuvent pas être inventés.

## Métriques

| Métrique | S708 officiel CAL32 | S806 FIT80 (à mesurer) |
| --- | ---: | --- |
| Onset F1 | 74,2639 % | NON MESURÉ |
| Exact-K | 62,0455 % | NON MESURÉ |
| Active Exact-K | 56,2762 % | NON MESURÉ |
| Offset accuracy | 77,3292 % | NON MESURÉ |
| Full Note F1 | 57,1319 % | NON MESURÉ |
| Frame Pitch F1 | 76,4626 % | NON MESURÉ |

Témoin indépendant FIT80 S803 : Onset 73,2520 %, Exact-K 61,6972 %, Active Exact-K 57,0195 %, Offset 69,8862 %, Full Note F1 50,6528 %, Frame Pitch F1 78,7278 %. **Ne pas comparer CAL32 et FIT80.**

## Comparaison à S708

Aucune promotion, aucun changement de NoteOn/NoteOff ou de modèles de production LIVE. Les candidats restent SHADOW. Le calcul des nouvelles caractéristiques peut avoir un coût CPU supplémentaire ; la latence iRig n'est **pas mesurée** et une promotion est interdite avant test physique.

## Audits

| Audit | Statut | Preuve |
| --- | --- | --- |
| Données | UNKNOWN | preuves S800 FIT80 à vérifier |
| Acoustique | UNKNOWN | 16 dimensions physiques et tests anti-futur |
| Architecture | UNKNOWN | cinq têtes gelées + nouveau head OOF |
| Apprentissage | UNKNOWN | exclusion de chaque composition |
| Métriques | UNKNOWN | 6 métriques MIDI256, 12 compteurs |
| LIVE | UNKNOWN | zéro hold/lookahead, matériel non mesuré |

Trois avis NVIDIA externes : UNKNOWN en l'absence de workflow_dispatch explicite. Consensus 9/9 : UNKNOWN, non PASS.

## Verdict

Aucun résultat ni gain S806 revendiqué. Le candidat ne peut passer le veto qu'avec zéro régression sur le global et les cinq compositions. S708 est protégée.

## Prochaine action

Exécuter les tests rapides, puis un seul run FIT80 sur GuitarSet réel, publier les métriques et les six audits. Refuser les variantes qui régressent, conserver S708 et documenter les caractéristiques acoustiques insuffisantes sans jamais prétendre à un spectre complet non présent.
