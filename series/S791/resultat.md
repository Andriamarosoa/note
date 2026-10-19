---
serie: S791
statut: termine
type: experience
reference: S708
branche: codex/note-s791-framepitch-complementary-experts
run_id: 39325760225
commit_sha: 6361a01a846f257c6c2039f243b1fe6bbb4bf536
promotion: false
---

# S791 — Résultats

## Exécution

- Calcul scientifique figé : [GitHub Actions #39280680273](https://github.com/Andriamarosoa/note/actions/runs/39280680273), SHA `ff96ffec8aa34b5dd55d92142413a4a8690c745c`.
- Le calcul FIT80/CAL32 a terminé avec un rapport `complete=true` : **80 enregistrements FIT, 32 CAL**. Le run global a échoué parce que chaque job indépendant manquait de NumPy, et non à cause d'un apprentissage interrompu.
- Correction auditeurs et replay **sans réentraîner** : [GitHub Actions #39325760225](https://github.com/Andriamarosoa/note/actions/runs/39325760225), SHA `6361a01a846f257c6c2039f243b1fe6bbb4bf536`, **6 audits PASS + consensus SUCCESS**. Artefact `S791_REPLAY_SIX_AUDITS_CONSENSUS`.
- S708 signé contrôlé : SHA256 checkpoint `c81d98dbce8d096b54d71dedd88f521162ea95b19d0c059ab13baf69d5cf0588`.
- FIT : BN1, Funk2, Jazz1, Rock3, SS1. CAL : BN2, Jazz2. HELD et TEST fermés.
- Pas de modification du modèle ni du pipeline LIVE.

## Métriques

Les six **métriques officielles du MIDI final (256 échantillons)** restent celles de S708. Les pourcentages expérimentaux ci-dessous proviennent uniquement de masques Frame Pitch **SHADOW 1024**, pas d'événements MIDI complets.

| Métrique officielle | S708 GuitarSet CAL32 | Nouvelle mesure officielle |
| --- | ---: | --- |
| Onset F1 | 74,2639 % | non mesuré |
| Exact-K | 62,0455 % | non mesuré |
| Active Exact-K | 56,2762 % | non mesuré |
| Offset | 77,3292 % | non mesuré |
| Full Note F1 | 57,1319 % | non mesuré |
| Frame Pitch F1 | 76,4626 % | non mesuré |

### Frame Pitch — Évaluation SHADOW 1024

| Variante | FIT OOF micro F1 | CAL32 F1 | Écart CAL vs source |
| --- | ---: | ---: | ---: |
| Source S708 sur grille 1024 | 79,9293 % | 77,8367 % | référence |
| S790 — tête spécialisée | 86,7532 % | **85,1177 %** | +7,28 pts |
| S791 — multi-têtes dynamiques | 83,3009 % | **82,2115 %** | +4,37 pts |

L'écart S791 vs S790 sur CAL est **−2,91 points**. Les légères différences de baseline CAL (77,8353 % S790 contre 77,8367 % S791) proviennent des deux runs distincts et imposent une vérification de parité exacte avant toute intégration.

### Expertises retenues sur FIT

| Source ou expert | FIT OOF micro F1 | Décision |
| --- | ---: | --- |
| H0 — source MIDI S708 (grille 1024) | 79,93 % | Référence |
| Généraliste | 80,99 % | Retenu |
| Graves | 80,48 % | Retenu |
| Harmoniques | 79,95 % | Écarté |
| Transitions temporelles | 82,16 % | Retenu |

Le sélecteur S791 a conservé **3 experts sur 4**, avec **8 combinaisons admissibles sur 16**. Ce sont des **modèles entraînés distinctement par région acoustique**, et non six copies du même modèle ou seize modèles individuels.

### Analyse fine CAL32

| Registre / texture | Source S791 (shadow) | S791 multi-têtes | S790 tête unique |
| --- | ---: | ---: | ---: |
| E2–B2 | 64,36 % | 74,44 % | 79,28 % |
| C3–B3 | 77,32 % | 82,05 % | 85,61 % |
| C4–E5 | 81,21 % | 84,42 % | 86,01 % |
| Accompagnement | 74,59 % | 80,82 % | 83,82 % |
| Solo | 85,29 % | 85,64 % | 88,27 % |

S791 par composition CAL : **BN2 81,39 %**, **Jazz2 83,05 %**. Les deux compositions dépassent 80 %, mais ce score est SHADOW uniquement ; S790 les surpasse toutes deux.

## Comparaison à S708

**S708 est la seule référence officielle.** S790 et S791 dépassent 80 % en tête isolée, mais aucune de ces expériences ne génère de nouveau flux MIDI à cadence 256 ni ne prouve la non-régression simultanée des cinq autres caractéristiques. S791 est inférieur au meilleur candidat expérimental S790 ; l'accroissement du nombre d'experts n'est donc pas une justification suffisante.

## Audits

Les six contrôles ont été **effectués en jobs distincts** sur les données figées de l'expérience S791, avec artefacts JSON et consensus :

| Rôle | Statut | Preuve |
| --- | --- | --- |
| Données | PASS | [Run #39325760225](https://github.com/Andriamarosoa/note/actions/runs/39325760225), `role_data.json` |
| Acoustique | PASS | [Run #39325760225](https://github.com/Andriamarosoa/note/actions/runs/39325760225), `role_acoustics.json` |
| Architecture | PASS | [Run #39325760225](https://github.com/Andriamarosoa/note/actions/runs/39325760225), `role_architecture.json` |
| Apprentissage | PASS | [Run #39325760225](https://github.com/Andriamarosoa/note/actions/runs/39325760225), `role_training.json` |
| Métriques | PASS | [Run #39325760225](https://github.com/Andriamarosoa/note/actions/runs/39325760225), `role_metrics.json` |
| LIVE | PASS | [Run #39325760225](https://github.com/Andriamarosoa/note/actions/runs/39325760225), `role_live.json` |

Le premier workflow #39280680273 est **FAIL** par défaut de dépendances d'audit. La correction est une vérification replay sur les mêmes rapports figés, **sans lancer un second calcul scientifique**. Le succès de l'audit valide la cohérence et le protocole, pas la supériorité de S791.

## Verdict

**S791 TERMINÉ, AUDITS 6/6 PASS, NON PROMU.** Les experts sont bien complémentaires selon leur contribution FIT, mais leur combinaison testée perd **2,91 points** face à S790 sur CAL32 SHADOW. Aucune prétention d'avoir dépassé 80 % sur le MIDI final. L'objectif reste la généralisation par erreurs et la validation de toutes les métriques sur des sorties MIDI effectivement émises.

## Prochaine action

Étudier pourquoi la combinaison des trois experts apporte moins qu'une tête S790. Tester une comparaison à parité stricte sur les mêmes échantillons/splits, puis une sélection par pitch réellement indépendante des décisions fragiles sur le grave et la polyphonie. Ne pas ouvrir TEST ni fusionner/ promouvoir avant une validation causale MIDI 256 et six audits actualisés.
