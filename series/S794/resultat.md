---
serie: S794
statut: en_cours
type: experience
reference: S708
branche: codex/note-s794-activek-zero-delay-forensics
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S794 — Résultats Active Exact-K causal MIDI 256

## Exécution

GuitarSet FIT80/CAL32, S708/S707 signés ; expérience et audit en cours. Aucun nouvel événement MIDI ni décodeur de production modifiés.

## Métriques

| Métrique officielle S708 CAL32 | Référence | S794 |
| --- | ---: | --- |
| Onset F1 | 74,2639 % | source inchangée, audit en cours |
| Exact-K | 62,0455 % | source inchangée, audit en cours |
| Active Exact-K | 56,2762 % | source inchangée, audit en cours |
| Offset accuracy | 77,3292 % | source inchangée, audit en cours |
| Full Note F1 | 57,1319 % | source inchangée, audit en cours |
| Frame Pitch F1 | 76,4626 % | source inchangée, audit en cours |

Les mesures diagnostiques visent les sous-comptages, surcomptages, K1–K6 et registres, et ne sont pas des gains ou de nouveaux scores.

## Comparaison à S708

S794 ne modifie aucun événement source. Le rapport scientifique doit démontrer une parité stricte avec les compteurs de S708 et identifier les causes dominantes des erreurs Active Exact-K avant toute proposition d'entraînement.

## Audits

| Rôle | Statut | Preuve |
| --- | --- | --- |
| Données | UNKNOWN | En attente |
| Acoustique | UNKNOWN | En attente |
| Architecture | UNKNOWN | En attente |
| Apprentissage | UNKNOWN | En attente |
| Métriques | UNKNOWN | En attente |
| LIVE | UNKNOWN | En attente |

## Incident de reproductibilité — Run #39333999180

Le calcul scientifique a traité **80/80 FIT et 32/32 CAL**, mais le garde-fou de signature S708 a refusé la comparaison. GitHub [run #39333999180](https://github.com/Andriamarosoa/note/actions/runs/39333999180) : `failure`, causée précisément par `k_correct=78367` contre `78366` signé. Les autres petites différences comptables sont `active_k_correct: 56414 vs 56413`, `frame_pitch_tp: 140540 vs 140541`, `frame_pitch_fp: 22501 vs 22504`, `frame_pitch_fn: 64022 vs 64021`. Aucun changement au modèle LIVE n'est à l'origine de cet échec, mais la **parité stricte n'est pas établie**. Onset et Offset sont inchangés. Ces données demeurent diagnostiques et ne peuvent autoriser aucune promotion.

L'audit a néanmoins calculé sur CAL32 les erreurs suivantes, sur les événements émis :

| Nombre réel de voix | Trames | K correct | Sous-comptages | Surcomptages |
| --- | ---: | ---: | ---: | ---: |
| K1 | 52 950 | 41 379 | 8 450 | 3 121 |
| K2 | 14 903 | 5 318 | 8 165 | 1 420 |
| K3 | 11 880 | 3 909 | 6 380 | 1 591 |
| K4 | 16 773 | 5 480 | 10 572 | 721 |
| K5 | 2 971 | 279 | 2 606 | 86 |
| K6+ | 766 | 49 | 717 | 0 |

Total trames actives sous-comptées : **36 890** ; surcomptées : **6 939**. Les fréquences élevées de sous-comptage ne constituent pas de nouveaux scores de modèle : elles motivent la recherche de spécialistes de voix manquantes.

Une deuxième exécution de reproductibilité (CAL32 deux fois avec hachage de chaque flux MIDI) est lancée dans [run #39335625292](https://github.com/Andriamarosoa/note/actions/runs/39335625292). Ne pas masquer l'écart par une tolérance sur la parité.

## Investigation indépendante de la reproductibilité

- [Rejeu CAL32 deux fois #39335625292](https://github.com/Andriamarosoa/note/actions/runs/39335625292) : **SUCCESS**. Les **32/32 fichiers donnent des événements MIDI strictement identiques d'un rejeu à l'autre** (`0` enregistrement divergent). Les deux répétitions présentent exactement le même écart de **+1 `k_correct`** avec le rapport signé de S708. Le problème n'est donc pas une variation aléatoire **entre ces deux répétitions**.
- [Comparaison ancienne voie S708 et wrapper moderne #39336494854](https://github.com/Andriamarosoa/note/actions/runs/39336494854) : **SUCCESS**. Les deux voies produisent les **mêmes événements sur 32/32 fichiers**. Toutes deux affichent les mêmes différences avec les compteurs signés, donc **le wrapper S733 n'est pas la cause**.
- Le score officiel S708 reste celui qui figure dans le rapport original. Les micro-différences entre le rapport historique et sa reconstruction exacte ne peuvent ni être masquées par une tolérance de parité, ni être présentées comme une amélioration de l'Active Exact-K.
- La divergence demeure **une question de reproductibilité inter-exécutions** (environnement numérique, calculs sensibles aux seuils ou autre facteur encore indéterminé). Aucun mécanisme précis n'a été démontré.
- La chaîne S794 a été replacée sur les fonctions S708 d'origine, strictement inchangées, pour une provenance plus claire. Toute promotion reste bloquée par l'absence de parité stricte et de mesure matérielle LIVE.

## Verdict

Analyse en cours. Aucune architecture multi-têtes promue, aucun changement du MIDI final ni de la latence.

## Prochaine action

Auditer les erreurs K1–K6 et leurs causes au niveau des événements réels, créer le nombre nécessaire de spécialistes, les valider sur FIT par composition, puis sur CAL avec toutes les métriques et la barrière S793.
