---
serie: S806
statut: termine
type: experience
reference: S708
branche: codex/note-s806-physical-fundamental-contrast-oof
run_id: 39373511511
commit_sha: 42d5e36742c879096aa73ce40aa0cc2829220895
promotion: false
---

# S806 — rapport final réel GuitarSet FIT80

## Exécution

[Run scientifique GitHub #39373511511](https://github.com/Andriamarosoa/note/actions/runs/39373511511) terminé **SUCCESS**, SHA scientifique `42d5e36742c879096aa73ce40aa0cc2829220895`. 80/80 pistes GuitarSet FIT80 OOF contrôlées via SHA256 et décodées sur les événements MIDI256 originaux. 306 536 frames, 189 124 compteurs `k_correct`, 12 compteurs natifs retrouvés et les six métriques d'origine strictement identiques aux données S800/S803. Cinq modèles croisés hors-composition entraînés sur les seuls champs acoustiques d'origine S613 sauvegardés par S800 (16 variables dérivées). Aucun réentraînement des cinq têtes S795. Aucun audio brut ni harmonique complète ajoutée ; aucune étude CAL32, HELD ou TEST.

Les six rôles indépendants locaux et leur consensus six sont **PASS**, mais les trois avis NVIDIA restent **SKIPPED/UNKNOWN** : une consultation explicite `workflow_dispatch` est nécessaire. Il est interdit d'afficher 9/9 PASS sur ce run.

## Métriques

Scores tous exprimés en pourcentage, **même corpus FIT80**, même unique sortie MIDI256 par politique :

| Politique | Onset F1 | Exact-K | Active Exact-K | Offset accuracy | Full Note F1 | Frame Pitch F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S800/S803 témoin FIT | **73,2520 %** | **61,6972 %** | **57,0195 %** | **69,8862 %** | **50,6528 %** | 78,7278 % |
| fundamental_consensus | 73,0042 % | 61,6704 % | 56,9899 % | 69,4977 % | 50,2011 % | 78,7626 % |
| fundamental_adaptive | 72,9278 % | 61,6782 % | 56,9995 % | 69,4668 % | 50,1261 % | **78,7808 %** |
| fundamental_strict | 73,1099 % | 61,6835 % | 57,0059 % | 69,7002 % | 50,4200 % | 78,7463 % |

| Politique | Propositions supplémentaires bloquées | Attaques ajoutées | Non-régression |
| --- | ---: | ---: | --- |
| fundamental_consensus | 46 | 90 | FAIL |
| fundamental_adaptive | 9 | 109 | FAIL |
| fundamental_strict | 88 | 48 | FAIL |

Chaque variante baisse **cinq des six métriques** globales. Pour la variante `fundamental_strict`, Onset F1 −0,1421 point, Exact-K −0,0137 point, Active Exact-K −0,0136 point, Offset accuracy −0,1860 point, Full Note F1 −0,2328 point et Frame Pitch F1 +0,0185 point. Toutes les cinq compositions présentent une régression d'Onset ou d'Offset ; Jazz1 et SS1 perdent également Exact-K/Active Exact-K. **Candidats FIT éligibles : zéro.**

## Comparaison à S708

| Métrique | S708 référence officielle CAL32 |
| --- | ---: |
| Onset F1 | 74,2639 % |
| Exact-K | 62,0455 % |
| Active Exact-K | 56,2762 % |
| Offset accuracy | 77,3292 % |
| Full Note F1 | 57,1319 % |
| Frame Pitch F1 | 76,4626 % |

**Ne pas comparer directement CAL32 à FIT80.** Aucune nouvelle variante promue : S708 officielle inchangée, tous les MIDI natifs conservés. S806 reste un essai SHADOW. L'absence de nouveau hold ou lookahead est vérifiée dans le décodeur, mais un CPU plus coûteux ou une latence matérielle supplémentaire ne sont pas exclus sans test iRig.

## Audits

| Audit | Statut | Preuve |
| --- | --- | --- |
| Données | PASS | 39373511511 — S800 signé 80/80, parité de 12 compteurs |
| Acoustique | PASS | 39373511511 — 16 variables physiques, pas de spectre complet prétendu |
| Architecture | PASS | 39373511511 — 5 têtes historiques gelées, 1 spécialiste acoustique croisé |
| Apprentissage | PASS | 39373511511 — 4 compositions d'entraînement pour chacune des 5 évaluées |
| Métriques | PASS | 39373511511 — 6 ratios, 5 compositions, veto des régressions |
| LIVE | PASS | 39373511511 — 0 maintien/anticipation additionnel, latence hardware NON MESURÉE |

Consensus déterministe local 6/6 PASS. Trois audits NVIDIA (Gemma, Nemotron, Poolside) NON EXÉCUTÉS, consensus 9/9 UNKNOWN et **jamais simulé**. Les six agents Work personnalisés ne sont pas les six scripts CI.

## Verdict

**S806 terminé, aucune amélioration musicale acceptable.** Trois politiques physiques croisées ont été réellement évaluées puis rejetées pour régressions. Les 16 variables S806 utilisent des fondamentales partielles déjà condensées, non les harmoniques brutes du WAV. Une nouvelle famille de représentation devra lire l'audio réel et calculer une séparation spectrale/polysource causale avec vérification de la latence CPU/entrée et des six métriques. Pas de promotion, S708 reste protégée.

## Prochaine action

Préparer une expérience indépendante avec le WAV GuitarSet authentique et des mesures **nouvelles** : fondamentale directe, partiels harmoniques jusqu'au 8e, canal spectral résiduel/contraste et différences pré-attaque calculées uniquement à la trame reçue. Interdire tout audio post-attaque futur, offset prolongé, latence ajoutée, fuite FIT vers CAL. Geler préalablement le protocole et contrôler coûts de décodage, tous les six scores, les cinq compositions, l'absence de régression et l'audit NVIDIA 9/9 réellement déclenché avant toute promotion.
