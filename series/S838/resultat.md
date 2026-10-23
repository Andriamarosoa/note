---
serie: S838
statut: en_cours
type: experience
reference: S708
branche: codex/note-s838-fragment-risk-gates256
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S838 — Résultat provisoire de la sélection apprise fragment PCM256

## Exécution
Nouvelle architecture de recherche créée avec deux têtes K indépendantes et une sélection *PAR FRAGMENT*, à la différence de la politique fixe par composition de S837. Les poids S708 et le MIDI officiel restent intacts. Tests de causalité et benchmark sur cinq vrais WAV à exécuter ; aucune nouvelle mesure scientifique S838 disponible avant la CI.

## Métriques
Scores d'origine S834 sur FIT80 80 WAV (non unifiés) :
| Caractéristique | SHADOW FIT80 | Objectif |
|---|---:|---:|
| Exact-K | 75,780659 % accuracy | >80 % |
| Active Exact-K | 65,193596 % accuracy | >80 % |
| Onset F1 | 71,382285 % | >80 % |
| Offset F1 événement | 38,677437 % | >80 % |
| Frame Pitch F1 | **84,998850 %** | préserver >80 % |
| Full Note F1 | 71,108037 % | >80 % |

Scores K de la dernière intervention S837 sur cinq vrais WAV : Exact-K 56,5995 % → 56,2706 % (**−66** blocs corrects) ; Active-K 46,7725 % → 45,4503 % (**−254** blocs corrects), rejetés. L'architecture S838 essaiera d'accepter localement les corrections bénéfiques sans les grosses régressions BN1/SS1 ; aucune affirmation de réussite avant audit OUTER.

## Comparaison à S708
La référence S708 CAL32 et H0 FIT80 sont figées. Cette expérimentation maintient la source S708 originale à 1024 utilisée en inférence native 256, sans mettre en vente les scores. Une tête K n'a pas le droit d'émettre de NoteOn/NoteOff ni d'accéder au sélecteur Onset/Offset. `Offset` reste une **métrique F1 distincte de l'accuracy Offset** H0 et doit être évalué séparément.

## Audits
| Rôle | État | Preuve |
|---|---|---|
| Données | UNKNOWN | WAV GuitarSet originaux à vérifier |
| Acoustique | UNKNOWN | 256 strict et 30 dimensions à tester |
| Architecture | UNKNOWN | banque et sélecteur privés |
| Apprentissage | UNKNOWN | compos TRAIN/CAL/OUTER disjointes à valider |
| Métriques | UNKNOWN | changements par fragment et vrai K0–K6 à produire |
| LIVE | UNKNOWN | aucun test Windows/iRig ni promotion |

## Verdict
**NON PROMU.** La nouvelle tête de sélection apprise répond à une hypothèse explicite mais n'a pas encore une preuve scientifique de gain ni un Live combiné. Les six caractéristiques ne sont pas encore toutes au-dessus de 80 %. Tout résultat inférieur ou régressif devra générer un audit causal sur les fragments perdus/gagnés et le type de polyphonie.

## Prochaine action
Exécuter les tests et les cinq WAV signés, mesurer si les décisions régressives S837 sont effectivement bloquées sans perdre les petites corrections prouvées, puis décider de l'extension 80 WAV. Répéter sur les têtes Onset et Offset avec banques/états privés et audit de vrais événements seulement après compréhension de leurs propres pertes.
