---
serie: S805
statut: en_cours
type: experience
reference: S708
branche: codex/note-s805-fit-crossfold-onset-risk
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S805 — résultats FIT80 et audits indépendants

## Exécution

Expérience initiale en attente du run GitHub. Base GitHub S804 `e020a00c0a6ee973e1a3be84ac9a3f205c102aa4`; sources vérifiées S800 [run 39352880035](https://github.com/Andriamarosoa/note/actions/runs/39352880035), S802 [run 39359182007](https://github.com/Andriamarosoa/note/actions/runs/39359182007) et S803 [run 39363062723](https://github.com/Andriamarosoa/note/actions/runs/39363062723). Run S805 : UNKNOWN jusqu'au replay authentique.

## Métriques

| Caractéristique | S708 officiel CAL32 | S805 FIT80 |
| --- | ---: | --- |
| Onset F1 | 74,2639 % | NON MESURÉ |
| Exact-K | 62,0455 % | NON MESURÉ |
| Active Exact-K | 56,2762 % | NON MESURÉ |
| Offset accuracy | 77,3292 % | NON MESURÉ |
| Full Note F1 | 57,1319 % | NON MESURÉ |
| Frame Pitch F1 | 76,4626 % | NON MESURÉ |

Témoin FIT80 figé S803 : Onset F1 73,2520 %, Exact-K 61,6972 %, Active Exact-K 57,0195 %, Offset 69,8862 %, Full Note F1 50,6528 %, Frame Pitch F1 78,7278 %. CAL32 et FIT80 ne sont **jamais interchangeables**.

## Comparaison à S708

Politique `spectral_balance` : UNKNOWN. Politique `risk_acoustic` : UNKNOWN. Politique `risk_joint` : UNKNOWN. Interdiction de promouvoir une variante qui régresse sur une métrique globale ou par composition ; aucune recherche de seuil sur FIT après vision des résultats. Les gains FIT, même significatifs, ne suffisent pas sans CAL32 indépendant et latence matérielle garantie.

## Audits

| Audit | Statut | Preuve |
| --- | --- | --- |
| Données | UNKNOWN | futur artefact FIT80 signé |
| Acoustique | UNKNOWN | futur artefact de rétroanalyse |
| Architecture | UNKNOWN | contrat des têtes gelées |
| Entraînement | UNKNOWN | exclusion de la composition cible |
| Métriques | UNKNOWN | 6 scores x 5 compositions et 12 compteurs |
| LIVE | UNKNOWN | zéro hold/lookahead additionnel |

Les trois conseillers NVIDIA (Gemma code, Nemotron données, Poolside red-team) restent PENDING et ne sont jamais simulés en l'absence de workflow manuel. Le consensus 9/9 est donc UNKNOWN tant que leurs avis et les six audits ne sont pas réellement reçus.

## Anomalies et mesures

Le filtre naïf de S802 rejetait trop de vraies notes (32 vrais contre 29 faux rejetés sur 136 attaques). C'est uniquement un constat rétrospectif, **pas** un gain Onset F1. S805 étudie une tête de risque hors-fold pour distinguer le caractère acoustique et les accords des vrais/faux NoteOn.

## Verdict

S708 reste officielle ; aucun déploiement LIVE ni nouveau score annoncé tant que les résultats réels et leurs audits ne sont pas disponibles.

## Prochaine action

Vérifier les 80 pistes S805, comparer les six métriques globales et les cinq compositions, puis documenter tous les échecs et PASS avant toute future discussion de promotion. Aucun ajout de latence LIVE n'est permis.
