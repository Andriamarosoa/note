---
serie: S807
statut: en_cours
type: experience
reference: S708
branche: codex/note-s807-blind-activek-three-model-review
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S807 — diagnostic et critique des modèles indépendants

## Exécution

Modèles non encore interrogés au moment de création de la série. Le workflow est déclenché seulement avec un commit explicite `[audit-activek]` après test smoke, afin d'éviter la répétition des run S804. La même preuve FIT80 signée sera remise à trois modèles, chacun isolé des autres.

## Métriques

Six scores officiels S708 CAL32 restent inchangés (ce ne sont PAS des nouvelles mesures S807).

| Mesure officielle S708 CAL32 | Score | S807 proposé |
| --- | ---: | --- |
| Onset F1 | 74,2639 % | NON MESURÉ |
| Exact-K | 62,0455 % | NON MESURÉ |
| Active Exact-K | 56,2762 % | NON MESURÉ |
| Offset accuracy | 77,3292 % | NON MESURÉ |
| Full Note F1 | 57,1319 % | NON MESURÉ |
| Frame Pitch F1 | 76,4626 % | NON MESURÉ |

Le témoin diagnostique FIT80 : 249 411 trames actives, 142 213 correctes, 80 654 sous-comptées, 26 544 surcomptées. Sources et contextes FIT distincts de CAL32.

## Comparaison à S708

Aucune mutation de code musical. Zéro évaluation de nouvelles sorties MIDI, aucune preuve de non-régression, aucun test physique de latence. Pas de promotion.

## Audits

| Audit | Statut | Preuve |
| --- | --- | --- |
| Données | UNKNOWN | provenance signée FIT80 et compteur source |
| Acoustique | UNKNOWN | extraits code, aucune donnée future |
| Architecture | UNKNOWN | cause hypothétique puis tests falsifiables |
| Apprentissage | UNKNOWN | composition cible non utilisée dans apprentissage |
| Métriques | UNKNOWN | six métriques obligatoires |
| LIVE | UNKNOWN | aucune attente autorisée, micro non testé |

Trois consultations NVIDIA : PENDING / UNKNOWN. Ne jamais transformer des avis en consensus scientifique sans réfutation et reproduction.

## Verdict

Les modèles sont sollicités pour une critique ouverte, pas pour rendre vrai un diagnostic préexistant. La référence S708 reste inchangée.

## Prochaine action

Lire les trois réponses authentiques, cataloguer chaque assertion par source, corriger les interprétations erronées, choisir des tests adversariaux et exécuter les hypothèses classées par force de preuve et coût de latence sur les véritables trames FIT80.
