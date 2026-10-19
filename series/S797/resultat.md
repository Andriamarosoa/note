---
serie: S797
statut: termine
type: experience
reference: S708
branche: codex/note-s797-fragment-router-six-audit-loops
run_id: 39344715462
commit_sha: 26d393c3536c628f6fa6e904d249ec6e04978055
promotion: false
---

# S797 — cinq boucles de sélecteur par fragment, sans promotion

## Exécution

Les cinq boucles S797 ont été exécutées sur une fixture MIDI synthétique, avec les preuves FIT80 S795 gelées réutilisées sans entraînement. [Run S797 #39344715462](https://github.com/Andriamarosoa/note/actions/runs/39344715462) : job initial PASS, 28 tests unitaires PASS, six jobs d'audit indépendants PASS et consensus fail-closed PASS. Commit scientifique `26d393c3536c628f6fa6e904d249ec6e04978055`. Le premier run 39344511685 échouait uniquement sur le chemin d'extraction d'artefact ; corrigé puis revérifié.

## Métriques

Les six scores officiels restent ceux de S708 CAL32 : Onset F1 74,2639 %, Exact-K 62,0455 %, Active Exact-K 56,2762 %, Offset 77,3292 %, Full Note F1 57,1319 %, Frame Pitch F1 76,4626 %. Les performances musicales nouvelles S797 sur GuitarSet **ne sont pas mesurées**. Fixture S797 : trois notes réellement attendues et une fausse harmonique. Identité : TP 0 / FP 0 / FN 3 ; consensus TP 3 / FP 1 / FN 0 ; veto harmonique TP 3 / FP 0 / FN 0 ; poly-gap TP 3 / FP 0 / FN 0 ; adaptatif TP 3 / FP 0 / FN 0. Ces TP/FP/FN sont des comptes synthétiques, absolument pas des taux de qualité GuitarSet ni des métriques de production.

## Comparaison à S708

Aucune comparaison réelle sur MIDI256 d'une sortie S797 contre le modèle signé S708 : les tests synthétiques ne prouvent pas la non-régression sur GuitarSet. Référence S708/S707 et pipeline LIVE inchangés, HELD et TEST fermés.

## Audits

| Rôle | Statut | Preuve |
|---|---|---|
| Données | PASS | [Run S797 #39344715462](https://github.com/Andriamarosoa/note/actions/runs/39344715462), artefact du rôle indépendant |
| Acoustique | PASS | [Run S797 #39344715462](https://github.com/Andriamarosoa/note/actions/runs/39344715462), artefact du rôle indépendant |
| Architecture | PASS | [Run S797 #39344715462](https://github.com/Andriamarosoa/note/actions/runs/39344715462), artefact du rôle indépendant |
| Apprentissage | PASS | [Run S797 #39344715462](https://github.com/Andriamarosoa/note/actions/runs/39344715462), artefact du rôle indépendant |
| Métriques | PASS | [Run S797 #39344715462](https://github.com/Andriamarosoa/note/actions/runs/39344715462), artefact du rôle indépendant |
| LIVE | PASS | [Run S797 #39344715462](https://github.com/Andriamarosoa/note/actions/runs/39344715462), artefact du rôle indépendant |

## Verdict

S797 terminée avec six audits indépendants PASS et consensus PASS. La sélection peut varier par fragment et hauteur ; le leurre harmonique artificiel est évité sans retarder l'horloge des événements natifs sur le test. Aucun bénéfice de transcription de guitare réelle ni aucune latence CPU réelle n'est démontré. S708 reste référence ; la branche S797 n'est pas intégrée dans LIVE. Les six agents Work personnalisés sont distincts de ces six rôles automatisés GitHub, et n'ont pas été exécutés.

## Prochaine action

Prochaine boucle scientifique : émettre et préserver des probabilités OOF par fragment et les événements MIDI256 sur FIT80 afin de mesurer la véritable qualité des cinq politiques, avec selection FIT uniquement. Exiger parité CAL32 signée exacte (bloquée par le drift +1 de S795), six métriques sans régression et mesures physiques répétées de latence. Ne pas relancer les mêmes entraînements si les caches probabilité/modèle ne sont pas disponibles ; instrumenter d'abord leur conservation.
