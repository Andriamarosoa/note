---
serie: S797
statut: en_cours
type: experience
reference: S708
branche: codex/note-s797-fragment-router-six-audit-loops
run_id: pending
commit_sha: pending
promotion: false
---

# S797 — cinq boucles de sélecteur par fragment, sans promotion

## Exécution

Cinq ablations S797 préenregistrées sont soumises au workflow GitHub Actions. Elles réutilisent les preuves du run S795 39340511467, sans réentraînement des têtes. Run S797 en attente ; aucune tâche de ce workflow n'a encore produit de verdict mesuré dans ce document initial.

## Métriques

Les six scores officiels restent ceux de S708 CAL32 : Onset F1 74,2639 %, Exact-K 62,0455 %, Active Exact-K 56,2762 %, Offset 77,3292 %, Full Note F1 57,1319 %, Frame Pitch F1 76,4626 %. Les performances musicales nouvelles S797 ne sont **pas mesurées**. Les compteurs de la fixture synthétique restent à obtenir du workflow.

## Comparaison à S708

Aucune comparaison réelle sur MIDI256 d'une sortie S797 contre le modèle signé S708 : les tests synthétiques ne prouvent pas la non-régression sur GuitarSet. Référence S708/S707 et pipeline LIVE inchangés, HELD et TEST fermés.

## Audits

| Rôle | Statut | Preuve |
|---|---|---|
| Données | UNKNOWN | En attente d'un rapport indépendant |
| Acoustique | UNKNOWN | En attente d'un rapport indépendant |
| Architecture | UNKNOWN | En attente d'un rapport indépendant |
| Apprentissage | UNKNOWN | En attente d'un rapport indépendant |
| Métriques | UNKNOWN | En attente d'un rapport indépendant |
| LIVE | UNKNOWN | En attente d'un rapport indépendant |

## Verdict

Recherche non promue ; cinq variantes de sélection par fragment évaluées uniquement sur cas synthétiques. Les six audits GitHub Actions devront être réellement exécutés. Les agents Work personnalisés ne sont pas appelés par ces jobs GitHub.

## Prochaine action

Contrôler les tests CI et tous les audits ; analyser sans réutiliser CAL pour la sélection. Une mesure FIT80 OOF à granularité fragment et les six métriques MIDI256 avec parité source stricte sont indispensables pour toute progression scientifiquement validée vers 80 %.
