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

# S797 — sélection de têtes causales par fragment

## Objectif

Décider les combinaisons de têtes séparément pour chaque paire fragment/hauteur et non plus avec un seul masque global. L'objectif futur reste 80 % pour six métriques officielles sans dégrader Onset, Exact-K, Active Exact-K, Offset, Full Note et Frame Pitch.

## Hypothèse

Les cinq têtes hétérogènes entraînées dans S795 détectent des régimes différents, tandis que le masque global provoquait de fortes régressions. Des règles contextuelles qui utilisent seulement la décision courante, l'activité native, les divergences Exact-K et la présence harmonique pourraient réduire les attaques fantômes.

## Modifications

Ajout du routeur S797 sur les prédictions S795 et cinq ablations préenregistrées : identité, consensus, veto harmonique, poly-gap et adaptatif. Les masques sont propres à chaque fragment et hauteur. Le décodeur S795 et le modèle de production S708/S707 ne sont pas modifiés.

## Protocole

Première boucle **uniquement synthétique et reproductible**, testant des cas avec trois vraies notes manquantes et un leurre harmonique. Les véritables scores FIT80/CAL32 pour le routeur ne sont PAS évalués : l'archive S795 ne contient pas de probabilités par fragment. Pas de nouvel entraînement coûteux ni d'ouverture HELD/TEST ; réutilisation du rapport d'échec S795 archivé. Aucune sélection scientifique de modèle sur ces données artificielles.

## Audits obligatoires

Six rôles indépendants : Données, Acoustique, Architecture, Apprentissage, Métriques, LIVE et consensus échec strict. Tester conservation intégrale des NoteOn et NoteOff sources et absence de buffers, holds et lookahead. La latence CPU physique réelle demeure non mesurée : toute promotion est interdite.

## Statut

Expérience S797 en cours : branche expérimentale uniquement, tests et runs CI en attente. Les cinq boucles sont conçues comme tests de programmation, pas comme démonstration d'un gain de score sur GuitarSet.
