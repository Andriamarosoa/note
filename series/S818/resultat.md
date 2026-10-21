---
serie: S818
statut: en_cours
type: experience
reference: S708
branche: codex/note-s818-activek-six-ordinal-heads
run_id: en_attente
commit_sha: en_attente
promotion: false
---
# S818 — Résultats de l'architecture ordinale Active-K

## Exécution

Expérience de recherche en préparation sur sources signées FIT80 de S800 et extracteur causal MIDI256 de S813. L'entraînement réel, les cinq plis composition-exclus et les signatures de checkpoints seront contrôlés dans GitHub Actions. Aucun nombre dérivé de tests unitaires n'est interprété comme un score GuitarSet.

## Métriques

| Caractéristique | S708 CAL32 officiel | S818 FIT80 |
|---|---:|---|
| Onset F1 | 74,2639 % | inchangé |
| Exact-K | 62,0455 % | inchangé |
| Active Exact-K | 56,2762 % | non mesuré |
| Offset accuracy | 77,3292 % | inchangé |
| Full Note F1 | 57,1319 % | inchangé |
| Frame Pitch F1 | 76,4626 % | inchangé |

La comparaison FIT80 équitable est H0 142213/249411 = 57,01953803 %, S815 et S817 141946/249411 = 56,91248582 %. Il ne faut pas comparer directement le pourcentage CAL32 de S708 avec celui d'un candidat FIT80.

## Comparaison à S708

Aucune tête S708, aucun modèle S815, aucune sortie S707 MIDI historique, aucun runtime LIVE ne sont modifiés. S818 est uniquement une autre tête Active-K SHADOW, entraînée séparément pour apprendre les seuils cumulatifs K>=j. Six métriques réelles sur un seul MIDI joint ne sont pas mesurées.

## Audits

| Axe | Statut | Commentaire |
|---|---|---|
| Données | UNKNOWN | Exiger manifestes SHA, 80 pistes sources |
| Acoustique | PASS | Contrat causal acoustique 256 échantillons sans futur |
| Architecture | PASS | Six objets tête privés, sélecteur interne et aucune dépendance inter-caractéristique |
| Apprentissage | UNKNOWN | Cinq jeux de six modèles réellement entraînés requis |
| Métriques | UNKNOWN | Parité S704 sur 249411 trames actives et ventilation K1–K6 requises |
| LIVE | UNKNOWN | Pas de benchmark iRig, pas de preuve de latence bout-en-bout |

## Verdict

Aucun score musical S818 encore validé. Promotion et unification interdites avant succès strict >85 % de chaque enseignant sur cohorte indépendante avec tests de veto des six sorties MIDI et latence matérielle réelle.

## Prochaine action

Lancer tests structurels, exécuter les cinq plis réels, vérifier le comptage des sous/sur-détections par K, puis rejeter tout candidat régressif, sans ajouter de délai temporel.
