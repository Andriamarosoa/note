---
serie: S818
statut: termine
type: experience
reference: S708
branche: codex/note-s818-activek-six-ordinal-heads
run_id: 39546770781
commit_sha: 50430c40bad0a37b990f5943d3e55a71b15f7b38
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
| Active Exact-K | 56,2762 % | **54,40578002 % FIT80 (135694 / 249411)** |
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


## Clôture scientifique vérifiée — S818

Run GitHub Actions #39546770781 : https://github.com/Andriamarosoa/note/actions/runs/39546770781 — SUCCESS, 5 plis × 6 têtes ordinales privées entraînées, extraction MIDI256 de 80 pistes GuitarSet à 44100 Hz. Artefact des 30 checkpoints, décisions et résultats #12322741581 : https://github.com/Andriamarosoa/note/actions/runs/39546770781/artifacts/12322741581.

* Vrai S818 Active-K FIT80 : **135694/249411 = 54,40578002 %**, 73512 sous-comptages et 40205 surcomptages.
* S815 figé : **141946/249411 = 56,91248582 %** ; delta S818 **−2,50670580 points**.
* H0 FIT80 historique : **142213/249411 = 57,01953803 %** ; delta S818 **−2,61375801 points**.
* Par composition : BN1 27669/47770 = 57,9213 % ; Funk2 27246/48827 = 55,8011 % ; Jazz1 22959/40548 = 56,6218 % ; Rock3 28487/54917 = 51,8728 % ; SS1 29333/57349 = 51,1482 %.

**REJET SCIENTIFIQUE :** S818 a régressé malgré des poids réellement indépendants. H0, S708, le MIDI LIVE et les cinq autres caractéristiques ne sont pas modifiés. Promotion=false, unification=false, objectif 85 % non atteint, validation externe et latence matérielle non mesurées. FIT80 est une cohorte historiquement explorée, non un benchmark indépendant certifiant la généralisation. Un second run #39546788233 a été déclenché par un correctif mineur des chemins de compilation, à contrôler séparément.
