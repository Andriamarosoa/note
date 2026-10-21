---
serie: S814
statut: en_cours
type: experience
reference: S708
branche: codex/note-s814-activek-private-midi256
run_id: en_attente
commit_sha: en_attente
promotion: false
---
# S814 — Résultats provisoires Active Exact-K indépendant

## Exécution

Branche S814 créée depuis S813 en conservant les fichiers et modèles historiques sans changement. Entraînement scientifique sur le PCM GuitarSet prévu dans le workflow propre à S814; aucun score nouveau revendiqué avant une exécution vérifiée et un rapport signé.

## Métriques

| Caractéristique | Référence S708 officielle CAL32 | Enseignant S814 FIT80 |
|---|---:|---|
| Onset F1 | 74,2639 % | non modifié / non mesuré |
| Exact-K | 62,0455 % | non modifié / non mesuré |
| Active Exact-K | 56,2762 % | en attente de la vraie expérience MIDI256 |
| Offset accuracy | 77,3292 % | non modifié / non mesuré |
| Full Note F1 | 57,1319 % | non modifié / non mesuré |
| Frame Pitch F1 | 76,4626 % | S813 expérimental 77,7132 % FIT80, non promu |

Le H0 FIT80 à reproduire avant toute analyse Active-K est 57,01953803 % (S812 / S800, 142213 sur 249411 trames actives). Ne pas confondre le score S810 shadow à 1024 avec la métrique finale MIDI256.

## Comparaison à S708

Aucun événement MIDI S708/S707 modifié, pas de refonte du décodeur ni d'unification. Les 220 variables proviennent des 222 variables acoustiques originales recalculées à 256 par S813. Le résultat expérimental, qu'il soit supérieur ou inférieur à H0, demeure SHADOW et n'altère pas les cinq autres caractéristiques.

## Audits

| Axe | Statut | Limite |
|---|---|---|
| Données | UNKNOWN | Checksum de 80 fichiers et de toutes les annotations à vérifier à l'exécution |
| Acoustique | PASS | Entrée acoustique native 256 imposée par contrat de forme et de causalité |
| Architecture | PASS | Deux poids distincts et sélecteur privé, aucune entrée d'un enseignant voisin |
| Apprentissage | UNKNOWN | Cinq vrais plis à entraîner et enregistrer avant conclusion |
| Métriques | UNKNOWN | Comparaison bit à bit avec le dénominateur S704 historique obligatoire |
| LIVE | UNKNOWN | Pas d'essai microphone, latence matérielle et CPU total inconnus |

## Verdict

S814 est une nouvelle expérience indépendante sans promotion. L'objectif >85 % est non vérifié, et aucun score fictif ou de données synthétiques ne sera utilisé comme mesure GuitarSet.

## Prochaine action

Exécuter l'entraînement complet sur les 80 sources signées avec les cinq compositions, contrôler les scores par vrai K, sous/surcomptages et registres, produire les checkpoints et conserver le veto de non-régression lors de toute éventuelle intégration MIDI.
