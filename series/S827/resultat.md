---
serie: S827
statut: en_cours
type: experience
reference: S708
branche: codex/note-s827-isolated-s708-activek-midi256
run_id: en_attente
commit_sha: en_attente
promotion: false
---
# S827 — Résultat vérifiable pour Active Exact-K strict256 exclusivement

## Exécution

En cours : tests de parité originelle S708 et replay GitHub de GuitarSet original signé. Les cinq autres modèles, décodeurs et scores de référence sont **gelés**. La série S826 complète, avec ses régressions sur d'autres axes, est non promue.

## Six axes officiels conservés — S708 CAL32

| Axe | Référence CAL32 | Effet S827 |
|---|---:|---|
| Onset F1 | 74,2639 % | Gelé, signatures invariantes à tester |
| Exact-K | 62,0455 % | Gelé, signatures invariantes à tester |
| **Active Exact-K** | **56,2762 %** | Reproduction S708 native256 privée, métrique en attente |
| Offset accuracy | 77,3292 % | Gelé, signatures invariantes à tester |
| Full Note F1 | 57,1319 % | Gelé, signatures invariantes à tester |
| Frame Pitch F1 | 76,4626 % | Gelé, signatures invariantes à tester |

## Benchmark authentique à respecter

Sur FIT80, H0 S800 Active-K est **142213/249411 = 57,019538 %**. S826, en modifiant toute la chaîne, obtenait **61,1665 % FIT80** à cadence 256, tout en dégradant trois autres caractéristiques : **cette architecture partagée a été rejetée**. La valeur 76,877523 % de S823 utilise des variables 1024 S795 et de l'état MIDI, pas de nouveaux postérieurs audio S708 à 256.

## Résultats nouveaux S827

Non mesurés jusqu'au succès de la CI. **Aucune promotion de la référence**. Les comparaisons porteront sur Active-K uniquement et sur la préservation des cinq signatures officielles, pas sur de nouveaux scores Onset/Offset/Exact-K causés par un décodeur partagé.

## Audits

| Domaine | Statut initial | Acceptation |
|---|---|---|
| Données | UNKNOWN | GuitarSet authentique SHA, S800 signé, 256 |
| Acoustique | UNKNOWN | Les 288D S708 recalculés sur signal nouveau chaque 256, pas as-of |
| Architecture | UNKNOWN | Aucun patch d'un composant commun, 5 axes H0 identiques |
| Apprentissage | PARTIEL | checkpoint S708 d'origine 1024, pas encore entraîné à 256 |
| Métriques | UNKNOWN | ActiveK S704 GT K>0; non-régression des autres axes par identité stricte |
| LIVE | NON INTÉGRÉ | Pas de modèle validé/promu, pas de claim de latence matérielle |

## Verdict

Aucun candidat de S827 ne doit être promu sur une simple réussite CI. Seule une tête ActiveK réellement indépendante peut, après preuve musicale et budget de latence, être étudiée pour une intégration séparée.
