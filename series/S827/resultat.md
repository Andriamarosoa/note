---
serie: S827
statut: termine
type: experience
reference: S708
branche: codex/note-s827-isolated-s708-activek-midi256
run_id: 39601164866
commit_sha: 6c57840deaa56d561628440354302b3b3fecce54
promotion: false
---
# S827 — Résultat vérifiable pour Active Exact-K strict256 exclusivement

## Exécution

**Run réel terminé et réussi : [GitHub Actions #39601164866](https://github.com/Andriamarosoa/note/actions/runs/39601164866).** Tests de parité originelle S708, invariance bit-à-bit des cinq autres axes, et replay sur **5 enregistrements signés** GuitarSet à 256. Les cinq autres modèles, décodeurs et scores de référence sont **gelés**. La série S826 complète, avec ses régressions sur d'autres axes, est non promue.

## Métriques

| Axe | Référence CAL32 | Effet S827 |
|---|---:|---|
| Onset F1 | 74,2639 % | Gelé, signatures H0 inchangées et testées |
| Exact-K | 62,0455 % | Gelé, signatures invariantes à tester |
| **Active Exact-K** | **56,2762 %** | 39,593961 % pilote 5 pistes FIT80, aucune promotion |
| Offset accuracy | 77,3292 % | Gelé, signatures invariantes à tester |
| Full Note F1 | 57,1319 % | Gelé, signatures invariantes à tester |
| Frame Pitch F1 | 76,4626 % | Gelé, signatures invariantes à tester |

## Comparaison à S708

Sur FIT80, H0 S800 Active-K est **142213/249411 = 57,019538 %**. S826, en modifiant toute la chaîne, obtenait **61,1665 % FIT80** à cadence 256, tout en dégradant trois autres caractéristiques : **cette architecture partagée a été rejetée**. La valeur 76,877523 % de S823 utilise des variables 1024 S795 et de l'état MIDI, pas de nouveaux postérieurs audio S708 à 256.

## Résultats nouveaux S827

**Mesure pilote terminée : 7606/19210 = 39,593961 % pour Active Exact-K S827 contre 6425/19210 = 33,446122 % H0 signé sur les mêmes 5 enregistrements**. Cinq parités de compte Active-K S827 ≡ S826 vrai S708 à 256 vérifiées sur chaque trame. Une véritable inférence PyTorch S708 à chaque 256. Les cinq autres axes restent EXACTEMENT S812 H0 sans modification de fichiers ni des événements MIDI officiels. **Aucune promotion de la référence**. Les comparaisons porteront sur Active-K uniquement et sur la préservation des cinq signatures officielles, pas sur de nouveaux scores Onset/Offset/Exact-K causés par un décodeur partagé.

## Audits

| Domaine | Statut initial | Acceptation |
|---|---|---|
| Données | PASS | Cinq WAV originaux GuitarSet SHA et cinq métadonnées signées S800 contrôlées |
| Acoustique | PASS | Les 288D S708 reconstruites vraiment toutes les 256 et égales à S708 offline sur préfixes |
| Architecture | PASS | Fichiers de cinq axes intacts, sélecteur S708 d'origine et classe protégée S827 avec 5 sorties H0 identiques |
| Apprentissage | FAIL | Checkpoint original entraîné à 1024 et pas encore réentraîné à 256; cible 85 % non atteinte |
| Métriques | PASS | Pilote 5 pistes, 7606/19210 ActiveK S827 contre H0 6425/19210, comparateur S826 identique; aucune validation FIT80 80 |
| LIVE | UNKNOWN | SHADOW uniquement, aucun moteur micro LIVE modifié et aucune mesure p99/iRig |

## Verdict

**S827 fonctionne en SHADOW sans aucune régression des cinq autres architectures, mais reste NON PROMU et n'atteint pas l'objectif 85 %.** La PR S826 #130 (décodeur commun modifié) a été fermée sans fusion. Aucun candidat de S827 ne doit être promu sur une simple réussite CI. Seule une tête ActiveK réellement indépendante peut, après preuve musicale et budget de latence, être étudiée pour une intégration séparée.

## Prochaine action

La comparaison par trame S827 = S826 pour cinq WAV est validée et les cinq autres axes H0 sont identiques. La prochaine phase distincte doit **réentraîner réellement S708 pour Active-K en 256**, en gardant les cinq autres têtes gelées, puis refaire les tests de généralisation/calibration sur splits disjoints et timing réel, sans changement de MIDI officiel jusqu'à preuve. Pas de promotion sans score Active-K et latence vérifiés.
