---
serie: S828
statut: en_cours
type: experience
reference: S708
branche: codex/note-s828-activek-s708-private-readout-256
run_id: en_attente
commit_sha: en_attente
promotion: false
---
# S828 — évaluation de la tête Active-K privée à partir de S708 256

## Exécution

**PILOTE SCIENTIFIQUE TERMINÉ AVEC SUCCÈS : [GitHub Actions #39603907835](https://github.com/Andriamarosoa/note/actions/runs/39603907835).** Cinq WAV GuitarSet originaux signés SHA, une piste par composition, cinq plis composition-disjoints avec TRAIN/CAL/TEST, quatre audits de non-interférence. Une extension FIT80 entière a été lancée mais son résultat n'est pas encore connu. Aucun résultat de prédiction S828 ne doit être déclaré avant vérification d'un run GitHub. Les cinq autres caractéristiques et la production LIVE restent strictement inchangées.

## Métriques

| Axe officiel S708 CAL32 | Score conservé | S828 |
|---|---:|---|
| Onset F1 | 74,2639 % | Gelé, non modifié |
| Exact-K | 62,0455 % | Gelé, non modifié |
| **Active Exact-K** | **56,2762 %** | 44,414368 % pilote 5 pistes (8 532/19 210), FIT80 complet NON MESURÉ |
| Offset accuracy | 77,3292 % | Gelé, non modifié |
| Full Note F1 | 57,1319 % | Gelé, non modifié |
| Frame Pitch F1 | 76,4626 % | Gelé, non modifié |


### Scores du pilote 5 pistes — même cohorte et mêmes 19 210 trames actives

| Système | Active Exact-K | Correct/GT-actives |
|---|---:|---:|
| H0 S800 source historique | 33,446122 % | 6 425/19 210 |
| S827 S708 gelé privé à 256 | 39,593961 % | 7 606/19 210 |
| **S828 S708 gelé + nouveau readout Active-K entraîné à 256** | **44,414368 %** | **8 532/19 210** |

S828: 6 313 sous-comptages, 4 365 surcomptages; S827: 9 160 sous-comptages, 2 444 surcomptages. **Attention : augmentation nette des surcomptages malgré le gain d'exactitude (+4,82 points vs S827).** L'entraînement n'a modifié que la tête de comptage privée; aucune autre tête ni événement MIDI S708 signé. Le S708 original n'a pas été affiné. Ce pilote est petit et FIT historiquement exploré : aucun claim de performance généralisée et **objectif 85 % non atteint**.

## Comparaison à S708

H0 officiel est inchangé. GuitarSet FIT80 H0 142213/249411 = 57,019538%; S827 256 acoustique vraie 61,1665% sur FIT80 lorsque recalculé dans S826 entier, mais S827 privé certifié jusque-là sur 5 pistes à 39,593961%. La sortie S828 entraînée à 256 ne constitue PAS un réentraînement des couches partagées S708. Elle doit être comparée au H0 et au S827 sur la même cohorte à 256, sans régression des autres axes (invariance bit-exact par construction).

## Audits

| Domaine | Statut | Preuve nécessaire |
|---|---|---|
| Données | PASS | Pilote 5 WAV originaux SHA, splits 3 compositions TRAIN / 1 CAL / 1 TEST |
| Acoustique | PASS | 288D S708 originales recalculées à chaque 256 dans les 5 WAV |
| Architecture | PASS | Seuls les nouveaux fichiers privés S828; 5 autres axes intacts |
| Apprentissage | PASS | Readout K privé réellement entraîné à 256; 5 sélecteurs choisis sur CAL disjointe, 4/5 plis classifieur direct et 1/5 témoin S708; 85 % non atteint |
| Métriques | PASS | Pilote 8 532/19 210 = 44,414368 %, contre S827 7 606 et H0 6 425 sur la même cohorte; FIT80 80 en cours |
| LIVE | UNKNOWN | Aucun déploiement et pas de p99 matériel mesuré |

## Verdict

AUCUNE PROMOTION avant confirmation technique et musicale. Ne pas confondre réussite du code, amélioration scientifique sur FIT80 et déploiement strict LIVE. Même en présence d'un gain ActiveK, la tête privée ne peut modifier les NoteOn/Off ou les métriques de ses cinq voisins.

## Prochaine action

Les tests du pilote sont terminés et la procédure complète FIT80 (80 pistes) est lancée. Vérifier les scores finaux uniquement après réussite du run GitHub; aucune promotion automatique. Comparer toutes les sélections pré-enregistrées sur CAL uniquement, identifier les erreurs restantes K1–K6 et retenir uniquement les variantes gagnantes non régressives. Ne pas former de moteur MIDI commun avant 85% et preuve matérielle.
