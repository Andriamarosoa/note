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

Suite de S827 : premier pilote SCIENTIFIQUE et contrôle d'isolation à lancer. Aucun résultat de prédiction S828 ne doit être déclaré avant vérification d'un run GitHub. Les cinq autres caractéristiques et la production LIVE restent strictement inchangées.

## Métriques

| Axe officiel S708 CAL32 | Score conservé | S828 |
|---|---:|---|
| Onset F1 | 74,2639 % | Gelé, non modifié |
| Exact-K | 62,0455 % | Gelé, non modifié |
| **Active Exact-K** | **56,2762 %** | Enseignant privé natif256, résultat en attente |
| Offset accuracy | 77,3292 % | Gelé, non modifié |
| Full Note F1 | 57,1319 % | Gelé, non modifié |
| Frame Pitch F1 | 76,4626 % | Gelé, non modifié |

## Comparaison à S708

H0 officiel est inchangé. GuitarSet FIT80 H0 142213/249411 = 57,019538%; S827 256 acoustique vraie 61,1665% sur FIT80 lorsque recalculé dans S826 entier, mais S827 privé certifié jusque-là sur 5 pistes à 39,593961%. La sortie S828 entraînée à 256 ne constitue PAS un réentraînement des couches partagées S708. Elle doit être comparée au H0 et au S827 sur la même cohorte à 256, sans régression des autres axes (invariance bit-exact par construction).

## Audits

| Domaine | Statut | Preuve nécessaire |
|---|---|---|
| Données | UNKNOWN | WAV d'origine signé SHA + splits stricts de compositions |
| Acoustique | UNKNOWN | S708 288D authentiques à chaque 256 |
| Architecture | UNKNOWN | Diff uniquement S828, cinq modèles gelés |
| Apprentissage | UNKNOWN | K readout privé entraîné à 256 et sélectionné sur CAL disjointe |
| Métriques | UNKNOWN | Trames actives K>0, classes K1–K6, comparaison H0 S827 |
| LIVE | UNKNOWN | Aucun déploiement et pas de p99 mesuré |

## Verdict

AUCUNE PROMOTION avant confirmation technique et musicale. Ne pas confondre réussite du code, amélioration scientifique sur FIT80 et déploiement strict LIVE. Même en présence d'un gain ActiveK, la tête privée ne peut modifier les NoteOn/Off ou les métriques de ses cinq voisins.

## Prochaine action

Lancer les tests de provenance et les inférences réelles d'ActiveK, puis lire et enregistrer les scores GuitarSet authentiques. Comparer toutes les sélections pré-enregistrées sur CAL uniquement, identifier les erreurs restantes K1–K6 et retenir uniquement les variantes gagnantes non régressives. Ne pas former de moteur MIDI commun avant 85% et preuve matérielle.
