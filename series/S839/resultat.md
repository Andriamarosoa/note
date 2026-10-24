---
serie: S839
statut: en_cours
type: experience
reference: S834
branche: codex/note-s839-strict-trained-pcm256-contract
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S839 — Contrôle natif 256 de NOTE Onset

## Exécution
Contrat de conformité `s839_native256_contract.py`, tests de non-régression `test_s839_native256_contract.py` et workflow de contrôle en préparation. Aucun checkpoint Onset entièrement entraîné sous ce contrat n'a été produit par cette série.

## Métriques
| Mesure | Valeur connue | Définition |
|---|---:|---|
| Référence H0 S708 | 73,252019 % F1 | FIT80, teacher historique cadence 1024 |
| S834 Onset | 71,382285 % F1 | FIT80 80 WAV, réinférence d'un modèle entraîné 1024 tous les 256 |
| Nouveau modèle entraîné strictement PCM256 | **NON MESURÉ** | Aucune donnée pour affirmer un score |
| Objectif utilisateur | **>80 % F1** | 80 pistes, sans latence, autres têtes indépendantes |

Les chiffres H0/S834 ne sont pas de nouveaux scores du modèle **entraîné** en PCM256 ; ils restent dans les archives sans être effacés ni modifiés. Le score Live complet à six têtes n'est pas non plus certifié.

## Comparaison à S708
L'actuel teacher S708 conserve `original_teacher_training_hop=1024` et échoue au nouveau contrat `train_hop_samples=256`. Cette exclusion est intentionnelle, même si S708 est appelé à pas de 256. Les modèles et les autres caractéristiques restent intacts ; aucun héritage de poids non certifiés n'est toléré pour revendiquer un entraînement strict.

## Audits
| Rôle | État | Preuve requise |
|---|---|---|
| Données | UNKNOWN | checksum et splits du nouvel entraînement |
| Acoustique | PASS code | assertions fenêtre brute = 256, aucun futur |
| Architecture | PASS code | S708-1024 refusé ; autres architectures inchangées |
| Apprentissage | UNKNOWN | nouveaux vrais minibatches et poids à entraîner |
| Métriques | UNKNOWN | nouveau F1 calculé sur 80 pistes |
| LIVE | UNKNOWN | p99 CPU / périphérique et six axes identiques |

## Verdict
**NON PROMU.** Le garde-fou empêche de réétiqueter l'inférence 256 de S708 comme une architecture réellement entraînée 256. Il ne prouve ni une amélioration de F1 ni une réduction de latence.

## Prochaine action
Former une nouvelle tête Onset effectivement native PCM256 à partir de RAW WAV, mesurer sur toutes les mêmes pistes et conditions, auditer FN/FP par identité, puis décider d'une promotion seulement avec scores supérieurs à 80 %, CPU p95/p99 cible et absence de régression des cinq autres caractéristiques.
