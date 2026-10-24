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

# S839 — Contrat strict entraînement ET inférence PCM256

## Objectif
Respecter la règle exprimée depuis le départ : tout nouvel Onset prétendant être « 256 échantillons » doit être **réellement entraîné** avec entrée brute mono de 256 échantillons, pas de 256 au seul appel Live d'un modèle 1024. Le budget est 256/44100 = 5,80498866 ms ; zéro attente ajoutée et aucun accès à l'audio futur.

## Hypothèse
Dans S834→S838, une inférence à chaque 256 réutilise des poids S708 entraînés originellement à une cadence effective 1024. Par conséquent les 71,382285 % F1 Onset de S834 sont une **preuve de réévaluation à pas 256 d'un professeur 1024**, pas un vrai score d'un nouveau réseau entièrement entraîné sur PCM256. Même la banque sharp 8192 fait des analyses sur plus de 256 échantillons passés. Interdiction de qualifier cela de « tout 256 » au sens utilisateur.

## Modifications
- `src/causal_note/s839_native256_contract.py` : assertions bloquantes de taille `train_input_samples=256`, `train_hop_samples=256`, `inference_input_samples=256`, `inference_hop_samples=256`, fenêtre brute directe maximale 256, `lookahead_samples=0`, `added_wait_samples=0`, `pretraining_source_hop_samples=256` et aucune réutilisation d'un teacher pré-entraîné 1024.
- `provenance_from_observed_training` : données de provenance construites à partir des formes des **vrais minibatches** observés par le futur entraînement (N,1,256). Les simples métadonnées auto-déclarées ne constituent pas une preuve de training.
- `Exact256Stream` : garde de flux de blocs bruts 256 consécutifs, avec émission au front reçu, sans accumulation d'un second bloc.
- `test/test_s839_native256_contract.py` : tests négatifs explicites pour entraîner avec 1024, introduire un teacher 1024, une fenêtre d'analyse supérieure à 256, un décalage, une attente, une entrée audio 128/512/1024 ou une fausse métadonnée.
- Ancien moteur Live stable, les 6 têtes, les modèles S708/S834 et les résultats signés restent totalement intacts.

## Protocole
1. Entrées brutes de 256 mono 44,1 kHz à chaque décision et pendant **l'entraînement** ; aucune trame brute de 1024, aucune conversion d'un professeur 1024 présentée comme native.
2. Pas de lookahead, aucune file d'attente ajoutée pour confirmer Onset ; audit à l'horloge des 256 échantillons et temps de traitement CPU inférieur au budget.
3. Entraîner **Onset uniquement** avec ses poids propres à partir de l'audio GuitarSet, split strict TRAIN/CAL/OUTER et métrique TP/FP/FN à 80 ms de tolérance, sans importer de scores 1024.
4. Refaire l'évaluation sur les 80 pistes et au-delà, distinguer F1 de chaque protocole, refuser toute promotion régressive sur les cinq autres axes.
5. L'étape actuelle est la **sécurisation du contrat**. Elle n'entraîne encore aucun nouveau modèle et ne garantit pas que 256 seuls suffiront aux fondamentales graves ; cela doit être établi scientifiquement sans enfreindre la contrainte.

## Audits obligatoires
| Rôle | Contrat |
|---|---|
| Données | PCM mono 44 100 Hz, origine WAV et vérité immuables |
| Acoustique | 256 samples maximum par fenêtre directe ; aucune FFT de 8192 annoncée comme fenêtre 256 |
| Architecture | poids initialisés pour modèle spécifique Onset, pas réemploi S708 ; autres 5 axes immuables |
| Apprentissage | vérification des vrais minibatches N×1×256 et des étiquettes sur splits hors composition |
| Métriques | score natif strict **inconnu** jusqu'au vrai entraînement et test comparable |
| LIVE | pas d'ajout de latence, vérifications physiques Windows/iRig requises |

## Statut
En cours. Aucun nouveau score natif strict 256 obtenu à ce stade, aucune promotion, aucune prétention que le seul contrôle de métadonnées prouve l'entraînement. C'est le garde-fou pour ne pas répéter S834→S838 en les qualifiant improprement d'entraînement natif 256.
