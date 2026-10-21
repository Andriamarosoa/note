---
serie: S829
statut: en_cours
type: experience
reference: S708
branche: codex/note-s829-s809-exactk-native256
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S829 — résultats Exact-K natif256, SHADOW seulement

## Exécution
Branche de recherche créée depuis reference/active-exact-k-s828-midi256. Exécution scientifique FIT80 : NON MESURÉE. Aucun ID de run complet, de modèle entraîné audité, ni de mesure physique LIVE disponible à la rédaction. Les checkpoints originaux S810 à 1024 ne sont pas réutilisés tels quels.

## Métriques
L'axe ciblé est Exact-K K0..K6 sur toutes les trames et non Active Exact-K (seulement trames actives). Le score candidat S829, la comparaison aux K raw du S708 relancé à 256 et aux comptes H0 MIDI sur les mêmes 80 pistes seront renseignés uniquement par le rapport JSON vérifié. Vérifier les 306 536 trames complètes et les comptes exacts par K.

| Métrique | S708 officiel CAL32 | S829 nouveau MIDI natif256 |
|---|---:|---|
| Onset F1 | 74,2639 % | NON MESURÉ |
| Exact-K | 62,0455 % | NON MESURÉ (SHADOW) |
| Active Exact-K | 56,2762 % | NON MESURÉ |
| Offset accuracy | 77,3292 % | NON MESURÉ |
| Full Note F1 | 57,1319 % | NON MESURÉ |
| Frame Pitch F1 | 76,4626 % | NON MESURÉ |

Témoins historiques de protocoles différents : S809 Exact-K 84,24 % SHADOW à 1024 ; S810 76,80 % SHADOW à 1024 ; S828 Active Exact-K 73,831146 % FIT80 natif256. Ces chiffres ne remplacent pas un résultat Exact-K natif256.

## Comparaison à S708
Référence officielle S708 et six métriques CAL32 intactes. La sortie Exact-K privée n'est pas fusionnée aux événements MIDI ; les cinq axes voisins ne sont donc pas réévalués. Encodeur neural S708 gelé mais partagé comme source, l'indépendance de l'apprentissage de bout en bout n'est pas établie.

## Audits
| Rôle | Statut | Preuve manquante |
|---|---|---|
| Données | UNKNOWN | SHA et comptages 80 WAV + annotations |
| Acoustique | UNKNOWN | Recalcul PCM256 et mutation du futur |
| Architecture | UNKNOWN | Perturbation de readout, identité cinq axes |
| Apprentissage | UNKNOWN | Traces TRAIN/CAL/OUTER et entraînement |
| Métriques | UNKNOWN | K0–K6, 80 pistes, dénominateur et erreurs |
| LIVE | UNKNOWN | Budget CPU p99 et iRig |

Trois avis NVIDIA et six agents Work personnalisés non invoqués : aucun consensus 9/9 revendiqué.

## Verdict
EN COURS — NON PROMU. Aucun nouveau score revendiqué. Une expérience Exact-K natif256 isolée est spécifiée, sans garantie d'atteindre 84,24 % ni de respecter un budget CPU physique de 5,80 ms.

## Prochaine action
Exécuter la CI smoke et le replay signé FIT80 80/80 à 256. Contrôler K0, sous-comptages K2..K6, provenance S708, six audits, trois avis NVIDIA authentiques, et comparaisons sans fusion MIDI. Rejeter tout lookahead et hold.
