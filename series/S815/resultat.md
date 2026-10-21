---
serie: S815
statut: en_cours
type: experience
reference: S708
branche: codex/note-s815-activek-pitch-evidence
run_id: en_attente
commit_sha: en_attente
promotion: false
---
# S815 — Résultats indépendants Active-K

## Exécution

Nouvelle expérience en cours sur branche S815 depuis S814 échoué. Source S800 signée et extraction originale MIDI256 réutilisées, sans transfert de poids ni sélection à partir des données évaluées. Le workflow doit fournir toutes les preuves CPU, annotations et compteurs réels avant clôture.

## Métriques

| Caractéristique | Officiel S708 CAL32 | Nouveau S815 FIT80 |
|---|---:|---:|
| Onset F1 | 74,2639 % | non modifié |
| Exact-K | 62,0455 % | non modifié |
| Active Exact-K | 56,2762 % | en attente |
| Offset accuracy | 77,3292 % | non modifié |
| Full Note F1 | 57,1319 % | non modifié |
| Frame Pitch F1 | 76,4626 % | non modifié |

Comparaison FIT80 sur le même dénominateur 249411 : référence H0 57,01953803 %, candidat S814 54,67280914 %, proposition S815 pas encore évaluée. La présence d'un troisième expert par hauteur n'autorise pas de lui attribuer le score Frame Pitch indépendant.

## Comparaison à S708

Le flux S708/S707 reste entièrement protégé et les cinq autres caractéristiques ne sont pas modifiées. Seule l'architecture de l'enseignant Active Exact-K est modifiée en SHADOW. Aucune transcription MIDI finale ni métrique combinée inventée.

## Audits

| Axe | Statut | Preuve et limite |
|---|---|---|
| Données | UNKNOWN | Exiger les 80 WAV mono originaux et les signatures S800 |
| Acoustique | PASS | Contractuellement [T,49,222] réellement à MIDI256 |
| Architecture | PASS | Trois modèles à poids privés, aucun accès aux autres enseignants |
| Apprentissage | UNKNOWN | Cinq entraînements composition-disjoints à exécuter |
| Métriques | UNKNOWN | En attente d'un vrai score Active-K par K1–K6 et sur/sous-comptages |
| LIVE | UNKNOWN | Aucun benchmark matériel CPU complet / iRig / émission finale |

## Verdict

Pas de promotion avant résultat mesuré, six validations communes indépendantes et veto de latence matériel. L'amélioration est une hypothèse, pas un chiffre acquis.

## Prochaine action

Lancer les tests GitHub puis l'expérience authentique à 256 échantillons, publier les scores par composition et par vrai K, refuser tout candidat moins bon et conserver S708.
