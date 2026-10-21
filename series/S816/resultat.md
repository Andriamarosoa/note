---
serie: S816
statut: en_cours
type: experience
reference: S708
branche: codex/note-s816-activek-risk-selector
run_id: en_attente
commit_sha: en_attente
promotion: false
---
# S816 — Résultat du sélecteur privé anti-surcomptage

## Exécution

Expérience de resélection sans réentraînement sur les poids immuables S815 et les WAV GuitarSet mono 44,1 kHz signés. Exécution GitHub et métriques finales en attente.

## Métriques

| Caractéristique | S708 officielle CAL32 | S816 shadow FIT80 |
|---|---:|---|
| Onset F1 | 74,2639 % | inchangée, non recalculée |
| Exact-K | 62,0455 % | inchangée, non recalculée |
| Active Exact-K | 56,2762 % | en attente |
| Offset accuracy | 77,3292 % | inchangée, non recalculée |
| Full Note F1 | 57,1319 % | inchangée, non recalculée |
| Frame Pitch F1 | 76,4626 % | inchangée, non recalculée |

Référence comparable exclusivement sur FIT80 : H0 57,019538 %, S815 56,912486 %. Deux valeurs issues de l'exécution [S815 #39536863317](https://github.com/Andriamarosoa/note/actions/runs/39536863317). La précision d'une tête Active-K ne correspond pas au score du MIDI joint final.

## Comparaison à S708

Aucun code ni événement de la sortie MIDI S708/S707 modifié. S816 ne remplace aucun détecteur LIVE, aucune modification des cinq autres axes ni de leur modèle. La règle de sélection agit sur la seule sortie SHADOW Active-K des trois poids gelés.

## Audits

| Axe | Statut | Preuve/limite |
|---|---|---|
| Données | UNKNOWN | Attendre preuve SHA PCM et manuscrit 80/80, partition authentique |
| Acoustique | PASS | Extraction native MIDI256 inchangée et schéma S813 requis |
| Architecture | PASS | Trois modèles S815 gelés, sélecteur propre à Active-K seulement |
| Apprentissage | PASS | Pas de réentraînement, choix uniquement sur métriques de calibration interne |
| Métriques | UNKNOWN | Exiger la reconstitution des 80 scores S815 et H0 puis le score S816 |
| LIVE | UNKNOWN | CPU complet/iRig et latence physique non mesurés |

## Verdict

En attente de scores authentifiés. Promotion=false même en cas de progrès FIT80 ; le seuil de 85 % sur validation indépendante et la validation MIDI final/hardware restent requis.

## Prochaine action

Exécuter le workflow scientifique S816, auditer sous/surcomptages K1–K6, documenter les éventuelles régressions et geler le candidat jusqu'à validation réelle.
