---
serie: S811
statut: en_cours
type: experience
reference: S708
branche: codex/note-s811-blind-nvidia-s810-error-attribution
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S811 — résultats de la consultation contradictoire indépendante

## Exécution

Run NVIDIA NIM à vérification : tentative indépendante de 3 modèles, un seul input SHA commun, trois fichiers de réponses distincts et contrôle des références. Ni CAL32 ni HELD/TEST ni aucun vrai enregistrement audio partagé. Aucun entraînement et aucune modification du décodeur MIDI officiel. Les réponses ne sont pas encore réputées valides ; source `S810_six_trained_independent_models_FIT80.zip` (programme original hors dépôt) et `series/S811/fit80_blind_evidence.md` pour les extraits publics.

## Métriques

Les métriques SHADOW de S810 à la cadence 1024 (FIT80) sont :

| Axe | S810 diagnostic individuel | Nature |
| --- | ---: | --- |
| Onset F1 | 11,3986 % | 10002 vrais, 154245 faux, 1247 manqués |
| Offset F1 | 12,7457 % | 10479 vrais, 142827 faux, 647 manqués |
| Début Full Note F1 | 14,8887 % | 8518 vrais, 94655 faux, 2731 manqués |
| Exact-K accuracy | 76,8042 % | 58832 / 76600 trames |
| Active Exact-K accuracy | 76,5656 % | 47708 / 62310 trames actives |
| Frame Pitch F1 | 78,7230 % | 132708 TP / 64190 FP / 7546 FN |

Aucun score officiel S704 du MIDI S810 final n'existe à cette étape. Référence S708 CAL32 figée : Onset F1 74,2639 % ; Exact-K 62,0455 % ; Active Exact-K 56,2762 % ; Offset accuracy 77,3292 % ; Full Note F1 57,1319 % ; Frame Pitch F1 76,4626 %. Les deux horloges et méthodologies ne sont pas interchangeables.

## Comparaison à S708

S708 MIDI non modifié. Les scores officialisés sont inchangés ; aucun verdict de « baisse » sur les autres axes du moteur courant n'est autorisé à partir des F1 SHADOW. Toute modification de production attendrait six mesures finales sur un même flux MIDI et une preuve de latence non régressive.

## Audits

| Audit | Statut | Preuve |
| --- | --- | --- |
| Données | UNKNOWN | empreintes de 80 traces S810 et contexte borné |
| Acoustique | UNKNOWN | pas d'audio brut partagé |
| Architecture | UNKNOWN | aucun décodeur fusionné testé |
| Apprentissage | UNKNOWN | causes non encore réfutées |
| Métriques | UNKNOWN | différences des métriques à expliciter |
| LIVE | UNKNOWN | aucune modification ni test matériel |

## Verdict

État initial : aucune conclusion de modèle NVIDIA considérée scientifiquement validée. Même si trois réponses sont reçues, vérifier leurs références ligne à ligne et exécuter les ablations indépendantes. Aucune promotion.

## Prochaine action

Comparer les hypothèses réellement proposées par Gemma, Nemotron et Kimi, rejeter leurs erreurs de lecture du code, puis sélectionner des contre-expériences mesurables sur les vraies 80 traces FIT (précision/rappel, seuil de décision, exemples, fréquence par composition) sans entraîner sur le fold évalué ni utiliser CAL32.
