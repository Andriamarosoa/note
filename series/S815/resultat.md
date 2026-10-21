---
serie: S815
statut: termine
type: experience
reference: S708
branche: codex/note-s815-activek-pitch-evidence
run_id: 39536863317
commit_sha: dafb9eba0e587ae48156c46859bcfad034bfbaad
promotion: false
---
# S815 — Résultats indépendants Active-K

## Exécution

Exécution véritable **réussie sur GitHub** : [S815 #39536863317](https://github.com/Andriamarosoa/note/actions/runs/39536863317), cinq plis, 80 morceaux GuitarSet FIT80, extraction acoustique fraîche MIDI256, trois poids privés par pli, parité référence historique. Artefact S815_ACTIVE_K_PRIVATE_MIDI256_FIT80_RESULTS, ID 12317827253.

## Métriques

| Caractéristique | Officiel S708 CAL32 | Nouveau S815 FIT80 |
|---|---:|---:|
| Onset F1 | 74,2639 % | non modifié |
| Exact-K | 62,0455 % | non modifié |
| Active Exact-K | 56,2762 % | **56,91248582 % FIT80**, sous H0 FIT80 57,01953803 % |
| Offset accuracy | 77,3292 % | non modifié |
| Full Note F1 | 57,1319 % | non modifié |
| Frame Pitch F1 | 76,4626 % | non modifié |

Sur le même dénominateur FIT80 249411 trames actives : S815 141946 correctes, 65530 sous-comptages et 41935 surcomptages ; H0 142213 correctes, 80654 sous-comptages, 26544 surcomptages. Écart −0,10705222 point vs H0 ; +2,23967668 points vs S814, mais non certifié hors FIT80. Par composition : BN1 57,9485 % ; Funk2 57,3085 % ; Jazz1 56,4023 % ; Rock3 58,1004 % ; SS1 54,9356 %.

## Comparaison à S708

Le flux S708/S707 reste entièrement protégé et les cinq autres caractéristiques ne sont pas modifiées. Seule l'architecture de l'enseignant Active Exact-K est modifiée en SHADOW. Aucune transcription MIDI finale ni métrique combinée inventée.

## Audits

| Axe | Statut | Preuve et limite |
|---|---|---|
| Données | PASS | SHA des 80 traces, manifeste et archive audio vérifiés dans le run scientifique |
| Acoustique | PASS | Contractuellement [T,49,222] réellement à MIDI256 |
| Architecture | PASS | Trois modèles à poids privés, aucun accès aux autres enseignants |
| Apprentissage | FAIL | 15 modèles réellement entraînés ; 56,9125 % < H0 57,0195 %, objectif >85 % échoué |
| Métriques | PASS | Vraie mesure MIDI256 active 141946/249411, parité H0 et compteurs S704 vérifiés |
| LIVE | UNKNOWN | Aucun benchmark matériel CPU complet / iRig / émission finale |

## Verdict

**REJET MUSICAL S815** : écart FIT80 −0,1071 point vs H0 malgré récupération par rapport à S814. Le GitHub SUCCESS confirme les calculs et pas une victoire de performance. Aucune modification de S708, aucun LIVE ou unification validé.

## Prochaine action

Faire un audit S815 par K et par politique sélectionnée depuis les décisions signées. Tester une variante privée de sélecteur à risque calibré sans accéder aux plis évalués, jamais sur les résultats S815 FIT80 pour choisir après coup.
