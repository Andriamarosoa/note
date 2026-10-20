---
serie: S799
statut: termine
type: infrastructure
reference: S708
branche: codex/note-s799-nvidia-kimi-k25-adviser
run_id: 39351032882
commit_sha: 428950c46e995754b42af3522ed90a0738bc6a5d
promotion: false
---

# S799 — résultat du véritable septième avis indépendant NVIDIA

## Exécution

[GitHub Actions run #39351032882](https://github.com/Andriamarosoa/note/actions/runs/39351032882) : **SUCCESS**. Commit testé `428950c46e995754b42af3522ed90a0738bc6a5d`. Six audits déterministes PASS, consensus PASS et consultation externe PASS. Le modèle ayant effectivement répondu est `z-ai/glm-5.3-flash`, sans repli. L'artefact du run nommé historiquement `S799_SEVENTH_OPINION_KIMI_NOT_PROMOTION` contient le JSON de cet avis GLM malgré l'ancien nom d'artefact. Réponse HTTP réelle reçue et analysée, 4 617 tokens de prompt, 2 024 tokens de réponse. Ce n'est pas Kimi : l'ancien Kimi K2.5 et le repli Llama 3.3 ont tous deux renvoyé HTTP 410 dans l'essai préalable [#39350745687](https://github.com/Andriamarosoa/note/actions/runs/39350745687).

## Métriques

Les mesures officielles GuitarSet CAL32 S708 restent inchangées :

| Métrique | Référence S708 | S799 |
| --- | ---: | --- |
| Onset F1 | 74,2639 % | non mesuré |
| Exact-K | 62,0455 % | non mesuré |
| Active Exact-K | 56,2762 % | non mesuré |
| Offset accuracy | 77,3292 % | non mesuré |
| Full Note F1 | 57,1319 % | non mesuré |
| Frame Pitch F1 | 76,4626 % | non mesuré |

## Comparaison à S708

S799 n'est **pas** une amélioration de la transcription. Aucun entraînement, rechronométrage, surcoût CPU LIVE, retard d'offset, attente de confirmation ou nouveau signal n'est ajouté. Les scores et la référence S708 ne sont ni remplacés ni re-signés. Aucune mesure physique de latence microphone n'a été réalisée.

## Audits

| Rôle | Statut | Preuve |
|---|---|---|
| Données | PASS | [Audit S799 #39351032882](https://github.com/Andriamarosoa/note/actions/runs/39351032882) |
| Acoustique | PASS | [Audit S799 #39351032882](https://github.com/Andriamarosoa/note/actions/runs/39351032882) |
| Architecture | PASS | [Audit S799 #39351032882](https://github.com/Andriamarosoa/note/actions/runs/39351032882) |
| Apprentissage | PASS | [Audit S799 #39351032882](https://github.com/Andriamarosoa/note/actions/runs/39351032882) |
| Métriques | PASS | [Audit S799 #39351032882](https://github.com/Andriamarosoa/note/actions/runs/39351032882) |
| LIVE | PASS | [Audit S799 #39351032882](https://github.com/Andriamarosoa/note/actions/runs/39351032882) |

Consensus strict : **PASS**. Le septième conseiller : **PASS pour la communication API**, et **non validé pour tout gain scientifique**.

## Verdict

Le conseiller GLM-5.3-Flash a identifié comme obstacle prioritaire l'absence des **probabilités OOF par fragment/hauteur**, indispensables pour auditer réellement le routeur S797. Il propose ensuite un découplage du contrôle Offset/Onset et un veto harmonique conditionnel, puis un choix du nombre de têtes validé FIT seulement. Ces hypothèses ne sont pas des résultats GuitarSet. Voir [avis_externe.md](avis_externe.md) pour la synthèse attribuée au modèle. L'avis reconnaît que les 3 TP / 0 FP de S797 proviennent d'une fixture synthétique et ne prouvent aucun gain de transcription.

## Prochaine action

S800 : instrumenter la sortie des probabilités OOF et décisions MIDI256 réelles par fragment/hauteur sur FIT80, conserver la parité signée S708 (attendu `k_correct=78366`), ablater séparément H4 Offset, H3 Onset, H2 harmoniques, H5 sélection des têtes. Ne promouvoir que si les six scores appariés CAL32 ne régressent pas ET que le contrat de causalité et de latence est respecté. Le modèle externe reste un conseiller sans pouvoir d'action.
