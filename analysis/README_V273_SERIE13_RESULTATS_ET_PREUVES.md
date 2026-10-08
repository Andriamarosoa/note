# V27.3 — Série 13 : preuve des corrections K0 sans régression

Protocole préannoncé : [README_V273_SERIE13_DYNAMIQUE_K0.md](README_V273_SERIE13_DYNAMIQUE_K0.md). Entraînement : https://github.com/Andriamarosoa/note/actions/runs/37854384791. **Rejeu indépendant du décodeur** : https://github.com/Andriamarosoa/note/actions/runs/37854672038.

Les 24 stratégies de veto K0 ont été régénérées de zéro à partir des probabilités archivées avec concordance de toutes les décisions et les **76 fichiers de modèles** (19 morceaux × 2 entrées × 2 algorithmes) vérifiés par SHA256. Les données source audio et le fold évalué sont exclues du fit, et la cible évaluée n'est pas une feature. Le parent série9 demeure post-hoc sur cette cohorte. Les résultats **ne constituent ni une validation indépendante ni une promotion**.

| Variante | Corrections vs freeze | Régressions vs freeze | Régressions évitées vs S9 | Corrections perdues vs S9 | Exact global | Exact poly |
|---|---:|---:|---:|---:|---:|---:|
| Série 9 parent | 2 934 | 2 226 | 0 | 0 | 82,8913 % | 40,4739 % |
| Sér. 13 `flow_logistic p0>0.975` | **2 934** | **2 223** | **3** | **0** | **82,8964 %** | **40,4739 %** |
| Sér. 13 `flow_logistic p0>0.9` | 2 915 | 2 191 | 35 | 19 | 82,9183 % | 40,4739 % |
| Sér. 13 `flow_logistic p0>0.85` | 2 898 | 2 154 | 72 | 36 | 82,9520 % | 40,4604 % |

Le choix conservateur `p0>0.975` restaure exactement les trois prédictions suivantes, **sans toucher aux 2 934 corrections**.

| ID natif | Fold | Vrai K | Freeze K | Série 9 K | Série 13 K | Probabilité apprise d'absence de nouvelle attaque |
|---:|---:|---:|---:|---:|---:|---:|
| 25912 | 0 | 0 | 0 | 1 | 0 | 0.981016 |
| 34704 | 4 | 0 | 0 | 1 | 0 | 0.980301 |
| 45494 | 4 | 0 | 0 | 1 | 0 | 0.981512 |

Les trois erreurs étaient des **surcomptages K0→K1**. Cela constitue un signal descriptif à auditer au niveau acoustique, **pas** une preuve causale de la forme du signal ou de stabilité hors cohorte.

Archive **durable** des poids, probabilités, résultats par K/fold et cas de correction : https://github.com/Andriamarosoa/note/releases/tag/v273-series13-research-37854672038 (prépublication, non modèle de production).

## Interprétation

Les garde-fous des séries 11 et 12 peuvent retirer beaucoup de régressions (1 100 / 1 122) au prix de 806 / 849 corrections perdues. La série 13, qui cible les naissances K0 et exploite les trajectoires pré/on/post, obtient un petit résultat strict sans perte (+3), tandis que des seuils plus bas échangent encore des régressions contre des corrections. Préserver les 24 décisions et leurs traces, sans remplacer le parent au titre de succès général : pour dépasser YourMT3+, les lacunes K2–K6 et la stabilité des sélections doivent encore être traitées. Les scores externes sur morceaux inédits restent nécessaires.
