# V25 : bilan du comptage et hypothèse V26

Source : [run 34183897369](https://github.com/Andriamarosoa/note/actions/runs/34183897369), commit `042be93f7e6406fbf23c7e106c366a1ed0488e29`. Cinq paires réussies et synthèse réussie. Le run initial incomplet 34183558561 est exclu, y compris son fold 3 réussi.

Les données archivées dans v250-count-analysis.json sont extraites des journaux des cinq jobs et de la synthèse, avec leurs identifiants. Ce fichier est un extrait des résultats, pas le rapport intégral de l’artefact. Les sommes TP/FP/FN et les effectifs des matrices ont été vérifiés : 76 768 fenêtres, 44 220 événements de référence. Les résultats par K ci-dessous agrègent les cinq folds ; les journaux ne fournissent pas les matrices K croisées par fold.

## Résultat officiel à 50 ms

| Modèle | F1 | FP | FN |
|---|---:|---:|---:|
| V10.4 | 80.38 % | 6399 | 10203 |
| V24 | 79.77 % | 6989 | 10245 |
| V25 catégoriel | 79.13 % | 8749 | 9543 |
| V25 ordinal | 79.47 % | 7676 | 10000 |

## Résultats par fold

Les epochs sont celles retenues par la validation interne pour le réentraînement final.

| Fold | F1 catégoriel | F1 ordinal | Écart en points | Epochs cat. / ord. | Exactitude poly cat. / ord. |
|---|---:|---:|---:|---:|---:|
| 0 | 78.41 % | 78.33 % | -0.080 | 8 / 8 | 40.41 % / 41.40 % |
| 1 | 79.55 % | 79.54 % | -0.004 | 9 / 9 | 40.65 % / 37.22 % |
| 2 | 77.73 % | 79.39 % | 1.654 | 3 / 10 | 40.33 % / 33.08 % |
| 3 | 80.23 % | 79.91 % | -0.325 | 8 / 4 | 37.91 % / 33.67 % |
| 4 | 79.67 % | 80.21 % | 0.536 | 8 / 10 | 32.83 % / 39.88 % |

L’ordinal gagne sur les folds 2 et 4 seulement. Sur le fold 2, il retire 1 066 FP au prix de 473 FN supplémentaires ; les epochs sélectionnées diffèrent (3 contre 10). Hors fold 2, le F1 agrégé passe de 79,4578 % à 79,4928 % : l’avantage global est largement concentré sur un fold. Cette exclusion est une analyse de sensibilité descriptive, pas une nouvelle métrique de sélection. Une seule seed est testée ; aucune significativité statistique n’est établie.

## Erreurs selon le nombre réel K

K est la cible de comptage par fenêtre. Les taux concernent l’argmax de la tête, avant limitation par les candidats disponibles ; ils ne sont pas des F1 événementiels.

| K réel | Fenêtres | K exact cat. | K exact ord. | Sous-comptage cat. | Sous-comptage ord. |
|---|---:|---:|---:|---:|---:|
| 0 | 51956 | 93.14 % | 94.20 % | 0.00 % | 0.00 % |
| 1 | 15411 | 64.84 % | 65.49 % | 19.10 % | 20.62 % |
| 2 | 4279 | 44.19 % | 38.19 % | 32.46 % | 36.18 % |
| 3 | 2952 | 39.84 % | 41.16 % | 42.07 % | 38.55 % |
| 4 | 1628 | 26.04 % | 32.99 % | 63.21 % | 58.05 % |
| 5 | 438 | 27.63 % | 25.34 % | 72.37 % | 74.66 % |
| 6 | 104 | 0.00 % | 0.00 % | 100.00 % | 100.00 % |

- K=0 : les fausses présences passent de 3 565 à 3 015 fenêtres (6,86 % à 5,80 %). Elles représentent respectivement 5 227 et 4 272 unités de comptage en excès, avant réalisation temporelle. Ces nombres ne doivent pas être assimilés aux FP du score officiel.
- K=1 : les omissions passent de 2 944 à 3 178 fenêtres.
- K=2 : l’exactitude baisse de 44,19 % à 38,19 %, et le sous-comptage augmente.
- K=4 : l’ordinal progresse, mais sous-compte encore 58,05 % des fenêtres.
- K=6 : aucune des deux têtes ne prédit 6 dans l’ensemble des fenêtres ; seulement 104 fenêtres ont cette cible. Ce constat ne suffit pas à démontrer un bug.
- En polyphonie (K>=2), l’exactitude globale recule de 38,42 % à 37,20 %. Le gain en F1 de l’ordinal n’établit donc pas une meilleure estimation générale du nombre de notes.

## Décision et hypothèse V26

V10.4 reste la référence sur cette évaluation. Ne pas promouvoir V25 ordinal. L’effet observé ressemble surtout à un déplacement du compromis entre fausses présences et omissions ; changer la représentation de sortie ne résout pas le comptage.

Hypothèse à tester : la pondération inverse-racine des classes utilisée par les deux bras V25 favorise les classes non nulles et contribue aux fausses présences. Le code confirme cette pondération, mais V25 ne contient pas de témoin sans pondération : sa responsabilité reste une hypothèse.

V26 proposé : A/B catégoriel avec pondération V25 versus sans pondération (poids unitaires), même encodeur frais, mêmes seeds, partitions imbriquées, limites d’epochs, patience et NLL interne. Dans chaque bras, appliquer le même choix de pondération au fit et à sa NLL de validation. Conserver les candidats, timestamps et classement figés. Cette comparaison teste la politique de pondération complète, y compris son effet sur le choix d’epoch. Garder les deux bras dans le même run et publier tous les folds.

Critère principal : F1 agrégé à 50 ms, accompagné de TP/FP/FN, faux K>0 lorsque K=0, omissions K=1, et matrices K. Ne pas ajuster de seuil ni sélectionner des folds à partir des résultats externes. Ne pas ajouter simultanément une tête de présence, une nouvelle perte ordinale ou une nouvelle règle d’arrêt. Aucun entraînement V26 n’a été lancé dans cette analyse.
