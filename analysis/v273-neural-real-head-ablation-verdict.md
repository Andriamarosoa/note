# V27.3 — Ablation causale du réseau neuronal entraîné

**Date : 8 octobre 2026**

## Exécution et authenticité

- [GitHub Actions — run 37751698564, succès](https://github.com/Andriamarosoa/note/actions/runs/37751698564).
- Modèle : réseau TensorFlow/Keras à **14 têtes** avec sélection conditionnelle au K proposé, et choix KEEP séparé.
- Expérience : 59 309 événements natifs K0–K6, dont 7 493 révisables ; folds 0,1,2,4 ; fold3/player05 exclus.
- Reproduction : **0 décision différente sur 7 493** par rapport au run original 37747861591. Nous avons donc reproduit **exactement** les sorties, bien que les modèles précédents ne possédaient pas de checkpoint archivé.
- Exact-K reproduit : **82,3012 % global**, **28,3006 % poly K2–K6**. Référence gelée inchangée : **81,6976 % global**, **34,2586 % poly**.
- Intervention : le **même modèle entraîné par fold** est inféré avec une tête ou un groupe masqué : aucune modification de poids, de cible, de données ou de baseline entre les interventions.

## Résultat 1 — Toutes les têtes acoustiques H1–H5 sont utiles au réseau

| Retrait | Erreurs corrigées | Nouvelles erreurs | Bilan vs réseau intact | Régressions K2→K1 réparées |
|---|---:|---:|---:|---:|
| H1 Spectral | 161 | 189 | **−28** | 38 |
| H2 Cycle de vie | 161 | 191 | **−30** | 40 |
| H3 Harmoniques | 158 | 191 | **−33** | 40 |
| H4 Fondamentales | 159 | 189 | **−30** | 38 |
| H5 Sources | 157 | 194 | **−37** | 39 |
| H1–H5 ensemble | 737 | 1 305 | **−568** | 196 |

**Conclusion :** enlever les têtes acoustiques récupère une partie des
cas K2→K1, mais détruit davantage de bonnes décisions ailleurs. Le
réseau fait usage de leur information ; les supprimer n'est pas une
réponse générale au défaut de son décodeur.

## Résultat 2 — La contribution d'une tête est fortement dépendante du VRAI K

Voici le changement en **nombre de prédictions correctes** du modèle
lorsqu'on retire une tête, relativement au modèle intact (nombres
descriptifs calculés après inférence ; le vrai K n'est jamais un
sélecteur à l'inférence).

| Tête retirée | Δ K2 | Δ K3 | Δ K4 | Δ global |
|---|---:|---:|---:|---:|
| **C23**, correction K2→K3 | **+67** | **−140** | +9 | +10 |
| **C32**, correction K3→K2 | **−194** | **+87** | +14 | −47 |
| **C34**, correction K3→K4 | +17 | **+60** | **−73** | +22 |
| **C43**, correction K4→K3 | +28 | **−103** | **+94** | +25 |

**Point central demandé par l'utilisateur :** une action peut être
importante pour préserver K3 et nuisible pour K2. Un seul coefficient
global ne peut représenter ces profils opposés.

Quelques interprétations contrôlées :
- Retirer C23 procure +67 vrais K2 mais détruit 140 vrais K3.
- Retirer C32 détruit 194 vrais K2 tout en gagnant 87 vrais K3.
- Retirer C34 détruit 73 vrais K4 mais récupère 60 K3.
- Retirer C43 récupère 94 vrais K4 et détruit 103 vrais K3.

Un bilan net global positif **ne constitue pas** un motif automatique
de suppression : **C34, C43 et C23 montrent des arbitrages contraires
entre K2/K3/K4**.

## Résultat 3 — Les effets diffèrent même entre les folds

Bilan net global de retrait par fold :

| Fold | Sans C23 | Sans C32 | Sans C34 | Sans C43 |
|---|---:|---:|---:|---:|
| 0 | −10 | +6 | −9 | +1 |
| 1 | +6 | −5 | +20 | +22 |
| 2 | +1 | −2 | +2 | −1 |
| 4 | +13 | −46 | +9 | +3 |

Le retrait de C32, par exemple, est presque neutre pour certains folds
et détruit 46 prédictions nettes au fold 4. Ces distributions ne
justifient pas une règle statique globale.

## Résultat 4 — Ablations de sous-ensembles K-spécifiques

Une banque de **7 têtes** contient bien **127 sous-ensembles** non
vides. Mais H0 est indispensable au modèle actuel pour conserver
une voie de sortie K0/K1 ; une suppression de H0 invaliderait son
contrat de sortie. Seuls les **64 sous-ensembles incluant H0**
sont donc fidèlement ablatables, par transition ; les **63 sans
H0** sont inaccessibles sans *changer le modèle*.

Chaque intervention de sous-ensemble est une inférence du **vrai
réseau neuronal**, sans réentraîner ses paramètres, et diffère
du précédent audit de votes directs simplifiés.

| Scénario par K initial | Sous-ensembles comparés | Retraits à bilan positif | Retraits à bilan négatif | Meilleur Δ vs réseau intact | Pire Δ |
|---|---:|---:|---:|---:|---:|
| K2→K3 | 64 | 6 | 58 | +27 | −648 |
| K3→K2 | 64 | 1 | 62 | +1 | −133 |
| K3→K4 | 64 | 0 | 64 | −53 | −202 |
| K4→K3 | 64 | 32 | 32 | +22 | −19 |

Attention : **le bilan d'un sous-ensemble** est ici mesuré sur tous
les cas dont la **baseline a annoncé le K initial**, pas uniquement
sur les interventions où la sortie finale prend le K destination.
Les autres têtes de correction et KEEP sont elles aussi masquées.
Les résultats ne constituent donc pas une performance d'un sélecteur
dynamique de 64 experts entraîné.

Les sous-ensembles présentant les bilans descriptifs extrêmes :
- K2→K3 : H0+H1+H2+H3+H4+H5+C23 **+27**, H0 seul **−648**.
- K3→K2 : toutes les têtes autorisées **+1**, H0 seul **−133**.
- K3→K4 : **même le meilleur cas −53** (toutes les têtes autorisées),
  H0+C34 **−202**. Retirer les têtes de KEEP/autres corrections
  a ici un effet notable.
- K4→K3 : H0+H2+H3+H5 **+22**, H0+H3+C43 **−19**.

**Ne pas choisir les meilleurs masquages après avoir regardé le fold
tenu à l'écart** : ce serait un biais de sélection. Ces résultats sont
uniquement des diagnostics.

## Résultat 5 — Pourquoi K2→K1 demeure dominant

Le run forensique indépendant précédent :
[37748832939](https://github.com/Andriamarosoa/note/actions/runs/37748832939)
a démontré que la décision K1 est produite par le **décodeur libre**
alors que la banque de têtes actuelle n'apporte aucune probabilité
K1 explicitement entraînée. Notre ablation confirme qu'il n'y a pas
une tête acoustique unique à enlever : retirer H1–H5 récupère certes
196 régressions K2→K1, mais entraîne **1 305 nouvelles erreurs**
contre 737 réparations neuronales ailleurs, soit **−568** net.

De plus, la tête de risque auxiliaire n'entre pas dans les logits de
la décision finale. Ce défaut de connexion est inchangé dans ces
interventions : elles ne peuvent que mesurer des sensibilités, pas
réparer le contrat d'action.

## Verdict de conception

1. **Le problème n'est pas qu'une tête acoustique est globalement
   mauvaise.** Les cinq sont utiles à l'intérieur du modèle malgré
   des erreurs localisées.
2. **Les têtes d'action ont des effets antagonistes K par K** ; toute
   sélection globale ou simple tri selon net correct/regression est
   une mauvaise abstraction.
3. **Les règles de support K ne contraignent pas suffisamment le
   décodeur de classes K0/K1**.
4. **Le risque appris n'est pas branché dans l'arbitrage KEEP**.
5. La vraie amélioration devra porter sur **le contrat d'évidence
   par classe et le risque de changer un K déjà correct**, avec
   validation réellement indépendante.

## Reproduction

Script :
\`scripts/audit_v273_neural_head_ablations.py\`

Tests :
\`test/test_v273_neural_head_ablations.py\`

Run :
https://github.com/Andriamarosoa/note/actions/runs/37751698564

Artefact :
https://github.com/Andriamarosoa/note/actions/runs/37751698564/artifacts/11538885226

Il contient :
- \`rootcause-ablation.json\` : bilans complets par K, scénario, fold ;
- \`all-ablation-folds.csv\` : chaque tête et chaque sous-ensemble par fold ;
- \`decisions-by-head.npz\` : toutes les décisions originales et par
  ablation individuelle/groupée, avec labels de diagnostic ;
- \`verdict.md\`.

Aucune modification de \`freeze_local_combo\` ni promotion de modèle.
