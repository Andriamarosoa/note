# S67 — Effet de l'ordre sur les 111 régressions sans veto

L'audit S66 avait 111 régressions du fold 1 où aucune des 7 têtes de veto n'est active. La demande S67 teste explicitement l'ordre sur **chacun de ces 111 cas**.

## Expérience 1 : 7 têtes déjà enregistrées

Pour chaque cas, toutes les 7! = 5 040 permutations ont été rejouées : **559 440 séquences**. Elles ont **toutes exactement la même sortie S58 incorrecte**. Les 7 têtes sont des refus figés : aucune ne se déclenche sur ces cas. Leur combinaison par OR est commutative.

## Expérience 2 : ordre des 36 politiques gelées

Les 36 politiques PR16 sont corrélées, ce ne sont pas 36 modèles indépendants. Ces 111 cas possèdent respectivement 1, 2, 3, 4, 5 K différents parmi les politiques, dans **11, 48, 41, 9, 2** cas. La vérité initiale K est retrouvée dans au moins une des 36 sorties **103 fois** et absente **8 fois** (le chiffre auparavant évoqué de 110 est corrigé).

**Règle séquentielle expérimentale** : lire les votes politiques l'un après l'autre, verrouiller le premier K atteignant q votes, sinon garder S58. Ceci est un **décodeur hypothétique sur prédictions figées**, pas un repassage neuronal dépendant du K modifié. Aucune vraie étiquette ne gouverne le décodage.

Pour chaque fragment et seuil, l'ensemble exact des sorties possibles par n'importe quel ordre des 36 politiques est : les K ayant au moins q votes ; si aucun, S58. La preuve est constructive : placer en premier tous les votes de la classe voulue. Un ordre témoin utilisant les 36 indices de politiques a été vérifié pour chaque classe accessible. On **n'énumère pas matériellement les 36! permutations**. Contrôle croisé par permutations exhaustives de six votes.

| Seuil q | Cas sensibles à l'ordre | Cas où au moins un ordre retrouve le vrai K | Cas où tous les ordres retrouvent le vrai K |
|---:|---:|---:|---:|
| 8 | 53 | 90 | 37 |
| 12 | 32 | 88 | 57 |
| 16 | 7 | 77 | 70 |
| 18 | 0 | 74 | 74 |
| 24 | 0 | 46 | 46 |
| 32 | 0 | 11 | 11 |

Parmi les 32 cas sensibles à l'ordre à q12, **31 ont au moins un ordre aboutissant au vrai K** ; un cas n'a que des K incorrects dans cette stratégie.

## Ordre fixe d'origine contre inverse, non choisi par label

Les simulations sur **tout le fold 1 (13 868 fragments)** comptent aussi les régressions hors des 111 cas.

| q | Cas parmi les 111 où origine/inverse diffèrent | Corrigés parmi les 111, origine/inverse | Différences origine/inverse dans fold entier | Gain global fold 1 vs S58, origine/inverse |
|---:|---:|---:|---:|---:|
| 8 | 29 | 80/63 | 999 | -19/+62 |
| 12 | 20 | 81/68 | 521 | +10/+75 |
| 16 | 1 | 77/76 | 112 | +44/+56 |
| 18 | 0 | 74/74 | 42 | +48/+57 |
| 24 | 0 | 46/46 | 0 | +70/+70 |
| 32 | 0 | 11/11 | 0 | +33/+33 |

**Cas 993** : vrai et initial K4, S58 K3 ; 22 votes K4 et 14 votes K3. À q8 ordre source K4, ordre inversé K3. À q12 K3 ou K4 possibles selon un ordre ; à q16 K4 seul, à q32 aucun ne passe et K3 est conservé. **Cas 20261** : vrai et initial K2, S58 K3 ; à q12, l'ordre d'origine K2, l'ordre inversé K3.

## Limites

- Les sept têtes historiques sont des **veto purs** ; on a testé exactement leur ordre, et il est sans effet sur ces 111 cas.
- La seconde expérience emploie un **nouveau décodeur séquentiel expérimental**. Elle ne recalcule pas les 36 réseaux ni les têtes à chaque changement d'état. Les 36 sorties figées ne prouvent pas l'existence d'un avantage avec un vrai réseau récurrent.
- Les 111 cas sont sélectionnés après comparaison à leurs vraies étiquettes, donc ni le meilleur ordre ni un seuil réglé dessus ne constitue une généralisation.
- Sur le fold entier, on constate qu'un ordre corrigeant davantage de ces 111 cas peut perdre ailleurs. Exiger une évaluation indépendante et le coût des régressions avant toute activation.
- Ce rapport ne modifie aucun réseau ; absence de promotion.

**Fichiers expérimentaux locaux S67** : `s67_order111/s67_test_order_111.py`, `s67_order111/s67_order_protocol_comparison.py`, `s67_order111/test_s67_order.py`, `s67_order111/per_case_111_order_audit.csv`, `s67_order111/witness_orders_examples.json`, `S67_ORDRE_111_REPRODUCTIBLE.zip`. Trois tests de validation passent.
