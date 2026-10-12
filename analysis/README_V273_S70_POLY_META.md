# S70 — Nouveau contrôleur polyphonique via les probabilités PR16 (non promu)

Audit local sur **59 309 fragments**, dont **7 385 vrais accords poly**, folds historiques 0/1/2/4. Parent S69 polymax (référence figée) **49 148 corrects global / 3 094 corrects poly**.

| Modèle | Correct global | Correct poly | Changements | Corrections additionnelles / régressions | Gain global / poly |
|---|---:|---:|---:|---:|---:|
| S69 polymax | 49 148 (82,8677%) | 3 094 (41,8957%) | — | — | — |
| **S70 proposed_poly** | **49 202 (82,9587%)** | **3 166 (42,8707%)** | 1 263 | 451 / 397 | **+54 / +72** |
| S70 downward_poly | 49 208 (82,9689%) | 3 154 (42,7082%) | 777 | 281 / 221 | +60 / +60 |

Sur la première configuration, fold 0/1/2/4 : +7/+18/+19/+10 global, **+10/+26/+26/+10 poly**. Sur la deuxième : **+8/+31/+11/+10 global et poly**.

## Constat décisif sur la génération de propositions

Des 4 291 vrais accords polyphoniques incorrects sous S69-polymax :
- **587** ont leur vrai K dans les 192 sorties S46,
- **2 402** ont leur vrai K dans les 36 sorties PR16,
- **2 485** le trouvent dans l'union PR16/S46/S69 équilibré.
Le plafond **oracle** avec S69+S46 seul est 3 681/7 385 = **49,8443%**, tandis que l'union PR16+S46 atteint **75,5450%**. Ces plafonds utilisent la vérité terrain uniquement pour l'audit ; **ce ne sont pas des prédictions utilisables**. L'écart explique pourquoi un arbitre sur S46 seul plafonne plus tôt.

## Entrées et protocoles du nouveau modèle

Sur les **11 301 fragments** identifiés *sans vérité* par `S69 K>=2 ou au moins une proposition PR16 K>=2`, entrée :
- 12 sorties neuronales PR16 × 7 classes + moyenne et écart-type par classe ;
- votes des 36 politiques et classes gelées S69/S18/S69 équilibré ;
- écarts K et indications poly prédites, sans K vrai comme entrée.

Régresseur logistique multiclasse C=0,01. Pour chaque fold externe, fit sur les trois autres folds, prédictions test sur le fold tenu à l'écart, et seuil choisi uniquement sur les sorties **inner-OOF des trois folds de fit**, sous contrainte de gain global/poly positif **sur les folds de fit**. Après inférence, S69 est conservé si le candidat n'est pas admissible ou si le seuil n'est pas atteint. Aucune étiquette du fold courant ne règle ses seuils.

`proposed_poly` accepte seulement la proposition PR16 méta **K2–K6**, entraînement coût poly ×8 ; `downward_poly` accepte uniquement un candidat poly inférieur au K actuel, coût 1. **Aucune modification du modèle principal.**

## Essais négatifs préservés

- S69 équilibré/polymax : Ridge, LR, HGB et 58 variables audio ; **pas de gain poly**.
- 192 propositions S46 au vote majoritaire : pertes poly.
- Routeur par classe initiale entre toutes les politiques S46 : +29 net global, **−3 poly**.
- Retour sélectif vers S18 (17 actions) : +9 net global, **+1 poly**, trop petit.

## Limites

**Expérimental seulement :** chaque source PR16 est OOF pour sa propre sortie, mais les modèles amont qui produisent les *features d'entraînement du méta-apprenant* ne sont **pas complètement nested** avec le fold externe (information indirecte possible). Les mêmes folds ont guidé le développement historique ; la valeur +72 peut être optimiste. Les variantes C=0,01/0,1, plusieurs domaines et seuils ont été comparées sur ces mêmes folds. Une nouvelle cohorte musicale tenue à l'écart et un entraînement upstream nested sont nécessaires pour valider. Les 36 politiques sont corrélées, pas indépendantes.

Reproduction : `scripts/audit_v273_s70_pr16_poly_meta.py` ; workflow `v273-s70-pr16-poly-meta.yml`. Sources : run S69 **38151710659**, run PR16 **37837235723**. Les décisions, seuils et scores doivent être vérifiés dans la CI avant toute conclusion de reproductibilité distante. Rapport complet et toutes les ablations disponibles dans archive locale S70.


## GitHub Actions — répétitions de la mesure

- [Run #38153394681](https://github.com/Andriamarosoa/note/actions/runs/38153394681) : **succès complet**, entraînement et contrôle des 59 309 fragments. S70 `proposed_poly` = **49 202 global et 3 166 poly, 451 corrections / 397 régressions** par rapport à S69 ; `downward_poly` = **49 208 global et 3 154 poly, 281 corrections / 221 régressions**. Artefact `v273-s70-poly-meta-replay`.
- [Premier run #38153163682](https://github.com/Andriamarosoa/note/actions/runs/38153163682) : entraînement réussi mais assertion d'égalité stricte rejetée. Une exécution numérique a choisi le seuil 0,15 au lieu de 0,06 sur le fold2 et donné **49 200 global / 3 161 poly** sur `proposed_poly` ; `downward_poly` a conservé **49 208 / 3 154**. Ne pas cacher cette sensibilité aux seuils.
- Le workflow vérifie désormais la cohérence interne des prédictions et du bilan, les classes natives et la non-interférence, et affiche les chiffres réellement observés ; il ne suppose pas que tous les environnements sélectionnent exactement le même seuil près de la frontière.

Ces tests confirment la *reproductibilité sur les données de développement*, pas la performance sur musique inédite.
