# S72 — Audit des 308 régressions S71 et nouvelles têtes temporelles

**Statut : exploration sur folds historiques 0/1/2/4 ; NON PROMU.** 59 309 fragments dont 7 385 vrais accords poly. Base S69 polymax (49 148 corrects global, 3 094 poly). S71 risk correct **49 220 global, 3 184 poly**, avec 380 corrections, 308 régressions, 341 changements faux→faux par rapport à S69.

## Audit des régressions S71

Les 308 régressions portent principalement sur : K4→K3 **76**, K2→K3 **69**, K3→K2 **68**, K3→K4 **29**, soit 242/308 (78,57%). Les mêmes transitions portent encore 101, 85, 80 et 25 corrections ; supprimer toute une transition est inadapté.

## Protocole S72

- 1 029 changements S71 par rapport à S69, sélectionnés uniquement sur prédictions, sans vraie classe ; features extraites des archives PR16 couvrant les 59 309 IDs, alignment intègre.
- 58 caractéristiques acoustiques et **8×49 résumés temporels** de la trajectoire originale 42×49 : attaque, début, maintien, fin, maximum, variance, pente, variation absolue ; ancienne classe, nouvelle classe, ΔK.
- Nouvelle tête `HistGradientBoostingClassifier`, max_iter=45, max_leaf_nodes=7, min_samples_leaf=30, L2=30, learning_rate=.06, random_state=72. Cible : régression par rapport à S69 ; corrections et régressions poids=1, faux→faux poids=0,15.
- Pour chaque fold évalué, fit sur les 3 autres folds, et seuil appris **seulement sur scores inner-OOF des trois folds d'entraînement**.
- Famille exploratoire `K1→2, K3→4, K4→5` choisie *après* plusieurs recherches sur ces folds ; variante `poly_guard` refuse uniquement `K3→4, K4→5`, sans intervention depuis K1.

## Mesures appariées

| Version | Global correct | Poly correct | Régressions vs S69 | Gain net poly vs S71 parent |
|---|---:|---:|---:|---:|
| S71 risk | 49 220 (82,9891 %) | 3 184 (43,1144 %) | 308 | — |
| S71 poly | 49 226 (82,9992 %) | 3 190 (43,1957 %) | 328 | — |
| **S72 risk + poly_guard** | **49 232 (83,0093 %)** | **3 196 (43,2769 %)** | **282** | **+12** |
| **S72 poly + poly_guard** | **49 238 (83,0194 %)** | **3 202 (43,3582 %)** | **302** | **+12** |
| S72 risk + tous les veto `K1/3/4` | 49 238 | 3 191 | 271 | +7 poly / +18 global |

La variante poly_guard annule **56 changements** : 26 régressions évitées, 14 corrections perdues, 16 faux→faux annulés. Sur K3→4 : 32 refus, 13 régressions évitées, 8 corrections perdues = +5. Sur K4→5 : 24 refus, 13 régressions évitées, 6 corrections perdues = +7.

**Gains poly par fold 0/1/2/4** : +7/+4/+0/+1. Sur 19 morceaux, 13 positifs, 3 nuls, 3 négatifs ; bootstrap descriptif par morceau 95 % approximatif **[+3,+22]**, **non corrigé de multiples essais**.

## Essais négatifs

Ablations têtes votes seule, acoustic 104D, temporal 450D, neuro probabilities+acoustic, combined et temporal+vote avec LR, HGB ; 3 coûts et plusieurs regroupements : plusieurs modèles ne trouvent **aucun seuil rentable sur leurs propres folds d'entraînement**, ou perdent des corrections. Un modèle probabilités + acoustic ajoute +7 net global mais perd 7 décisions sur le fold2. Aucun modèle ne prouve une résolution générale des 308 régressions.

**Non promotion :** famille K et variante retenues ex post ; les producteurs PR16/S70/S71 ne sont pas intégralement nested pour un nouveau méta-apprenant ; morceaux historiques explorés ; aucun test sur morceaux inédits. Le bénéfice de +12 est une observation de développement reproductible, **pas une estimation non biaisée**. Référence officielle inchangée.

Code, scores des grilles et décisions par cas dans l'archive S72 locale, plus nouveau pipeline de rejeu.