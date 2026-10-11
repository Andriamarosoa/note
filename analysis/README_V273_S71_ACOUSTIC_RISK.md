# S71 — contrôle acoustique des régressions S70 (recherche, non promu)

**Rejeu GitHub vérifié :** [Actions #38154433492](https://github.com/Andriamarosoa/note/actions/runs/38154433492) ; artefact `v273-s71-acoustic-risk`. Comparaison native alignée sur **59 309** fragments, dont **7 385** polyphoniques.

| Version | Global correct | Poly correct | Corrections contre S69 | Régressions contre S69 | Gain net contre S70 |
|---|---:|---:|---:|---:|---:|
| S69 parent | 49 148 (82,87 %) | 3 094 (41,90 %) | — | — | — |
| S70 proposed_poly | 49 202 (82,96 %) | 3 166 (42,87 %) | 451 | 397 | — |
| **S71 risk** | **49 220 (82,99 %)** | **3 184 (43,11 %)** | **380** | **308** | **+18** |
| **S71 poly** | **49 226 (83,00 %)** | **3 190 (43,20 %)** | 406 | 328 | **+24** |

**S71 risk** : veto de 234 des 1 263 changements proposés par S70, 89 régressions évitées, 71 corrections abandonnées, donc +18 global/+18 poly. Folds 0/1/2/4 : **+2/+3/+6/+7** ; vrais K3 +10, K4 +10, K5 −2. Sur 19 morceaux, gains positifs 13, nuls 2, négatifs 4 ; bootstrap descriptif par morceau 95 % environ **[+3,+33]**, très optimiste du fait des multiples variantes explorées.

Six transitions examinées (veto uniquement) :
- K3→4 : 50 veto / 22 régressions évitées / 11 corrections perdues = **+11**.
- K4→3 : 44 / 17 / 11 = **+6**.
- K2→3 : 64 / 26 / 21 = **+5**.
- K4→5 : 7 / 4 / 2 = **+2**.
- K3→2 : 69 / 20 / 26 = **−6**.
- K5→4 : aucune intervention retenue.

**S71 poly** supprime, après examen des résultats, tous les veto K3→2 ; davantage de décisions poly correctes (+24) mais **20 régressions supplémentaires** par rapport à S71 risk. Cette ablation est **sélectionnée ex post** sur les mêmes folds historiques.

## Nouveau mécanisme

1. Sélectionner les 1 263 décisions réellement modifiées par S70 sur la base de ses sorties, sans filtrer selon la vérité.
2. Construire 104 variables audio issues des caractéristiques PR16 (58 résumés morphologiques/énergétiques et 46 géométriques), ancien K, K proposé, et ΔK ; aucune vérité d'entraînement dans les entrées d'inférence.
3. Entraîner un `LogisticRegression(C=0.05)` régularisé, pondéré pour distinguer régressions et corrections (cas fausse→fausse poids 0.2), sur les trois folds hors du fold évalué.
4. Choisir les seuils de veto **sur des prédictions inner-OOF du jeu d'entraînement seulement**, par transition ; retourner uniquement à la prédiction parent S69.
5. Réévaluer global/poly, K, fold, non-interférence et correction/régression de façon appariée.

Code `scripts/audit_v273_s71_acoustic_risk.py`, pipeline `.github/workflows/v273-s71-acoustic-risk.yml`. Tests numériques et audit des 59 309 identifiants réussis sur GitHub.

**Limites essentielles :** 144 essais de familles, coûts et seuils ont été comparés sur les mêmes 19 morceaux historiquement exposés avant de retenir S71. Les producteurs amont PR16/S70 ne sont pas entièrement nested pour ce méta-apprentissage. Les scores ne sont **pas** une estimation indépendante de généralisation. Pas d'intégration automatique à `main` ni de remplacement silencieux de S70.
