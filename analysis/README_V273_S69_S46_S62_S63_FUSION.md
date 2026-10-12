# S69 — Fusion S46 (propositions K) + contrôles S62/S63 (experimental only)

**Statut : non promu.** Analyse sur 59 309 fragments (folds historiques 0,1,2,4), 7 385 vrais accords polyphoniques. Référence S18, **49 178/59 309** global, **2 998/7 385** poly.

| Variante | Globaux corrects | Poly corrects | Corrections / régressions vs S18 | Gain vs S18 |
|---|---:|---:|---:|---:|
| S46 polymax brut | 49 039 (82,6839 %) | 3 087 (41,8009 %) | 808 / 947 | −139 |
| S46 polymax + contrôle S63 propre à S46 | 49 148 (82,8677 %) | **3 094 (41,8957 %)** | 649 / 679 | −30 |
| S46 équilibré brut | 49 178 (82,9183 %) | 3 063 (41,4760 %) | 671 / 671 | 0 |
| **S46 équilibré + contrôle S63 propre à S46** | **49 210 (82,9722 %)** | **3 063 (41,4760 %)** | **604 / 572** | **+32** |

Les nouveaux contrôles sont des **veto par transition K**, réentraînés sur les trois autres folds, fondés sur les comptes des 36 politiques corrélées PR16 et les K initial/proposé, sans vérité terrain du fold évalué lors de la décision. Transitions étudiées : K3→2, K2→3, K4→3, K3→4, K1→0, K0→1. Coût de régression testé : 1,5 pour le parent équilibré et 1,2 pour polymax. Ces coûts et variantes ont été comparés sur des folds de développement préalablement explorés : **sélection ex post = estimation optimiste**, et les sources amont ne sont pas toutes de provenance nested prouvée.

**Effet S63 par fold 0/1/2/4 :**
- Équilibré : gains globaux +10/+3/+7/+12 ; différence poly +1/0/−1/0. Les 99 régressions évitées se paient de 67 corrections perdues.
- Polymax : gains globaux +19/+41/+18/+31 ; différence poly −2/+3/+3/+3. Le score global reste inférieur à celui de S18.

**Audit distinct du contrôle S62 morphologique :** sur 58 descripteurs acoustiques, 46 géométriques et les statistiques 42×49 temporelles, têtes HGB régularisées par transition, choix representation/seuil uniquement sur trois autres folds : S46 équilibré +0 net / +0 poly ; S46 polymax −1 net / −1 poly. Ajout aux veto S63 : **aucun gain supplémentaire** pour équilibré ; −1 global et −1 poly sur polymax. **Non retenu.**

**Transfert brut des anciens 39 refus S62/S63 appris sur S58 :** seulement 3 cas transférables sur S46 équilibré, 1 cas sur polymax ; gain 0 pour chacun. Ne jamais réutiliser sans adaptation au producteur.

Sources :
- Série S46, exécution GitHub Actions **37891336823**, artefact `v273-series46-poly-class-compromises`, prédictions 193 politiques
- PR16, run **37837235723**, artefact `v273-open-summary`, prédictions 36 politiques
- Audio PR16, quatre artefacts `v273-open-features-{0,1,2,4}`
- Ancien audit S62+S63, non promu, scripts et prédictions S60–S64

Audit complet avec évaluations par vrai K, source K, fold et prédictions : archive locale `S69_REPRODUCTIBLE.zip`. **Jamais d'intégration automatique dans le réseau principal : validation externe sur nouveaux morceaux et producteurs imbriqués nécessaire.**
