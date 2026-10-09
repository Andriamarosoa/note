# V27.3 — Boucles S36/S37, conservation des bonnes décisions (H9 exclue)

## Résultats exécutés, avec vérifications

**Référence historique S18** inchangée : 49 178/59 309 Exact-K global = 82,9183 % ; 2 998/7 385 Exact-K poly = 40,5958 %. Toutes les actions acoustiques viennent du catalogue S35 (18 modules sans H9, A/B historique en plus), H8 étant bloquée tant qu'il manque les prédictions de vrais WAV pitch-shift. Aucune politique de ce rapport ne modifie les poids de S18 ou ne crée de règle par identifiant d'un événement.

### S36a — ablations structurelles sans entraînement nouveau

Run [37869563550](https://github.com/Andriamarosoa/note/actions/runs/37869563550) ; archive [S36a](https://github.com/Andriamarosoa/note/releases/tag/v273-series36a-research-37869875118).

| Variante | Exact global | Exact poly | Corrections S18 | Régressions S18 |
|---|---:|---:|---:|---:|
| S35 globale λ1/0,05 | 83,0414 % | 38,1043 % | 382 | 309 |
| S36a S35 globale limitée aux anciens K≤3 | **83,0532 %** | 38,1991 % | 323 | 243 |
| S36a S35 prudente limitée aux anciens K≤3 | 82,9250 % | **40,6364 %** | 6 | 2 |

**Aucune des 20 protections sans apprentissage ne réalise un gain positif avec zéro régression.** Bannir toutes les corrections issues des K2/K3/K4 efface également de bons gains.

### S36b — sécurité apprise sur les autres morceaux du même fold

Protocole : `analysis/README_V273_SERIE36_GARDE_REGRESSIONS_POLY.md`. Entraînement [37869664139](https://github.com/Andriamarosoa/note/actions/runs/37869664139) ; rejeu indépendant réussi [37869936965](https://github.com/Andriamarosoa/note/actions/runs/37869936965) ; [archive S36b](https://github.com/Andriamarosoa/note/releases/tag/v273-series36b-research-37869936965).

38 modèles de décision `Ridge`, deux groupes de descripteurs (signal audio/harmonique et signal+chemin), apprentissage par morceau exclu, 19 morceaux et les mêmes folds OOF S35. Pour chaque proposition de K, estimer *probabilité de correction* et *risque de destruction d'une ancienne réponse correcte*, puis conserver S18 si le score `fix−λ·regress` est trop faible. Les vrais labels du morceau cible ne servent **pas** aux décisions.

Candidats gelés : 2 sources S35, 2 vues, 3 poids λ, 3 seuils, donc **36 politiques apprises**, plus S18 et 2 références S35. Le rejeu indépendant confirme **39/39 vecteurs de prédiction exactement identiques** et **38/38 modèles rechargés avec exclusion du morceau évalué**.

| Variante | Global Exact-K | Poly Exact-K | Nouvelles corrections | Nouvelles régressions |
|---|---:|---:|---:|---:|
| S18 | 82,9183 % | 40,5958 % | 0 | 0 |
| S36b prudente `S35 λ4/0 — full_path — λ2 / seuil0,005` | **82,9216 %** | **40,6229 %** | **2** | **0** |
| S36b ouverte `S35 λ1/0,05 — full_path — λ1 / seuil0,02` | 83,0296 % | 39,1740 % | 237 | 171 |

Le candidat prudent fait **49 180 bonnes réponses globales et 3 000 poly**, contre 49 178/2 998 dans S18. **Deux événements réellement corrigés, aucun des 49 178 corrects S18 perdu sur ce corpus**. Leurs identifiants natifs sont **57164 et 37676**, extraits du rejeu, pas utilisés pour l'apprentissage ou les règles. Trois seuils annoncés 0 / 0,005 / 0,02 conduisent au même ensemble de ces deux événements ; la vérification archive séparément leurs identifiants pour éviter de supposer cette identité à partir des seuls scores.

**Attention : gain minuscule et sélection sur développement historiquement exposé.** La vérification indépendante signifie *reproduire le calcul à partir des poids sauvegardés*, PAS prédire correctement sur des compositions inédites. Aucune promotion en production.

### S37 — neuf sorties S35 possibles avec les mêmes modèles S36b, gelés

Protocole : `analysis/README_V273_SERIE37_NEUF_PARCOURS_VETO.md`, run terminé [37870170306](https://github.com/Andriamarosoa/note/actions/runs/37870170306). Cette boucle a essayé les neuf parcours S35, sans réentraîner les 38 modèles de risque : 2 vues × 3 λ × 5 seuils = **30 politiques** ; pour chaque événement choisir le parcours ayant la meilleure utilité estimée, sinon s'abstenir.

| Variante S37 | Global | Poly | Corrigées | Régressions |
|---|---:|---:|---:|---:|
| Meilleur global `full_path λ1 seuil0,02` | **83,0498 %** | 38,6594 % | 361 | 283 |
| Variante plus prudente `audio_flow λ4 seuil0,10` | 82,9807 % | 39,7833 % | 144 | 107 |
| Contrôle S36b gelé | 82,9216 % | **40,6229 %** | **2** | **0** |

**Aucune des 30 stratégies S37 ne préserve toutes les bonnes réponses S18.** Multiplier les chemins indépendamment de la bonne discrimination harmonique restaure aussi les mauvaises sélections. Les corrections et régressions (même négatives) sont archivées, sans sélection de seuil par vraie classe.

## Décision expérimentale et suite

- Conserver **l'architecture de l'ordonnanceur par événement**, H1–H7/H10, correcteurs et KEEP, expert K1/K2, H8 masquée, H9 exclue.
- Garder **S18 référence**, et enregistrer S36b 2/0 comme *premier candidat de recherche satisfaisant zéro perte sur la cohorte explorée*, sans promotion ni prétention à dépasser YourMT3+.
- L'audit des têtes S35 a révélé 302/309 régressions sur les vrais K2/K3/K4. La priorité est une **preuve de présence de note/attaque indépendante** fiable par signal et par candidat, plutôt que le vote moyen de nombreuses têtes. Les prochains examens devront garder un **nouveau jeu de compositions non consultées jusqu'ici** séparé dès la conception, et mesurer le bénéfice polyphonique avant de modifier des modèles de production.
- Les 2 corrections exactes sont conservées dans l'archive du rejeu et ne constituent jamais une condition ou un masque de prédiction. Tous les scores doivent être rapportés avec l'effet K0–K6 et les coûts de calcul (S35/S37 utilisent encore des sorties pré-calculées, donc aucune économie réelle mesurée).
