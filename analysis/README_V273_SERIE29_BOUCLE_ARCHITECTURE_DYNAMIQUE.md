# V27.3 — Série29 : ordonnanceur réellement dynamique, avec réévaluation après chaque sélection

**Résultats vérifiés** — Run principal : https://github.com/Andriamarosoa/note/actions/runs/37865173636. Audit indépendant des routes, corrections et coûts : https://github.com/Andriamarosoa/note/actions/runs/37865318652. Archive permanente avec les 38 modèles d'ordonnancement, 59 309 décisions, routes, compteurs et 21 événements corrigés/régressés : https://github.com/Andriamarosoa/note/releases/tag/v273-series29-research-37865318652. Protocole d'abord déposé dans [README_V273_SERIE29_SCHEDULER_ETAT_PAR_ETAT.md](README_V273_SERIE29_SCHEDULER_ETAT_PAR_ETAT.md).

## Ce qui a été implémenté

Deux têtes **réellement disponibles** : A (prédiction des probabilités K0–K6) et B (compatibilités des sept classes). Un **ordonnanceur à état** voit les 90 premières caractéristiques signal/votes, la distribution K courante, le message B déjà produit, sa variation depuis la dernière analyse et l'étape. À chacune des trois étapes possibles, il peut choisir :

- `STOP` : conserver l'état sans nouvel appel ;
- `A` : recalculer **seulement A** sur le message B en mémoire ;
- `B→A` : recalculer B puis A sur le nouvel état.

**Un choix est effectué de nouveau après l'action**, donc un événement peut suivre `A→STOP`, `B→A→STOP`, `A→(B→A)→A`, `(B→A)→(B→A)→(B→A)`, etc. Les réseaux sont évalués **uniquement sur les événements choisis** ; le coût de 0,31 % n'est pas un artifice de masquage après calcul.

La politique n'est pas le réseau final multi-têtes complet. Il s'agit d'une boucle opérationnelle sur A/B dont les poids sont ceux de la série 20 ; les 38 modèles Ridge (2 types d'action × 19 morceaux exclus tour à tour) apprennent, sur 7 états explorés par événement de formation, `gain attendu = corrections − λ·régressions − coût`. Les morceaux évalués sont exclus du fit des modèles de porte, et les vraies classes ne participent à aucune décision à l'inférence. En revanche, les poids des producteurs S20 et le parent S18 ont déjà servi à des recherches sur cette cohorte : les scores ne sont **pas une validation sur compositions entièrement inédites**.

## Meilleur résultat exploratoire, réglage préannoncé λ=2 / seuil=0,02

| Statistique | Série18 (référence) | Série29 |
|---|---:|---:|
| Exact-K global | 82,9183 % | **82,9402 %** |
| Exact-K poly | **40,5958 %** | **40,5958 %** |
| Corrections nouvelles vs S18 | — | **17** |
| Régressions nouvelles vs S18 | — | **4** |
| Gain net global | — | **+13** |
| Appels A / événement | 0 pour corrections supplémentaires | **677/59 309** |
| Appels B / événement | 0 pour corrections supplémentaires | **599/59 309** |
| Appels A+B relatifs à S20 4A+3B par événement | — | **0,31 %** |

**L'économie est en unités d'appels de têtes, pas en millisecondes, et exclut le coût de la porte Ridge et de l'extraction de caractéristiques**. Elle reflète aussi un choix très abstentionniste : la politique conserve S18 sans nouvel appel sur la grande majorité des événements. L'économie totale du pipeline, qui doit déjà produire S18, n'a pas été mesurée.

Répartition exacte des **59 309 premiers choix** :
- STOP immédiat : **58 912** événements ;
- A d'abord : **49** événements ;
- B→A d'abord : **348** événements.

Parmi les 397 événements où une première action est exécutée, 142 reçoivent une deuxième action et 138 une troisième. Les chemins sont multiples ; tous les chemins et les actions par événement sont dans `routes.npz`. Sur les 21 changements qui touchent une ancienne décision correcte ou corrigent une ancienne erreur, 17 sont des corrections et 4 des régressions ; le CSV `all_21_corrected_or_regressed_cases.csv` en conserve **chaque ID, vrai K, ancien K, nouveau K et chemin**.

## Contrôles/limites

- Série29 : 9 stratégies λ∈{1,2,4} et seuil∈{0,0,005,0,02}. Meilleur compromis observé λ2/seuil0,02 ; **aucune stratégie à gain positif sans ancienne correction détruite**.
- S27 avait déjà prouvé que le repassage complet n'était pas toujours nécessaire : arrêter sur stabilité avec seuil 0,02 réduisait à 59,86 % les appels A+B avec **zéro différence d'argmax sur cette cohorte**, mais les distributions de probabilités changent légèrement.
- S28 montrait qu'un ordonnanceur au début de l'événement obtenait 82,9419 % global / 40,5958 % poly, avec 16 corrections et 2 régressions. S29 démontre **le choix réellement étape par étape**, sans amélioration strictement supérieure aux meilleurs candidats précédents.
- **La série18 n'est pas modifiée ni promue** : 2 934 corrections conservées sur les 59 309 événements face à freeze, et 2 210 régressions déjà historiques.
- Des morceaux réellement **inédits depuis le début de la conception** sont nécessaires avant d'interpréter la faible régression (4) comme une assurance générale. Les évaluations successives de la cohorte ont influencé les choix de l'architecture.

## Décision de conception retenue

**Oui : conserver le scheduler réentrant à état (choix A / B→A / STOP à chaque étape) comme prototype d'architecture pour l'ajout futur des têtes harmoniques, morphologiques et d'attaque.** Ne pas remplacer S18 maintenant. Le prochain problème vérifiable : isoler les **quatre** régressions enregistrées sans utiliser leurs vraies étiquettes dans le contrôle ; ajouter des variables de preuve musicales fiables, entraîner sur des compositions distinctes, puis évaluer sur un ensemble inédit verrouillé. Des règles ajustées aux quatre IDs seraient une fuite d'évaluation et ne sont pas acceptables.
