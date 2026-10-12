# S140–S146 — Récupération de producteurs historiques, tests WAV réels et verrou S41

**Statut : travail expérimental sur `codex/v273-scope-rescue`; S80 officiel NON promu.**
La référence `freeze_local_combo` n'est pas modifiée.

## Nouveauté majeure : vrais checkpoints et code historique retrouvé

Le [run S140](https://github.com/Andriamarosoa/note/actions/runs/38192579770)
a récupéré **six fichiers Python strictement identiques** au commit
`854ada718ac9bf1556f00c96b8e1aa90402d6ab3`.
Les six selftests originaux S12 / S20 / S24 / S27 sont passés.

Les workflows historiques permettent d'accéder aux poids déjà entraînés :

| Source | Run d'origine | Modèles authentiques exécutés sur GOAT | Portée |
|---|---|---:|---|
| S12 logistic LOPO | `37853979683` | 19 | sorties brutes du classifieur, **masque historique S9 absent** |
| S38 quatre experts | `37880386031` | 16 | K1–K6 ; source 335/825D |
| S40 CNN temporel | `37880762289` | 4 | K1–K6 ; trajectoire 42×49 |
| S41 fusion six actions | `37881191015` | 19 | routeur original, 377D |
| **Total** | | **58 checkpoints** | **pas les 16 familles S45 complètes** |

Ils proviennent des véritables exécutions de recherche, sans réentraînement
sur les exemples GOAT. Les entrées audio originales S97 donnent
58D summary, 245D flow, 490D morphology et 42×49D trajectory.
Le vrai `vote32` S11 utilisait notamment le parent S18 et ses
décisions globales/poly. Pour cette prévalidation sur nouvel audio,
les scripts prennent une **nouvelle règle documentée** :
`b = freeze_local_combo ; g = étudiant S135 ; p = PR16 politique #22`.
**C'est une adaptation**, pas la reproduction bit-exacte du S18/S45 original.

## Résultats réels, sans lire les MIDI

| Test | Exécution vérifiée | Résultat sur 57 événements |
|---|---|---|
| [S141](https://github.com/Andriamarosoa/note/actions/runs/38192776863) | 4 vrais checkpoints S40 | moyenne brute K0..6 : **[0,55,2,0,0,0,0]** |
| [S142](https://github.com/Andriamarosoa/note/actions/runs/38192916733) | S38 `morph825__extra_trees`, 4 vrais checkpoints | appliqué à K1+ : **[44,11,2,0,0,0,0]** |
| [S143](https://github.com/Andriamarosoa/note/actions/runs/38193017646) | S12 19 modèles logistiques historiques | moyenne brute : **[46,9,2,0,0,0,0]** |
| [S144](https://github.com/Andriamarosoa/note/actions/runs/38193181798) | Chaîne complète **S38×16 → S40×4 → S41×19** poids originaux | **[44,8,4,1,0,0,0]** : 57 KEEP et **0 modification** |

Les scripts S141–S144 produisent des probabilités et des vecteurs
décisionnels `row_id` vérifiés sur le nouvel audio. Aucune de ces sorties
ne prouve la fidélité aux **anciens** 16 champions S45 ; ces derniers
ont été choisis a posteriori sur des tableaux historiques exposés.

## [S145 — diagnostic précis des 57/57 KEEP](https://github.com/Andriamarosoa/note/actions/runs/38193316422)

- 44 groupes ont un parent K0 ; la politique originale S41 K1–K6
  **ne peut pas les modifier**.
- 13 groupes ont un parent K≥1.
- 47 groupes présentent au moins une proposition alternative
  parmi les cinq experts, mais **seulement 3** d'entre eux sont éligibles
  au changement K1–K6.
- Sur les **3 cas éligibles** possédant une alternative, le sélecteur
  original S41 agrégé vote KEEP **3 fois** et accepte **0** alternative.
- Poids moyen KEEP **0,91268078** ; marge médiane KEEP
  sur la meilleure proposition alternative **0,96386892**.
- Le refus vient donc du **routeur actuel**, et pas uniquement
  de l'absence de propositions. Impossible d'attribuer avec certitude
  cette domination au changement de parent sans un test de contrôle
  dans le même domaine.

**Attention** : ces 57 groupes GOAT ont fait l'objet d'un dévoilement
MIDI au test S137. Aucun réglage futur fondé sur leur exactitude
ne pourra servir ensuite de validation indépendante. S145 n'a pas lu
leurs annotations, seulement les postérieurs déjà gelés.

## Verrous à lever pour l'authentique S80

1. **S18 d'origine sur audio inédit**, incluant les votes S11/S5
   globaux et polyphoniques d'origine (pas substitués par S135).
2. **16 champions S43/S44** : les 12 autres familles non opérationnelles
   sur GOAT et l'équivalence historique des familles S12/S38/S40/S41.
   Familles originales : S1 acoustic, S2 source, S3 repair, S4 repair,
   historical repair, S25 Bfirst, S24 Bfirst, S40 CNN, S19 recurrent,
   S11 votes, S38 morphology, S12 logistic, S20 repeated ABA,
   S27 early stop, S41 fusion, S35 multiroute.
3. Exécuter les risques S45, dont les 304 poids Ridge historiques
   et la règle prédéfinie d'agrégation entre morceaux.
4. Alimenter S46 (193 décisions) puis S69/S70/S71/S72 en
   **prédiction label-free de morceaux inconnus**.
5. Appliquer S80 avec sa règle de déploiement K1 et les routes K0/K2.
6. **Figer les sorties**, puis évaluer un corpus indépendant complet,
   [IDMT-SMT-GUITAR_V2](https://zenodo.org/records/7544110),
   contre `freeze_local_combo` en Exact-K global, K0..6, poly,
   avec corrections/régressions appariées.

Un remplacement partiel des parents crée nécessairement une nouvelle
version `S80-deploy-v1` : l'ancien 83,09 % global / 43,87 %
poly ne devient **pas** son score garanti.

**Décision** : Aucun de S140–S145 ne franchit la porte de promotion.
Les 58 vrais checkpoints constituent un progrès important en
disponibilité technique, mais leur existence seule ne suffit pas
à établir une supériorité indépendante.

*Préservation des poids originaux lancée via le
[workflow S146](https://github.com/Andriamarosoa/note/actions/runs/38193466752).*