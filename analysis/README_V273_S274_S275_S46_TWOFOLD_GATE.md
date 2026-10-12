# S274/S275 — Quorum de deux folds pour les propositions S46

**Branche de recherche** : codex/s274-s46-twofold-audit  
**Référence officielle inchangée** : freeze_local_combo  
**Promotion S80** : NON

## Hypothèse testée

Le mécanisme S219 considère les folds silencieux comme des abstentions.
Une seule famille de folds active peut ainsi provoquer une modification S46.
S274 teste, sans aucun réentraînement ni consultation des XML IDMT, une
variante qui exige le soutien d'au moins deux folds distincts dont les
parents S18 sont compatibles. Si cette condition manque, le K S18 est conservé.

L'audit S255 historique IDMT (dont les 16 WAV sont désormais exposés)
avait constaté 0 correction et 9 régressions lors des 9 modifications
S18 vers S46 polymax appariées XML. Les données IDMT déjà ouvertes ne
peuvent pas servir à optimiser la règle puis à la valider indépendamment.

## Résultat S274 — GitHub Actions 38315939252, réussi

Le premier job a figé la nouvelle proposition sans accéder à la vérité.
Il a compté, sur 59 309 événements GuitarSet et 193 politiques S46 :

- 209 540 interventions initiales (cellules politique × événement)
- 94 717 interventions supprimées faute de deux folds
- 114 823 interventions conservées
- 2 646 événements affectés
- SHA-256 du candidat : 5fecdfbe73882d5123f4495567fbd3e1a4ab2b2c6166945ab0316eeb479c1d2f

Le second job a évalué les décisions S46 AVANT S69 sur le GuitarSet
historique. Gains et pertes ci-dessous sont relatifs à S18.

| Sélection | Corrections globales | Régressions globales | Bilan global | Bilan poly K2+ |
|---|---:|---:|---:|---:|
| S46 polymax S219 | 243 | 1118 | −875 | −47 |
| S46 polymax S274 | 163 | 643 | −480 | −20 |
| S46 balanced S219 | 169 | 780 | −611 | −77 |
| S46 balanced S274 | 137 | 466 | −329 | −22 |

**La règle S274 réduit les régressions MAIS les bilans restent négatifs.**
Cet audit utilise les données historiques GuitarSet vues à l'entraînement
par trois des quatre folds ; aucun score ici n'est OOF ou indépendant.

## S275 — Rejeu de bout en bout

Les scripts S275 adaptent uniquement les décisions S46 au contrat des
runtimes historiques puis réexécutent S69, S70, S71, S72, S79 et S80
avec les paramètres originaux inchangés. Les prédictions sont figées
sans étiquettes dans un premier job ; un second job évalue l'Exact-K
global et poly sur les mêmes événements GuitarSet et compare au S272
original complet (79,24 % exploratoire).

Cette comparaison restera un diagnostic de développement biaisé et
ne prouvera pas que S80 généralise mieux sur des compositions inédites.

## Conditions impératives pour toute promotion ultérieure

- Analyser le comportement après les étapes S69, S70, S71/S72 et S80,
  pas seulement la baisse des erreurs sur la sortie brute S46.
- Ne jamais utiliser les XML IDMT déjà ouverts pour choisir la règle
  puis revendiquer sur ces mêmes données une validation indépendante.
- Geler poids, décisions, seuils, WAV et identifiants d'événements AVANT
  une nouvelle évaluation sur un corpus disjoint par composition avec
  vérité K0–K6 complète.
- Mesurer corrections, régressions par K, bootstrap groupé par morceau,
  comparaison à freeze_local_combo et robustesse du gain.

**Le déploiement S274/S275 est purement expérimental ; aucun changement
de la référence officielle n'est autorisé par cet audit.**
