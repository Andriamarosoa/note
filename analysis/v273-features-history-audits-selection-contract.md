# V27.3 — Caractéristiques, historique des audits et sélection des têtes

**8 octobre 2026 — audit de connexion, pas promotion de modèle.**

## Verdict sur la dernière version exécutable

Dans \`scripts/learn_v273_transition_combo_risk.py\`, le réseau reçoit :

1. **Des caractéristiques acoustiques globales** (les 26 premières
   colonnes actuellement disponibles correspondant aux familles
   \`spectral__\`, \`birth__\`, \`persistence__\`, \`damping__\`).
2. **Le K initial** (one-hot) et **le K proposé**.
3. **Des propriétés de chaque combinaison** : marge « candidat contre
   KEEP », probabilités KEEP/candidat, signe du vote, proportion de
   têtes dans le sous-ensemble et présence d'un correcteur spécialisé.
4. **Quatre statistiques d'audit d'entraînement par combinaison et
   direction** : \`log1p(nombre de propositions)\`, taux de correction,
   taux de régression et taux neutre lissés. Chaque audit est calculé
   sur les autres folds internes lors de l'entraînement, ou sur
   l'ensemble du train externe lors de la prédiction externe.
5. Les caractéristiques KEEP des 14 adaptateurs disponibles.

Le sélecteur calcule effectivement
\`state = subset_embed(concat(combination_features, base_K,
target_K, acoustic_context))\` puis un softmax conditionnel
sur les **combinaisons réellement disponibles**.

**Mais il ne reçoit pas encore un vecteur explicite de la composition
H0/H1/.../Cxy**. Il connaît la taille du sous-ensemble, et les
votes agrégés, mais **pas directement l'identité de chaque tête**.
Deux combinaisons différentes avec mêmes votes, mêmes audits lissés
et même taille peuvent donc être difficiles à distinguer.

## Distinction obligatoire : les audits passés et les audits connectés

Le projet possède une longue série d'audits historiques, dont :
- erreurs et corrections par vrai K2/K3/K4, sur/sous-comptage ;
- phénomènes de fondamentale/harmonique et audit K3 des candidats ;
- attaques, durée, persistance et morphologie de trajectoire ;
- pitch-shift/compression, transposition en demi-tons ;
- cluster A/B (référence de l'analyse ancienne), sous-clustering ;
- tests des correcteurs C23/C32/C34/C43 et de la branche KEEP ;
- exploration de YourMT3+, énergétique/flux.

**Ces conclusions ne sont pas encore toutes représentées par des
caractéristiques numériques indépendantes à l'entrée du réseau.**
Le catalogue de commits/modèles historiques n'implique pas que leurs
prédictions ou audits deviennent des têtes exécutables.

Le niveau de connexion réel est :

| Famille d'information | Dans le réseau actuel ? | Manque pour l'apprentissage audit-aware |
|---|---|---|
| Contextes spectral, naissance, persistance, amortissement | Oui, partiellement | Des attributs de morphologie et d'harmonicité localisés et nommés |
| Correction/régression des sous-ensembles pour K source/cible | Oui, audit OOF de 4 statistiques | Profil par sous-population, effectif, incertitude, stabilité fold à fold |
| Identité exacte H0..H5/Cxy d'un sous-ensemble | Non, seulement taille/présence du correcteur | Masque de composition à 14 bits comme entrée explicite |
| Audits A/B et sous-clusters de défaillance | Non | Recalcul de profils exclusivement sur train du fold ; ne pas injecter l'ancien cluster labellisé sur holdout |
| Harmoniques vs fondamentales, faux candidats | Signaux indirects via H3/H4 | Descriptions d'ambiguïté disponibles au moment de l'inférence |
| Morphologie temporelle, attaque, extinction | En partie via famille acoustique | Mesures dédiées, si disponibles, et OOF d'efficacité de la correction dans chaque régime |
| Pitch-shift et stabilité de la cardinalité | Non, pas de tête autonome | Adapter réellement exécutable et audit OOF, avant tout vote |
| Tests historiques de correcteurs sans gains | Archivés, pas numériquement liés | Recalcul indépendant sur train, pas copie d'un score de fold d'évaluation |
| F_keep2/F_keep3/F_keep4/F_keep_any | Caractéristiques KEEP agrégées | Conditionnement de leur bénéfice sur contexte et K initial |

## Ce que doit être le vrai apprentissage des sélections

Pour chaque événement x, combinaison S, K initial b, cible k,
le routeur doit disposer de :

- \`bitmask(S)\` : identité des têtes actives ;
- \`audio(x)\` : spectre, fondamentale/harmonique, naissance, persistance,
  morphologie, etc. réellement mesurés ;
- \`outputs(S,x)\` : votes des têtes et contradictions ;
- \`audit_OOF(S,b,k,regime(x))\` : corrections, régressions, neutres,
  couverture, intervalle/incertitude statistique et stabilité des folds,
  **calculés uniquement sur les données de TRAIN** ;
- \`risk(S,b,k,x)\` : bénéfice et dommage prédits par un réseau.

La sélection doit être contextualisée, sans seuil global ni poids
d'audit figé :

\`W(S|x,b,k)=softmax_{S éligibles} fθ(audio(x),bitmask(S),
  outputs(S,x),audit_OOF(S,b,k,regime(x)))\`.

La décision finale est produite par un arbitre KEEP/correction qui
reçoit les risques de la sélection. Cela permet à la même tête d'aider
K3 sans être autorisée d'office à modifier un K2 correct.

## Garde-fous nécessaires

- **Ne jamais injecter directement les chiffres des audits antérieurs
  sur les folds 0/1/2/4 dans le modèle évalué sur ces mêmes folds** :
  cela divulguera le résultat et créera un biais de sélection.
- Garder le vrai K strictement dans les étiquettes d'entraînement et
  les rapports, jamais dans la décision ou l'activation d'une tête.
- Une hypothèse comme « cluster B plus difficile » n'est pas un signal
  numérique à elle seule. Il faut pouvoir affecter un nouveau son
  à un régime **par ses caractéristiques acoustiques**.
- Vérifier l'influence réelle des audits par interventions sur les
  caractéristiques **à poids du modèle fixé**, et pas seulement en
  inspectant la forme d'un tenseur.
- Vérifier ensuite la robustesse sur des données nouvelles non exposées.
- La référence \`freeze_local_combo\` reste inchangée.

## Vérification instrumentée

Script :
\`scripts/audit_v273_selection_feature_influence.py\`.

Il reproduit exactement le modèle évalué et mesure l'effet d'une
permutation ou neutralisation des audits par sous-ensemble, puis d'un
mélange des caractéristiques acoustiques. Il ne change aucune
référence et ne prétend pas que ce test est une validation indépendante.

[Run d'audit d'influence](https://github.com/Andriamarosoa/note/actions/runs/37757350325).
