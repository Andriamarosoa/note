# Architecture proposée : têtes historiques + actions + sélection NEURONALE

**Date : 8 octobre 2026.**

## Intention

L'ensemble du travail historique doit être conservé. Aucun modèle, expérience
acoustique, hypothèse, correcteur, fix ou régression n'est éliminé simplement
parce que son bilan global est mauvais.

**Le routeur et sa politique d'action sont appris par un réseau neuronal** ;
aucun audit ne fixe directement une pondération immuable d'une tête.

Deux principes sont distincts :

1. **Conservation des pistes** : registre automatique de tous les scripts
   et documents d'expérience sur toutes les branches Git récupérées,
   complété par les commits historiques explicites de correction/fix.
2. **Activation scientifique** : une piste n'est utilisée par le réseau
   sur Exact-K qu'après adaptation avec sortie compatible, vérification de
   ses prédictions hors-fold, audit de ses corrections/régressions,
   et respect du même horizon temporel/dataset.

Les entrées du registre sont des **candidats**, pas autant de réseaux
entraînés. Le premier inventaire recensait 9 497 occurrences de sources
à travers les branches. Un passage complémentaire a recensé **166 commits
explicites de fix/correction** parmi **9 665 entrées de provenance**
après déduplication par identifiant de provenance, pas par hypothèse unique.
Ces totaux peuvent évoluer avec l'historique Git.

## Correction importante : une sélection n'est pas ouverte à tous les K

La version précédente produisait un seul poids par tête, identique pour
toutes les classes. **C'était une erreur de conception.**

Le nouveau `ClassConditionalAuditSelector` produit une matrice
**[événement, tête, K0..K6]** et un modèle distinct pour KEEP :

- `M(x,h,k)` : compatibilité *structurelle* d'un adaptateur avec une
  proposition vers Kk (la compatibilité est définie sans vrai K).
- `S(x,h,k)` : score **appris** de cette tête pour ce K, alimenté par
  les activations acoustiques, les autres têtes et ses audits OOF.
- `W(x,h,k) = softmax_h S(x,h,k)` pour les têtes compatibles ;
  `W=0` pour les couples tête-K structurellement non compatibles.
- Les représentations et probabilités pondérées sont calculées séparément
  pour chaque K ; le décodeur neuronal propose K0–K6 ou KEEP.
- KEEP dispose de sa **propre attention** ; les têtes de conservation/fix
  n'ont pas à prétendre reconnaître une classe K donnée.

Exemple : pour une prédiction originale **K2**, l'adaptateur `C23`
peut encore voter **pour une hypothèse K3**. Il est impossible de savoir
que le vrai K est K3 à l'inférence ; nous ne filtrons donc **jamais**
les têtes sur la classe réelle et ne verrouillons pas toutes les
sélections sur le K initial de la référence.

Le masque représente seulement la **possibilité structurelle** d'une
proposition. Une tête éligible n'obtient aucun privilège garanti :
**l'utilité est entièrement apprise par le réseau**. Le masque
`head_mask` continue de contrôler la provenance et l'alignement OOF
d'une tête, indépendamment du masque `head_k_mask`.

La version actuelle garde H0 comme contexte de repli pour K0..K6,
H1–H5 avec sorties poly K2..K6, C23 uniquement pour une proposition
K3, C32 pour K2, C34 pour K4, C43 pour K3, et les quatre fixes
uniquement pour la branche de KEEP. Cela correspond aux capacités
actuelles des adaptateurs et **pas à un jugement statique sur leur
précision**. Les autres anciennes pistes, une fois raccordées,
pourront définir d'autres compatibilités.

Les audits par vrai K sont produits sur les folds d'entraînement ;
au moment de prédire, le vrai K n'est jamais disponible.
Le rapport expérimental exporte désormais les poids conditionnels
de dimension `[7493, 14, 7]`, et non seulement la moyenne
`[7493, 14]`. Une amélioration éventuelle doit être mesurée
sur les 59 309 cas natifs complets.

## Contrat universel des têtes

Chaque module est accompagné de :
- identité stable et source commit/branche ;
- type `feature`, `predictor`, `correction` ou `fix` ;
- statut d'adaptateur, présence et qualité des prédictions OOF ;
- distribution de propositions d'action K0..K6 et **KEEP** ;
- masquage des modules absents/non alignés ;
- audit hors-fold par vrai K : actions, corrections, régressions, neutres ;
- contexte audio observable, latence et contraintes de causalité.

### Apprentissage

Le réseau `LearnedAuditSelector` est un modèle **TensorFlow/Keras** :

- encodeur partagé et embeddings de type de tête ;
- **attention multi-têtes**, permettant de modéliser les interactions ;
- sélection neuronale `softmax` par événement, masquée pour les
  têtes inaccessibles ou non pertinentes ;
- décodeur neuronal non linéaire sur les interactions et le contexte ;
- choix d'action `K0..K6` ou `KEEP` ;
- tête auxiliaire qui apprend la probabilité de correction et de
  régression de chaque spécialiste sur la base de labels **OOF
  d'entraînement**.

Le gradient d'entraînement apprend comment utiliser le bilan historique
et les interactions. Les coefficients ne sont pas choisis manuellement
à partir de `corrections - régressions`.

Le vecteur d'audit par vrai K est un **descripteur d'entraînement**.
Le vrai K évalué n'est jamais fourni à l'inférence. Pour les têtes
actives, les audits de chacun des folds intérieurs sont calculés sur
les *autres* folds, et l'audit du fold externe tenu à part provient
uniquement des folds d'entraînement.

Une nouvelle combinaison peut ainsi avoir une valeur non additive.
Deux spécialistes isolément faibles peuvent produire une bonne
prédiction ensemble : l'attention et le décodeur non linéaire peuvent
apprendre cet effet sans énumérer toutes les combinaisons
`2^nombre_de_têtes`.

## Preuves de fonctionnement

[GitHub Actions : inventaire et entraînement synthétique du routeur](https://github.com/Andriamarosoa/note/actions/runs/37745171794)
**réussi**.

Quatre tests ont passé, incluant un problème artificiel XOR : deux
têtes individuellement insuffisantes deviennent utiles uniquement par
combinaison ; le nouveau réseau apprend cette interaction.
Les tests vérifient aussi le masquage, la normalisation de la sélection,
la disponibilité des gradients et l'interdiction d'activer un fix
historique sans prédictions OOF alignées.

[GitHub Actions : registre enrichi des correctifs historiques](https://github.com/Andriamarosoa/note/actions/runs/37745637974)
(génération du registre terminée avec succès).

## Première évaluation native d'une version raccordée

[GitHub Actions : 14 têtes/acteurs, full Exact-K K0–K6](https://github.com/Andriamarosoa/note/actions/runs/37745469151).

La première version d'évaluation raccorde :
- **6 sorties** déjà disponibles et alignées H0–H5 ;
- **4 nouvelles têtes d'action** offrant les transitions K2→K3,
  K3→K2, K3→K4, K4→K3 ;
- **4 têtes de fix/conservation** pour K2, K3, K4 et abstention.

Les actions/propositions dépendent des probabilités observées, mais
leur sélection et leur combinaison sont **apprises**.
Ces huit têtes sont des **adaptateurs expérimentaux**, pas une
reproduction en cachette des anciennes versions historiques.

Le protocole reprend les **59 309 événements natifs**, 7 493 candidats
basés exclusivement sur les prédictions originales K2/K3/K4 ;
folds 0/1/2/4, **player05 et fold3 exclus**, sans modification
de `freeze_local_combo`. Le spectrogramme observe jusqu'à +160 ms
d'audio futur et n'est pas encore équivalent au runtime causal.

Ne pas présenter les nouvelles valeurs d'Exact-K comme confirmées
avant le succès de ce run et l'analyse des régressions par vrai K.

## Piste de développement

Activer progressivement les autres têtes **réelles** (expériences de
V8 à V28, compression, pitch-shift, morphologie, énergétiques,
corrigeurs, freezes et éventuelle représentation YourMT3+) par un
adaptateur qui produit des propositions et des audits alignés.
Les modules à provenance ou cible incompatible restent inscrits
mais masqués tant qu'ils ne peuvent être évalués loyalement.

Un registre complet ne signifie pas que les anciennes versions
produisent magiquement des logits compatibles avec la cohorte
actuelle. Ceci exige des réexécutions ou des adaptateurs explicites.

## Fichiers

- `scripts/build_v273_head_history_registry.py`
- `scripts/learn_v273_neural_head_selector.py`
- `scripts/evaluate_v273_neural_history_mix.py`
- `test/test_v273_neural_history_selector.py`
- `.github/workflows/v273-neural-historical-selector.yml`
- `.github/workflows/v273-native-neural-head-mix.yml`
