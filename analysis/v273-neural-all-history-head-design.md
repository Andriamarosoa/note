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
