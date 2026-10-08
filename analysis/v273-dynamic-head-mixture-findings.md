# Recherche V27.3 — réseau de têtes à combinaisons dynamiques

**8 octobre 2026. Prototype réellement implémenté et évalué sur Exact-K global. Aucune promotion.**

## Motivation et architecture

Les multiples expériences acoustiques (spectre, énergie/flux, harmonique,
persistance, extinction, contexte, transposition, morphologie) ne doivent
pas être réduites à un « meilleur correcteur » unique.

Proposition : **Mixture of Experts conditionnel** avec deux fonctions
séparées :

1. **Routeur** : apprend en fonction du contexte du signal les poids d'un
   mélange de probabilités d'experts. Plusieurs têtes peuvent contribuer
   simultanément, et leurs poids varient événement par événement.
2. **Arbitre de risque (étape suivante, pas encore implémentée)** : apprend
   quand une correction est vraisemblablement bénéfique et quand la
   prédiction intacte `freeze_local_combo` doit être conservée.
   Une correction n'est pas garantie sans régression sur des labels inconnus.

Aucune combinaison fixe du type 1+2+8 n'est codée en dur. Un réseau
`tanh` à 24 neurones cachés produit une distribution `softmax`
sur les têtes; leurs sorties sont fusionnées comme un mélange convexe.
Le modèle peut s'abstenir.

## Registre des têtes

| ID | Tête | Statut dans le prototype |
|---|---|---|
| H0 | Prédiction gelée de `freeze_local_combo` | Active ; reste un fallback |
| H1 | État spectral | Active |
| H2 | Naissance, persistance et extinction énergétiques | Active |
| H3 | Harmoniques complètes | Active |
| H4 | Fondamentales seules | Active |
| H5 | Sources candidates / cohérence temporelle | Active |
| H6 | Flux logarithmique / continuité énergétique | Recherché antérieurement ; non branché au routeur global |
| H7 | Morphologie des attaques et des pics | Piste préservée, tête à développer |
| H8 | Transformations pitch-shift / compression de hauteur | Piste préservée, tête à développer |
| H9 | Représentations et décodeur d'événements de YourMT3+ | Piste préservée, intégration à auditer |
| H10 | Couplage physique / flux de sources et amortissement | Hypothèse physique encore à tester, pas une équation de Navier-Stokes résolue |

Les H1–H5 utilisent plusieurs regroupements de caractéristiques proches :
ce ne sont pas cinq modèles *indépendants* au sens scientifique.
Les contrôles « désaccordé » et « trames brouillées » restent des
**contrôles négatifs**, non des têtes à promouvoir.

### Apprentissage sans fuite

Population originale vérifiée : **59 309 événements K0..K6**, folds
0,1,2,4. **7 493** cas éligibles uniquement parce que la référence a
prédit K2/K3/K4; les autres **51 816** lignes sont conservées.

Pour chacun des quatre folds externes :
- Entraîner les cinq spécialistes sur les **autres** folds, exclusivement
  sur leurs vrais K2–K6 pour le signal correctif.
- Fournir au routeur des probabilités des spécialistes **hors-fold**
  sur les trois folds d'entraînement ; ne pas utiliser de prédictions
  apprises sur leur propre ligne pour entraîner ce routeur.
- Adapter le routeur neuronal avec une entropie croisée pondérée par
  classe, sur les vrais événements polyphoniques d'entraînement.
- Réentraîner les spécialistes sur la totalité des autres folds,
  puis appliquer le routeur au fold tenu à part.
- Les sorties du routeur sont restreintes à K2–K6; jamais de proposition
  vers K0/K1. Les autres prédictions originales restent inchangées.

Ce protocole n'utilise pas les labels du fold externe pour entraîner
les têtes, le routeur ni les seuils (ces seuils sont fixés avant le test).

## Résultat expérimental

GitHub Actions :
[37742104638 — Dynamic Specialist Head Mixture](https://github.com/Andriamarosoa/note/actions/runs/37742104638)
**Terminé avec succès** ; compilation, tests synthétiques et audit complet
sur 59 309 événements.

| Politique | Exact-K global | Exact-K poly K2–K6 | Corrections | Régressions | Net |
|---|---:|---:|---:|---:|---:|
| Référence | 81.6976 % | 34.2586 % | — | — | — |
| Routeur sans abstention | 81.7886 % | 34.9898 % | 349 | 295 | +54 |
| Routeur confiance (marge ≥ 0.15) | 81.7751 % | 34.8815 % | 110 | 64 | +46 |
| **Routeur prudent** (marge ≥ 0.25, accord ≥ 3 spécialistes) | **81.7549 %** | **34.7190 %** | **58** | **24** | **+34** |
| **Moyenne fixe prudente** (même contrainte d'accord) | 81.7380 % | 34.5836 % | 40 | 16 | +24 |

**Comparaison dynamique vs moyenne fixe** : le routeur prudent
ajoute **10 bonnes prédictions nettes** en plus de la moyenne fixe,
sur les mêmes 59 309 événements et au même seuil.
Cela ne suffit **pas** à démontrer un apport substantiel, et
les deux politiques ne font pas exactement le même nombre d'actions.

### Résultat du routeur prudent par fold

| Fold | Corrections nettes |
|---|---:|
| 0 | +13 |
| 1 | +7 |
| 2 | +4 |
| 4 | +10 |
| **Total** | **+34** |

### Régression malgré l'abstention

| Vrai K | Gain net routeur prudent |
|---|---:|
| K0 | 0 |
| K1 | 0 |
| K2 | **+55** |
| K3 | **−16** |
| K4 | **−5** |
| K5 | 0 |
| K6 | 0 |

Le routeur apprend une combinaison mais **ne sait pas encore protéger
les K3/K4**. C'est exactement le problème de conception que la future
tête d'arbitrage des risques doit traiter.

Le précédent correcteur **fondamentales seules + protection poly** obtenait
**81.8375 % global / 35.3825 % poly**, mieux que le routeur actuel.
Ce prototype est donc une **preuve d'architecture** et non un nouveau
meilleur système.

### Poids réellement dynamiques

Exemple, **fold 2**, poids moyens conditionnés par la classe initiale :

| Tête | Préd. K2 | Préd. K3 | Préd. K4 |
|---|---:|---:|---:|
| Base | .244 | .090 | .241 |
| Spectral | .043 | .168 | .180 |
| Cycle de vie | .318 | .235 | .155 |
| Harmonique complet | .113 | .127 | .154 |
| Fondamentale | .126 | .266 | .249 |
| Sources | .156 | .113 | .020 |

Ces poids démontrent une politique **non constante** entre groupes, mais
n'identifient pas de manière causale une combinaison optimale.
La variabilité peut aussi dépendre du contexte acoustique et des
biais des modèles entraînés.

## Décision

**Conserver les têtes et le routeur comme infrastructure de recherche** ;
`freeze_local_combo` officiel reste à **81.6976 % global /
34.2586 % poly**, sans modification. Le meilleur compromis de correcteur
précédent reste exploratoire à **81.8375 % / 35.3825 %**.

Pour réduire réellement les régressions, prochaine étape méthodologique :

1. Ajouter une **tête d'arbitrage des risques** estimant séparément
   `P(proposition correcte)` et `P(référence correcte)`, en utilisant
   des prédictions hors-fold des autres têtes.
2. Apprendre des coûts de régression adaptés à K2/K3/K4 **uniquement
   sur les données d'entraînement**, avec une option d'abstention et un
   comparatif de coût des erreurs sur chaque vrai K.
3. Test contrôlé d'ablation du routeur : spécialistes constants,
   routeur seul, arbitre seul, routeur + arbitre ; présenter corrections
   et régressions ainsi que les scores K0 à K6.
4. Rendre H6–H10 disponibles via le même contrat d'entrées sans changer
   silencieusement la population, la provenance ni les fold splits.
5. Validation finale avec un jeu réellement inédit, plutôt que
   réoptimiser sur les mêmes folds déjà regardés. Le joueur 05,
   historiquement consulté, n'est pas une garantie de holdout entièrement
   vierge.

**Limites :** il s'agit d'une classification de *nouvelles attaques*
dans une fenêtre native, pas nécessairement du nombre de notes
simultanément tenues ; caractéristiques post-onset jusqu'à +160 ms ;
le premier prototype n'est pas causal en temps réel et n'a pas
résolu les équations de Navier-Stokes.

## Code

- `scripts/evaluate_v273_dynamic_heads.py`
- `test/test_v273_dynamic_heads.py`
- `.github/workflows/v273-dynamic-head-mixture.yml`
- Artefacts du run 37742104638 :
  `v273-dynamic-head-mixture-results` (sorties pour les 59 309 lignes,
  poids par événement, rapport JSON et tableau).
