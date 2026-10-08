# V27.3 — Un score commun aux groupes qui proposent le même verdict

Protocole fixé le 8 octobre 2026 après l’audit de group127, avant tout entraînement de ces deux variantes. Il s’agit d’une nouvelle expérience de développement, sans validation indépendante ni remplacement de référence.

## Constat mesuré avant modification

Dans le bras global archivé à +57, 942 des 7 493 fragments éligibles ont au moins un même verdict proposé par plusieurs groupes avec des gains estimés de signes opposés. L’écart médian entre probabilités de correction extrêmes d’un même verdict est de 0,0944, soit 9,44 points. Sur 375 fragments, la somme des probabilités maximales de correction des destinations distinctes dépasse même la probabilité annoncée que la référence soit fausse.

Le décodage sur les 127 groupes ajoute 66 corrections mais 77 régressions par rapport aux singletons du même réseau. Dans 75 de ces 77 régressions, le mauvais verdict retenu était déjà proposé par un singleton. Ce constat motive le test d’un regroupement des estimations par destination ; il ne démontre pas que celui-ci suffira à corriger les erreurs.

Un diagnostic sans réentraînement, utilisant la moyenne des probabilités de correction des groupes de chaque destination, donne +40 / +69 / +58 pour direct / global / local. Ces résultats sont explicitement exposés et ne sont pas une validation indépendante. Les nouveaux réseaux ci-dessous font une moyenne des **logits avant probabilités** et sont entraînés de nouveau : ce diagnostic n’est pas leur résultat.

Les données d’audit et les 143 changements de justesse du bras global sont conservés dans `evidence/v273-group-consensus/duplicate-audit.json` et `global-groups-vs-singletons.csv`.

## Deux contrôles fixes

Les deux variantes utilisent les entrées du bras **global seulement**, vérifiées identiques bit à bit à son archive, et le cache commun de 14 producteurs du run 37777844182. Les exclusions imbriquées des producteurs/audits et les 127 propositions restent identiques. Aucun vote ni proposition n’est supprimé parce qu’un membre se trompe seul.

Les 127 groupes passent chacun dans le réseau non linéaire partagé existant (64 puis 32 neurones). Le logit conditionnel d’une destination est la moyenne des logits des groupes qui proposent cette destination. Tous ces groupes reçoivent ensuite la même probabilité de correction et le même gain. Les groupes différents continuent donc de fournir des observations distinctes au réseau ; on ne remplace pas leurs caractéristiques par le seul K.

| Variante | Transformation finale | Perte conditionnelle |
|---|---|---|
| `pooled_bce` | Sigmoïde du logit moyen par destination | Exactement la BCE antérieure, moyennée sur les groupes changeants par événement |
| `pooled_ce` | Softmax des logits moyens des destinations changeantes disponibles et de OTHER | Une vraisemblance catégorielle par événement où la référence est fausse |

Le premier contrôle isole l’effet du partage de score par destination, en conservant la perte et la pondération antérieures. Le second ajoute simultanément la normalisation entre issues mutuellement exclusives et une perte par événement/destination, au lieu de multiplier les termes d’une destination suivant son nombre de groupes. Il ne permet pas de séparer ces deux derniers effets.

OTHER signifie « aucune des propositions changeantes disponibles n’est juste ». Cette issue est nécessaire : une seule alternative disponible peut être fausse elle aussi. Elle couvre notamment les vrais K0/K1 et les K non proposés. Son logit de référence est fixé à zéro ; les logits des autres issues sont appris relativement à celui-ci. Elle n’est jamais une action exécutable et n’ajoute aucun paramètre.

La probabilité `r` que la référence soit correcte est toujours apprise une seule fois par fragment, avec la même branche de réseau. Pour un groupe qui propose K différent de la référence :

```text
P(correction) = (1-r) q_K
P(régression) = r
P(neutre) = 1 - P(correction) - P(régression)
gain = P(correction) - P(régression)
```

Dans `pooled_ce`, les q des destinations disponibles et OTHER somment à 1. Dans `pooled_bce`, les sigmoïdes restent indépendantes : aucun simplexe entre classes n’est revendiqué. Les groupes conservant la référence ont exactement [0,0,1] et un gain nul dans les deux cas.

La décision conserve le meilleur gain strictement positif, sinon KEEP. Parmi plusieurs groupes de même K et même gain, le premier masque sert de représentant déterministe. **Ce masque n’est pas une attribution causale** : tous les groupes de cette destination ont alimenté son estimation. Aucun seuil n’est ajusté sur les folds évalués.

## Comparaison et critères de lecture

Budget inchangé : **12 994 paramètres**, 30 epochs, seed 27402, Adam 0,002, batch 192, aucune pondération de classes. Aucune recherche de seed, de seuil, de voisinage ou de fusion après résultat. La fusion définissant les propositions reste la moyenne fixe des votes ; seule la fiabilité finale est modifiée.

Comparer chaque variante à `freeze_local_combo`, au bras global archivé (+57), et à son décodage singleton (+68). Refaire aussi le décodage singleton de chaque nouveau réseau, sans le présenter comme une ablation réentraînée. Publier les scores global/poly, K0–K6, chaque fold, corrections/régressions, gain annoncé/observé et récupération des 16 vrais cas où aucun singleton ne peut réussir. Publier les deux variantes même si elles sont négatives.

Le test ne suppose pas qu’une probabilité normalisée soit calibrée. Le diagnostic vérifiera la disparition des contradictions de score entre groupes de même verdict et, pour `pooled_ce`, le respect du budget total de probabilités. Une amélioration de ces propriétés sans amélioration des décisions sera rapportée comme telle.

Les tests couvriront égalité des scores de même K, invariance à l’ordre des groupes, cas OTHER et absence d’alternative, complémentarité sans veto, vraisemblance catégorielle, gradients et entraînement court réel. Les contrôles existants d’exclusion des labels restent exécutés. Un vérificateur séparé recomputera les résultats depuis les archives.

## Périmètre

59 309 événements natifs, 7 385 vrais K≥2, 7 493 fragments initiaux K2/K3/K4 modifiables. Folds 0/1/2/4 déjà exposés, fold3/player05 exclus ; contexte jusqu’à +160 ms inchangé. Un seul seed, pas de conclusion de significativité. La référence reste gelée jusqu’à validation indépendante.
