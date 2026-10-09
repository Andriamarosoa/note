# S35 — état final du routeur 18 têtes (sans H9) et audit individuel

## Résultat exécuté et indépendamment vérifié

- Entraînement de 144 spécialistes à provenance OOF/nested folds : [run 37868356632](https://github.com/Andriamarosoa/note/actions/runs/37868356632).
- Vérification indépendante des 144 poids archivés, identifiants natifs, 9 politiques, scores et trajectoires sur 59 309 événements : [run 37868618331](https://github.com/Andriamarosoa/note/actions/runs/37868618331).
- Archive permanente : [release S35](https://github.com/Andriamarosoa/note/releases/tag/v273-series35-research-37868618331).
- Audit individuel descriptif supplémentaire de toutes les têtes : [run 37869018988](https://github.com/Andriamarosoa/note/actions/runs/37869018988) ; sources `scripts/audit_v273_s35_head_usage.py`.

**H9 exclus du routage**, tant de la liste d'actions que de ses entrées. Les classes véritables et les sorties YourMT3+ servent à l'audit en dehors de la prise de décision. **H8 pitch-shift ne peut pas être réellement exécutée** faute de prédictions de WAV transformés hors-fold complètes/aligneés ; son contrat masque correctement la tête (zéro appel). Sur le catalogue de 18 modules + ancien A/B, **18 actions distinctes ont été sélectionnées** dans certaines politiques, toutes sauf H8. Les 144 modèles entrainés correspondent à 9 experts sur quatre folds externes et trois internes : ce total ne signifie pas 144 architectures différentes.

## Comparatif S18 et S35

| Politique | Global Exact K0–K6 | Poly Exact K2–K6 | Corrections vs S18 | Régressions vs S18 | Gain net global |
|---|---:|---:|---:|---:|---:|
| S18 référence | 82,9183 % | **40,5958 %** | 0 | 0 | 0 |
| S29 A/B | 82,9402 % | 40,5958 % | 17 | 4 | +13 |
| S35 λ1/seuil 0,05 — meilleur global | **83,0414 %** | **38,1043 %** | 382 | 309 | +73 |
| S35 λ2/seuil 0,02 | 82,9739 % | 39,7156 % | 111 | 78 | +33 |
| S35 λ4/seuil 0 | 82,9233 % | 40,6229 % | 8 | 5 | +3 |

**Aucune des neuf politiques S35 ne conserve toutes les bonnes réponses S18.** La meilleure globale gagne +73 grâce aux K0/K1, mais perd environ 2,5 points de pourcentage d'Exact-K poly ; ne pas promouvoir. S18 inchangé. Cohorte de développement déjà largement explorée ; pas de validation indépendante sur compositions inédites.

## Audit usage individuel (politique λ1/seuil0,05)

Une tête **présente dans le chemin** d'un événement corrigé n'est pas nécessairement celle qui l'a corrigé. Les états intermédiaires de chaque K n'ont pas été archivés, donc cet audit est **descriptif**, pas une ablation causale.

| Tête | Nombre de sélections | Dernière sélection d'un chemin corrigé | Dernière sélection d'un chemin régressé |
|---|---:|---:|---:|
| E12 deuxième note | 1 351 | 99 | 64 |
| F_keep2 | 959 | 0 | 0 |
| F_keep_any | 554 | 0 | 0 |
| H6 flux énergie | 408 | 31 | 30 |
| H7 morphologie attaque | 346 | 86 | 32 |
| H2 cycle de vie | 322 | 33 | 38 |
| H3 harmoniques | 296 | 22 | 30 |
| H5 sources | 241 | 16 | 19 |
| H10 couplage | 229 | 34 | 34 |
| C43 | 192 | 15 | 10 |
| H1 spectral | 176 | 12 | 12 |
| C32 | 171 | 13 | 26 |
| AB état précédent | 117 | 9 | 2 |
| H4 fondamentale | 98 | 7 | 9 |
| F_keep3 | 84 | 0 | 0 |
| F_keep4 | 45 | 0 | 0 |
| C34 | 39 | 1 | 0 |
| C23 | 27 | 4 | 3 |
| H8 pitch shift | **0** | 0 | 0 |

Deuxième/ troisième sélection réalisée sur 2 196/1 092 événements. Aucun gain de coût en déduire : les prédictions des experts sont calculées en amont pour tous les événements possibles.

### Structure exacte des pertes du meilleur score global

| Vrai K | Corrigés | Régressés |
|---|---:|---:|
| K0 | 98 | 0 |
| K1 | 159 | 0 |
| K2 | 66 | **156** |
| K3 | 55 | **87** |
| K4 | 4 | **59** |
| K5 | 0 | 6 |
| K6 | 0 | 1 |

Le **défaut majeur** n'est donc pas une absence de gains : le routeur change trop de décisions initialement justes pour **K2/K3/K4**. Toute modification doit passer par l'audit individuel et la validation OOF sur nouvelles compositions ; l'ordre variable est prouvé opérationnel, l'amélioration polyphonique n'est pas démontrée.

## Actions de conception conservées

1. Exclusion H9 inconditionnelle.
2. H8 à sortie de vrai pitch-shift WAV et attestations OOF requise avant activation. **Ne pas inventer de scores.**
3. Maintenir le registre 18 modules et leurs masques K ; ne pas supprimer les têtes malgré leur moindre score initial.
4. Pour préserver le polyphonique, entraîner un ordonnanceur capable d'exiger une corroboration acoustique *indépendante* avant de remplacer une bonne décision K2/K3/K4, plutôt que d'abaisser les seuils pour un gain global K0/K1.
5. Avant tout gain déclaré, stocker les distributions et propositions *après chaque tête* pour auditer causalement leurs effets par K. Aucun ajustement aux vrais labels des quatre cas S29 ou aux IDs des régressions S35.
