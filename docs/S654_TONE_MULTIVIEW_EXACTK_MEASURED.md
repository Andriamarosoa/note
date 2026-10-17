# S654 — Vraie évaluation d'une tête de fusion Tone pour Exact-K

## Verdict validé par GitHub Actions

- CI réussite : https://github.com/Andriamarosoa/note/actions/runs/39007490136
- `scripts/s654_tone_exatk_head.py`
- `test/test_s654_tone_exactk_head.py`
- `S654_TONE_GUITARSET_K_MULTIVIEW_TRAINED_DIAGNOSTIC/report.json`

**Le meilleur modèle améliore le K diagnostique DEV de 42,39% à
44,35% (+31 décisions correctes sur 1 576), mais n'a PAS passé
le seuil d'approbation fixé sur TRAIN. Aucune promotion.**

## Contexte et méthode

Les trois vues sont la prise originale, un passe-bas causal 2800Hz
et un passe-bas causal 1100Hz, simulant grossièrement la variation
de timbre d'une guitare électrique, pas une vraie Stratocaster
calibrée. Ces vues passent dans l'architecture native S614.

Les labels Exact-K sont les hauteurs distinctes **encore actives 30ms
après les attaques annotées**. C'est le même protocole diagnostic
que S652 et il n'utilise pas l'annotation comme feature. Mais
l'instant choisi est fourni par la vérité terrain : **ce n'est
PAS un système autonome LIVE de détection Exact-K.**

S654 ne remplace pas le K original dans le système NOTE. Les
poids de S614, ses têtes de hauteur, sélections et moteur causal
LIVE sont inchangés.

Les quatre classifieurs de fusion expérimentés :
régression multinomiale (54D), régression sur les 21 probabilités
K des trois vues, petit HistGradientBoosting et Random Forest.
Tous sont choisis par 5-fold validation croisée **sur composition
entière** des 36 pistes GuitarSet TRAIN, jamais sur DEV.
12 pistes GuitarSet DEV observées dans l'historique fournissent
uniquement un benchmark exploratoire, pas de validation indépendante.

## Résultats TRAIN group OOF

- 5 284 observations K1-K6 sur 9 compositions ;
- tête K S614 originale **73,2210%** ;
- tête fusionnée `linear` **75,7381%**, **+2,5171 points**,
  progression sur 5 des 5 plis ;
- les autres têtes ont 75,4542% (linear_simple),
  73,8834% (trees), 75,2649% (forest) ;
- exigence de promotion préenregistrée : **+3 points ET 3/5 plis
  améliorés**. La condition du gain de +3 points échoue.
- **choix effectif déterminé sur TRAIN : S614_original_K**.

Le fait que les K TRAIN soient beaucoup mieux prédits que les K DEV
montre un **décalage de distribution important** : le défi est
la généralisation aux compositions/arrangements différents, pas
uniquement le seuil ou les harmoniques.

## Résultats K diagnostiques GuitarSet DEV (1 576 observations)

| Classe / modèle | Correct | Accuracy |
|---|---:|---:|
| K S614 natif | 668 | **42,3858 %** |
| Passe-bas léger | 674 | **42,7665 %** |
| Passe-bas fort | 671 | **42,5761 %** |
| `linear` multi-Tone (non promu) | **699** | **44,3528 %** |
| `linear_simple` | 684 | 43,4010 % |
| `trees` | 680 | 43,1472 % |
| `forest` | 672 | 42,6396 % |

### Par vrai K : tête `linear` contre original

| K | N | K original | K fusionné |
|---|---:|---:|---:|
| K1 | 441 | 70,29 % | **71,20 %** |
| K2 | 234 | 20,94 % | **26,92 %** |
| K3 | 458 | **39,96 %** | 35,81 % |
| K4 | 385 | 32,21 % | **40,26 %** |
| K5 | 58 | 3,45 % | **5,17 %** |
| K6 | 0 | — | — |

La tête gagne sur K1, K2, K4, K5, **mais regresse sur K3**.
La mesure générale est encore loin de 90%.

## Suites de recherche

- Cesser d'optimiser simplement les vues Tone :
  aucun filtre simple ne rend les partiels partagés uniques ;
- entraîner la compétition H0/H1 pour expliquer conjointement
  fondamentales et partiels (NNLS ou modèle de résidu
  régularisé), sans veto qui masque des vraies notes ;
- introduire des séries d'apprentissage diversifiées
  IDMT/GuitarSet, en maintenant un TEST vraiment inédit ;
- valider au niveau NoteOn/Exact-K complet et du débit CPU LIVE
  avant toute migration officielle.

**Cette campagne S654 a été réalisée et validée, aucun 90% ni
promotion revendiqué.**
