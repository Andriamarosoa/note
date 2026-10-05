# Exact-K : audit de stabilité temporelle

Audit du 6 octobre 2026 sur `codex/v273-failure-clustering`, après `d9f75cb`.
**Les fréquences attendues persistent davantage que les composantes non appariées,
mais les deux façons testées d'exploiter ce signal dégradent Exact-K. Elles sont
rejetées ; la référence est conservée.**

Protocoles écrits avant les évaluations correspondantes :
[deux fenêtres](v273-temporal-stability-protocol.md) et
[feature directe de persistance](v273-temporal-persistence-feature-protocol.md).
Les [preuves](evidence/v273-temporal-stability/) permettent de rejouer les
extractions et les modèles sans audio ni réajustement.

## Ce qui est comparé

Même cohorte de **488 cas** (272 K3, 216 K2), mêmes **1 666 lignes uniques**,
122 enregistrements et **845 lignes VAL d'action**. Folds internes **0, 1, 2, 4**
uniquement : les fichiers audio et annotations du fold 3 ne sont pas chargés.
Base neuronale, B_low, population, saillance, pool-64, gabarit gaussien `1/h²`,
petite LR, seuil et règle de sélection restent figés. Aucun réseau n'est entraîné.

À partir du début de candidat `s`, les fenêtres de 2 048 échantillons sont :

| Fenêtre | Intervalle | Durée |
|---|---|---:|
| Avant | `[s-2048, s)` | 46,44 ms |
| Première | `[s, s+2048)` | 46,44 ms |
| Seconde | `[s+2048, s+4096)` | 46,44 ms |

Les deux spectres positifs soustraient **le même spectre avant**. Soustraire la
première fenêtre à la seconde aurait effacé le maintien d'une note. Chaque
spectre est normalisé séparément. Le pool est produit seulement par la première
fenêtre et reste commun. Le contexte futur total est **92,88 ms**, soit un coût
supplémentaire de **46,44 ms**.

La première fenêtre reproduit les 1 666 spectres archivés avec un écart maximal
de **0**. Les fréquences sélectionnées sont identiques ; l'écart de résidu lié
au calcul vectorisé est au plus `6,28e-16`. Les features de contrôle archivées
sont réutilisées bit pour bit après ce contrôle. Aucun résidu nul dans les deux
fenêtres et aucun padding postérieur ; six fenêtres pré utilisent le padding
historique par zéro, sans exclusion de ligne.

Ce travail concerne **l'audio normal**. Le projet utilise aussi le chemin
compressé : aucun gain ni cause dans ce chemin n'est démontré ici. Ces folds ont
déjà été inspectés et ne constituent pas un test final intact.

## Ce qui persiste réellement

Sur les 1 666 lignes, les trois fréquences des deux triplets indépendants sont
appariables à 55 cents dans **414 cas**. Il en reste deux dans 592 cas, une dans
437 et aucune dans 223. Il s'agit de fréquences choisies, pas d'une preuve que
leurs coefficients représentent trois sources physiques actives.

Sur les 272 K3 de la cohorte diagnostique :

| Composantes du premier triplet | Total | Retrouvées dans le second | Fraction |
|---|---:|---:|---:|
| Appariées à une note attendue | 405 | 313 | 77,28 % |
| Non appariées | 411 | 165 | 40,15 % |

La stabilité contient donc une association avec la présence d'une fondamentale
attendue. **Cela ne suffit pas à séparer les accords K2 des accords K3.** Avec
exactement le même triplet calculé pour tous les vrais K, le nombre moyen de
fréquences persistantes est **1,884 pour K2** et **1,757 pour K3**. L'AUC brute
« davantage de persistance => K3 » est **0,4618**, sur ces seuls cas déjà inspectés.
Aucun sens de score ou seuil n'est choisi à partir de ce diagnostic.

| Vrai K | 0 fréquence persistante | 1 | 2 | 3 |
|---|---:|---:|---:|---:|
| K2, 216 cas | 23 | 45 | 82 | 66 |
| K3, 272 cas | 30 | 81 | 86 | 75 |

Dans la seconde fenêtre, toutes les notes attendues des 488 cas gardent une
couverture Hann supérieure ou égale à 10 %. Des notes étrangères y sont actives
dans **126/272 K3** et **128/216 K2** ; une nouvelle note étrangère commence
dans cette fenêtre dans respectivement **15** et **16** cas. Ces événements
décrivent le contexte, sans prouver l'origine d'une composante choisie.

## Essai 1 : même couple/triplet sur les deux fenêtres

Pour chaque combinaison, minimiser la moyenne des deux résidus NNLS normalisés.
Les fréquences sont communes, les amplitudes libres dans chaque fenêtre. Ce
critère permet une variation d'amplitude ; il n'impose pas une présence active
de toutes les composantes dans les deux fenêtres.

Il améliore le nombre de notes attendues retrouvées dans 50 K3 et le dégrade dans
21. Les triplets complets passent seulement de **31 à 33/272**. Pour K2, les
couples complets passent de **47 à 51/216**. La seconde fenêtre seule donne
26 triplets et 50 couples complets ; elle n'est pas évaluée comme un bras choisi.

Malgré cette petite amélioration des fréquences choisies, le correcteur Exact-K
fondé sur les deux résidus conjoints régresse :

| Bras fixe | Fold 0 | Fold 1 | Fold 2 | Fold 4 | Corrections | Régressions | Autres K | Net VAL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Première fenêtre, contrôle | +4 | −2 | −9 | +9 | 137 | 135 | 192 | **+2** |
| Critère conjoint | −5 | −3 | −25 | +1 | 135 | 167 | 183 | **−32** |

Nets FIT conjoints : **−81, −74, −90, −95**. Rejet et abstention dans les quatre
rotations. Le +2 du contrôle reste descriptif et rejeté sur FIT.

## Essai 2 : ajouter directement la persistance

Après ce premier rejet, un protocole additionnel fixe une seule feature : le
nombre de fréquences appariées entre les deux **triplets**, divisé par trois.
La feature n'utilise ni K réel ni fréquence annotée ; elle est ajoutée aux deux
résidus de la première fenêtre. Aucun seuil, durée ou poids n'est recherché.

| Bras fixe | Fold 0 | Fold 1 | Fold 2 | Fold 4 | Corrections | Régressions | Autres K | Net VAL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Première fenêtre, contrôle | +4 | −2 | −9 | +9 | 137 | 135 | 192 | **+2** |
| Avec feature de persistance | −4 | −7 | −15 | +5 | 124 | 145 | 191 | **−21** |

Nets FIT avec persistance : **−70, −62, −78, −78**. Rejet et abstention dans
les quatre rotations. Les actions « autres K » sont conservées dans le bilan :
elles restent incorrectes avant et après la transformation 3→2, donc leur
contribution au net Exact-K est nulle.

## Vérification et décision

- Les cinq nouveaux tests vérifient le pré commun, le contrôle historique,
  le solveur, des mélanges synthétiques à amplitudes variables, un transitoire,
  la séparation des folds et le rejeu de la LR à trois features.
- Le rejeu intégral recalcule les deux fenêtres archivées pour les 1 666 lignes,
  les fréquences et les 488 traces diagnostiques. Il n'utilise aucun audio.
- Les **16 modèles finaux et 48 modèles internes** sont rejoués sans réajustement :
  erreur de probabilité **0**, mêmes actions, mêmes choix et mêmes abstentions.
- Le contrôle reproduit les probabilités antérieures bit pour bit.
- Le précédent run [`37371872093`](https://github.com/Andriamarosoa/note/actions/runs/37371872093)
  est signalé `failure`, avec job `111970757347` annulé avant toute étape,
  aucun log disponible et aucun artefact. Sa cause n'est pas établie ; il ne
  documente pas un échec d'une assertion scientifique. Les vérifications locales
  et le nouveau contrôle CI sont distingués.

**Décision : conserver la base et rejeter les deux variantes temporelles.**
Ni la reconstruction sur deux fenêtres ni la feature de persistance ne fournissent
une correction sélectionnable. La meilleure couverture fréquentielle ne se
transfère pas automatiquement en meilleur Exact-K.

Avant un autre bras, la question à auditer est ce qui distingue une nouvelle
attaque d'une composante déjà présente : mesurer le support des fondamentales
dans les puissances pré, post-1 et post-2, puis après la soustraction. Réutiliser
les diagnostics pré/post déjà faits et ne compléter que les mesures manquantes
de la seconde fenêtre. Les annotations resteront diagnostiques. Aucun nouveau
réseau complet ni balayage de seuils n'est justifié par ce résultat.

## Reproduction

Extraire et vérifier les archives `v273-reconstruction-criterion`,
`v273-residual-acoustics` et `v273-temporal-stability`, puis :

```bash
PYTHONPATH=.:src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B \
  scripts/replay_v273_temporal_stability.py \
  --reference /chemin/reconstruction/hann \
  --temporal /chemin/temporal --diagnosis /chemin/diagnosis \
  --persistence /chemin/persistence --cases /chemin/acoustic/cases.jsonl \
  --output /chemin/replay.json
```

Environnement figé : Python 3.12.14, NumPy 1.26.4, SciPy 1.17.1,
scikit-learn 1.4.2. Les fichiers de sortie existants ne sont pas écrasés par
les scripts d'extraction, de diagnostic ou d'évaluation.
