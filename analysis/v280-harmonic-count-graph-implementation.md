# V28.0-B — Graphe du compteur harmonique

Date : **2026-09-09**

Branche : **`v280-harmonic-pitch-count-research`**

Référence préservée : **V27.3, 42,6019 % exact-K polyphonique**
Statut : **graphe et mathématiques implémentés ; smoke TensorFlow à exécuter**

## Architecture figée pour le smoke

Le script `scripts/train_v280_harmonic_count.py` construit le premier student
`V28-CHEC-direct`. Il ne reçoit que la carte causale `24 × 238 × 3` produite par
[V28.0-A](./v280-causal-features-implementation.md).

Le graphe contient :

1. deux blocs ResNet causaux `5×5`, avec 8 canaux ;
2. une réduction fréquentielle `stride=3`, alignant la grille sur un bin par
   demi-ton ;
3. trois blocs de tronc à 32 canaux ;
4. dans chaque bloc, 14 copies harmoniques fractionnaires, une agrégation `1×1`
   et un ResNet causal `3×3` ;
5. une carte auxiliaire de 44 hauteurs MIDI ;
6. une carte auxiliaire `6 cordes × 20 positions`, qui conserve les unissons ;
7. six probabilités de naissance de corde ;
8. une distribution Poisson-binomiale exacte `P_PB(K=0..6)` ;
9. un résidu catégoriel global initialisé à zéro ;
10. la sortie principale
    `softmax(log(P_PB + 1e-7) + logits_résiduels)`.

La normalisation se fait sur la fréquence séparément à chaque instant. Elle ne
mélange donc pas les trames futures. Toutes les convolutions temporelles sont
précédées uniquement d'un padding gauche.

## Budget

Le nombre de paramètres est calculé indépendamment de TensorFlow et vérifié à
la construction du graphe :

| Bras | Paramètres entraînables |
|---|---:|
| V28-CHEC harmonique | **110 402** |
| Ablation sans agrégation harmonique | **67 298** |
| Porte maximale | 300 000 |

Le modèle harmonique utilise donc 36,8 % du budget fixé et reste légèrement
plus petit que Harmonica-small (137 k), avant toute augmentation de capacité.

## Objectifs figés pour le smoke

| Sortie | Forme | Loss | Poids |
|---|---:|---|---:|
| `cardinality` | 7 | CE exacte sur K | 1,00 |
| `string_birth` | 6 | BCE | 0,25 |
| `string_fret_onset` | 6×20 | BCE | 0,15 |
| `pitch_onset` | 44 | BCE | 0,10 |
| `poibin_cardinality` | 7 | CE exacte sur K | 0,10 |

La dernière loss est le garde-fou de cohérence : les six preuves de corde
doivent elles-mêmes expliquer K, même si le résidu global peut corriger leurs
dépendances.

## Tests exécutés ici

Les **8 tests indépendants de TensorFlow** passent. Ils vérifient :

- la grille physique des six cordes et la conservation des hauteurs unisson ;
- les cas déterministes `K=0..6` de la Poisson-binomiale ;
- l'égalité avec la loi binomiale lorsque les six probabilités valent 0,5 ;
- l'identité du résidu nul et sa capacité de correction ;
- la cohérence des cibles synthétiques ;
- les erreurs de forme, de domaine et le budget exact de paramètres.

Deux tests de graphe TensorFlow sont présents mais correctement ignorés sur la
machine de travail actuelle, où TensorFlow n'est pas installé. Ils devront
passer dans l'environnement Python 3.11 / TensorFlow 2.15.1 avant tout accès aux
données :

- formes exactes des cinq sorties, budget et présence/absence du bloc harmonique ;
- probabilités finies, somme à un, identité initiale `P=P_PB` et premier pas de
  gradient fini.

Régression actuelle, sans TensorFlow : **326 tests passés, 15 ignorés**.

## Smoke autorisé suivant

Le premier smoke reste totalement synthétique :

```bash
python -m unittest -v \
  test.test_v280_causal_cqt \
  test.test_v280_harmonic_count

python scripts/train_v280_harmonic_count.py smoke \
  --output-dir artifacts/v280-chec-synthetic-smoke \
  --steps 32 \
  --rows 14
```

Le script refuse un dossier de sortie existant, borne `steps<=200` et
`rows<=128`, enregistre les poids et un rapport JSON, et affirme explicitement :

- aucun dataset indexé ;
- aucun fold outer ouvert ;
- aucune validation historique ou Locked12 ;
- aucun checkpoint externe chargé.

## Ce qui n'est pas encore autorisé par ce checkpoint

- aucun score exact-K n'a encore été produit ;
- aucun cache GuitarSet n'a encore été miné ;
- aucun entraînement inner ou outer n'a commencé ;
- aucune fusion avec V27.3 ;
- aucun téléchargement du teacher GAPS.

Après réussite du smoke TensorFlow, la prochaine modification sera le mineur de
caches par piste et la dérivation des labels corde/case sur les partitions
internes uniquement. Le protocole outer restera fermé jusqu'à réussite d'un
mini-overfit interne et de l'ablation harmonique.
