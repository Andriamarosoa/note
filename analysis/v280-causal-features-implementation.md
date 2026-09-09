# V28.0-A — Front-end CQT causal et tests anti-fuite

Date : **2026-09-09**

Branche : **`v280-harmonic-pitch-count-research`**

Référence préservée : **V27.3, 42,6019 % exact-K polyphonique**
Statut : **front-end implémenté et vérifié ; aucun modèle entraîné**

## Résultat de cette étape

La première étape du plan [V28-CHEC](./v280-polyphonic-count-model-research.md)
est implémentée : une représentation acoustique alignée sur les hauteurs, avec
endpoints explicites et sans centrage implicite.

Le composant `src/causal_note/v280_causal_cqt.py` fournit :

- une grille constant-Q de 3 bins par demi-ton ;
- 238 bins internes, de E2 jusqu'aux harmoniques utiles sous Nyquist ;
- des FFT multi-résolution groupées par octave et alignées à droite ;
- des caches float16 par piste, versionnés et liés à l'empreinte de leur config ;
- des crops fixes `24 × 238 × 3` par cluster ;
- les canaux log-magnitude, hausse face au pré-cluster et flux positif ;
- les 14 décalages fractionnaires utilisés par le shift-and-aggregate : sept
  sous-harmoniques et sept harmoniques, sans dupliquer le centre résiduel.

La fenêtre la plus longue mesure **27 528 échantillons = 624,22 ms**. Une trame
finissant à l'échantillon `e` lit uniquement l'intervalle demi-ouvert
`[e - longueur, e)`. La décision de cluster reste limitée à
`cluster_start + 1 764` échantillons ; aucun futur supplémentaire n'est ajouté.

## Vérifications

Commande ciblée :

```bash
PYTHONPATH=src:. python -m unittest -v test.test_v280_causal_cqt
```

Résultat : **13/13 tests V28 passés**.

Les tests couvrent notamment :

- localisation d'un sinus de 440 Hz sur la case MIDI attendue ;
- parité PCM16 / float normalisé ;
- invariance exacte des trames passées quand tout le futur est remplacé ;
- égalité exacte entre calcul piste entière et calcul par préfixes/chunks ;
- invariance du crop de cluster aux échantillons postérieurs à la décision ;
- padding gauche au début d'une piste ;
- sens, interpolation et masque de validité des décalages harmoniques ;
- round-trip du cache float16, refus d'écrasement et rejet d'une configuration
  différente.

Régression complète :

```text
Ran 316 tests in 0.854s
OK (skipped=13)
```

Les 13 tests ignorés dépendent principalement de TensorFlow, absent de
l'environnement courant. Les 316 tests exécutés passent.

## Benchmark CPU borné

Commande :

```bash
PYTHONPATH=src:. python scripts/benchmark_v280_causal_features.py \
  --seconds 3 --repeats 2
```

Le signal est entièrement synthétique et déterministe. Aucun jeu de données,
fold, label, checkpoint ou entraînement n'est ouvert.

| Mesure | Valeur |
|---|---:|
| Audio | 3,000 s |
| Temps des deux répétitions | 0,3007 s ; 0,2259 s |
| Médiane | 0,2633 s |
| Débit médian | **11,39× temps réel** |
| Cache | 517 × 238 float16 |
| Taille du cache 3 s | 0,235 Mio |
| Projection indicative pour 30 s | environ 2,35 Mio par piste |
| Crop de cluster | 24 × 238 × 3 float32 |
| Empreinte de configuration | `5fae41ea91e90b5cb2a7611edb41c6dba633eaf8853341a481ea3a2c9ba1e805` |

Cette mesure valide seulement la faisabilité du front-end sur le CPU courant.
Elle ne remplace pas le benchmark Mac M4 avec lecture WAV, écriture de caches et
charge système réelle.

## Garde-fous conservés

- V27.3 n'est ni modifiée ni remplacée.
- Aucun accès à la validation historique ou à Locked12.
- Aucun checkpoint Basic Pitch/Harmonica contaminé par GuitarSet.
- Aucun téléchargement GAPS/GOAT/SynthTab.
- Aucun seuil choisi et aucun résultat outer consulté.
- Aucun entraînement lourd lancé.

## Limite actuelle et prochaine étape

Ce checkpoint ne prouve encore **aucun gain exact-K** : il prouve que le nouveau
signal est causal, déterministe, cacheable et assez rapide pour justifier le
smoke suivant.

La prochaine étape est V28.0-B : construire le petit tronc harmonique avec
sorties corde×case, six preuves de corde, Poisson-binomiale et résidu catégoriel,
puis vérifier la forme du graphe, les gradients finis et l'overfit d'un mini-lot
strictement interne. Le teacher GAPS reste une étape séparée nécessitant une
autorisation explicite avant téléchargement.
