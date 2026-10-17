# S653 — Learned Tone NoteOn rescue: real GuitarSet results

## Verdict

**Évaluation GitHub SUCCESS :** https://github.com/Andriamarosoa/note/actions/runs/39007333374

Le Tone ajouté comme *tête de nouvelles attaques* au-dessus du
décodeur NOTE fournit des corrections mesurables, mais **ne bat PAS
S641** et n'approche pas 90%. Aucune promotion.

Fichiers :
- `scripts/s653_learn_tone_noteon_repair.py`
- `test/test_s653_learn_tone_noteon_repair.py`
- `S653_TONE_CAUSAL_NOTEON_REPAIR_GUITARSET/report.json`

## Protocole physique et séparation

- 36 GuitarSet TRAIN, neuf compositions; 12 GuitarSet DEV déjà
  analysées (benchmark exploratoire), aucun TEST utilisé.
- Original 44,1 kHz, hop causal 256 ; deux copies
  `scipy.sosfilt` 1 ordre passe-bas 2800/1100 Hz ;
  aucun pitch shift, pas de retard ajouté artificiellement.
- Les 3 vues passent dans les têtes **S614 gelées**.
  Tous les sélecteurs conservent **K du signal original**.
- Une nouvelle hauteur est candidate uniquement si son onset
  de Tone dépasse le seuil enregistré et si son onset natif
  n'était pas déjà confirmé.
- Un `CausalSelection` NOTE sans contrôle propose les candidats ;
  chaque événement émis sur TRAIN est étiqueté
  correction inédite / régression par comparaison aux annotations
  et aux vrais NoteOn S614.
- 5-fold GroupKFold par composition; 4 têtes séparées
  logistic C0.25, logistic C1, HGB, control K-only ;
  famille et seuil choisis **uniquement sur TRAIN**.
- Toutes les sorties sont évaluées en NoteOn réel ±80ms,
  sans oracle de timestamps au moment de la décision, sans
  réécrire le LIVE et sans antidater les attaques.

## TRAIN 5-fold OOF

**551 propositions d'onset Tone**, dont **71 vraies
corrections** nouvelles (sur 36 fichiers).

Utilité pré-déclarée `TP - 0.5 FP` :
- Logistic C.25 : 28 TP /37 FP, seuil **0.80**, utilité **9.5**
- Logistic C1 : 33 TP /50 FP, seuil 0.75, utilité 8
- HGB : 14 TP /11 FP, seuil 0.75, utilité 8.5
- K-only : 0 TP /0 FP, abstention totale, utilité 0

**Candidat retenu avant DEV : Logistic C0.25 @0.80**.

## GuitarSet DEV réel

| Politique | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|
| S614 original | 998 | 585 | 582 | **63.10465%** |
| Tone union sans sélection | 1033 | 711 | 547 | 62.15403% |
| Tone 2800 seule | 1024 | 648 | 556 | 62.97663% |
| Tone 1100 seule | 1025 | 696 | 555 | 62.10239% |
| **Logistic C.25 choisie TRAIN** | **1007** | **601** | **573** | **63.17440%** |
| Logistic C1 | 1011 | 607 | 569 | 63.22702% |
| Arbre HGB | 1005 | 593 | 575 | 63.24733% |
| Contrôle K-only | 998 | 585 | 582 | 63.10465% |
| S641 expérimental (mesure précédente) | 1009 | 594 | 571 | **63.39931%** |

La politique gagnante TRAIN apporte **+9 TP/+16 FP** à S614,
F1 +0.06976 point, mais reste −0.2249 point vs S641.

La variante HGB ne serait *pas* une promotion honnête :
elle donne davantage de F1 sur DEV, mais n'était pas celle
désignée sur TRAIN.

### Répartition réelle des corrections

Comparaison du routeur Logistic C.25 choisi sur TRAIN vs S614 :
- `Funk3-112-C#_comp` : **+4 TP/+1 FP** ;
- `Funk3-98-A_comp` : +2 TP/+5 FP ;
- `BN3-119-G_comp` : +1 TP/+0 FP ;
- `Rock1-130-A_solo` : +1 TP/+0 FP ;
- `BN3-154-E_comp` : **+0 TP/+5 FP**.
Aucune référence ne doit être changée sur des corrections
concentrées dans quelques morceaux.

E2: 0 TP avec toutes les propositions et variantes S653.
E3: S614 44 TP, Tone union 46, routeur 45.
K3+ : S614 550 TP, routeur 555.

### Limites structurelles importantes

La proposition Tone non gardée livre **35 TP de plus que S614**
mais +126 FP. Même un routeur sans erreurs sur ces seuls
candidats ne suffit pas à rapprocher NOTE des 90% de F1 :
il faut des propositions nouvelles issues d'une **meilleure
reconstruction H0/H1**, de l'identité des fondamentales et des
attaques, pas seulement trois timbres du même signal.

Cela constitue une **limite pratique de cette famille restreinte
de générateurs Tone**, et non une borne supérieure universelle
des algorithmes de transcription.

## Décision

**Aucune promotion ;** S614 officiel gelé, S641 meilleur
expérimental validé selon la sélection TRAIN. Les têtes Tone peuvent
alimenter une future sélection experte, mais ne doivent pas
automatiquement créer de NoteOn, ni masquer les vrais K3+.


## Quantitative 90% feasibility audit for S653's restricted candidate pool

The union created only **35 additional matched NoteOn annotations**
over S614 (1033 vs 998 TP). Suppose a hypothetical perfect policy
kept all original S614 predictions and retained **all 35 new true
Tone candidates** but **none of the 126 extra false events**.
Then the best restricted S653-style *add-only* result would be:
- TP 1033, FP 585, FN 547;
- **F1 = 64.6029%**, still far from 90%.

Even if this restricted 1033-TP pool could somehow also eliminate
**all false positives** without dropping any true note, its
F1 would be only **79.07%**.

**This is NOT a mathematical ceiling for NOTE or all possible
causal multi-view systems**: changing candidate generation,
decoder state, pitch identity, or offset/onset hypotheses could
recover events not in S653's fixed union. It is a concrete
diagnostic showing that **training a better gate on the exact
S653 candidate pool cannot by itself achieve 90%**.
Future loops must address missed fundamentals and attacks.
