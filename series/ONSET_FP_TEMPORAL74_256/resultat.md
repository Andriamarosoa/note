# Résultat — H_FP_PARENT_TIMELINE_01 (69D vs 70D vs 74D) PCM256

**Statut : RUN INITIAL À EXÉCUTER, preuve nouvelle non disponible avant CI. Ne pas promouvoir.**

## Référence vérifiée
- S613: TP8099 FP4600 FN3168 F1 67.5874155%.
- 3 têtes: TP8054 FP4411 FN3213 F1 67.8745997%.
- Meilleur 4 têtes: TP8160 FP4605 FN3107 F1 **67.9094541%**, avec 106 TP nets et 194 FP nets ajoutés par FNRecovery.
- Sous-échantillonnage négatif FNRecovery insuffisant [issue #141](https://github.com/Andriamarosoa/note/issues/141): ablation all-negatives donne TP8100 FP4500 FN3167 F1 67.8761%, donc rejeté.
- Audit FP [#39772165153](https://github.com/Andriamarosoa/note/actions/runs/39772165153) : 4411 FP MLP69 ; 782 FP et 678 TP avec parent natif récent à intervalle harmonique dans 80ms strictement passés; **3629 FP sans** ce motif; veto harmonique brutal contreproductif.

## Hypothèse H_FP_PARENT_TIMELINE_01 [issue #142](https://github.com/Andriamarosoa/note/issues/142)

Three independently trained true PCM256 neural networks on EXACT same TRAIN samples/seeds:
- 69D control: original main acoustic + context.
- 70D control: 69D + old MLP69 P(correct).
- 74D temporal: 70D + age of strictly previous Main attacks at four harmonic parent intervals.
15 MLP, 5 composition TRAIN/CAL/OUTER folds, weights+CAL thresholds frozen before OUTER, final TRUE 80 original WAV replay required. No labels/future entered any Live model.

**Preregistered acceptance criterion:** 74D real FP strictly below BOTH controls, real F1 at least **67.9094541%**, real F1 >= 70D; old four-head outputs unchanged (TP8160 FP4605 FN3107), all other axes/MIDI untouched. If FALSE, mark H_FP_PARENT_TIMELINE_01 REFUTED_OR_INCOMPLETE and reopen audit, NEVER substitute event-only proxy F1.

**New result:** NOT YET VERIFIED at the time of experiment commit. Real 256 audio decoding and latency validation not yet reported.
