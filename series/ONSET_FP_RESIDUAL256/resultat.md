# Résultats — audit FP MLP69 résiduel à PCM256

**Statut au dépôt :** source et tests publiés, audit scientifique indépendant lancé via `onset-fp-residual-audit256.yml` ; ne pas annoncer un nouvel effet avant que le job et l'artefact soient vérifiés.

### Résultats signés d'origine

- S613 : **TP8099 FP4600 FN3168 F1 67,5874 %**.
- 3 têtes MLP69 corrigé : **TP8054 FP4411 FN3213 F1 67,8746 %**.
- 4 têtes FNRecovery première boucle : **TP8160 FP4605 FN3107 F1 67,9095 %**, 300 NoteOn nouveaux, 106TP nets et 194FP nets.
- 4 têtes entraînées sur tous les négatifs TRAIN : **TP8100 FP4500 FN3167 F1 67,8761 %**, **échec au critère scientifique préenregistré** [#141](https://github.com/Andriamarosoa/note/issues/141), aucune promotion.

### Observations exploratoires résiduelles à confirmer par CI

- Solo : 863 FP, accompagnement : 3548 FP ; basse E2–B2 624 FP/1226 propositions ; médium C3–B3 2145 FP/5706 ; aigu C4–E6 1642 FP/5533.
- Les 69D et le MLP de risque laissent beaucoup de FN et FP déjà présents dans les *Main* ; il faut une amélioration de discernement acoustique, pas une baisse uniforme des seuils.
- Proximité d'une ancienne attaque native d'une note harmonique inférieure en moins de 80ms strictement passés : environ 782 FP **mais aussi 678 TP**. Un veto global sur ce motif serait contreproductif et ferait chuter la F1.
- Aucun de ces faits n'est une preuve qu'une variante 74D apprenante aura un bon score sur nouveaux WAV.

**Règle [#140](https://github.com/Andriamarosoa/note/issues/140) :** si l'intervention en question ne valide pas ses effets annoncés face à un modèle contrôle sur replay causal 256, considérer l'hypothèse incomplète ou réfutée, rouvrir l'audit et ne rien promouvoir.

### Artefacts attendus

`report.json` : comptage exact du MLP69 reconstruit avec cinq checkpoints signés SHA256, catégories TP/FP non exclusives, veto naïf en mode contre-factuel **événementiel seulement**. `fp_signals74.jsonl` : propositions natives conservées et **74 entrées causales**, étiquette supervisée après coup, partition composition. Ni ces fichiers ni leur F1 contre-factuel ne constituent une nouvelle F1 du décodeur Live.

**Production MIDI inchangée, autres cinq architectures inchangées, aucun poids entraîné1024.**
