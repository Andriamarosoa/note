# Onset — H_FN_NEGCOVERAGE_01 : ablation de couverture des négatifs FNRecovery PCM256

## Justification par audit réel, non interprétation au doigt mouillé

Référence **4 têtes indépendant**, 80 WAV/306536 décisions PCM256, run [#39759373185](https://github.com/Andriamarosoa/note/actions/runs/39759373185) : S613 d'origine TP8099/FP4600/FN3168/F1 67.5874155 %, triple-tête frozen MLP69 TP8054/FP4411/FN3213/F1 67.8745997 %, 4 têtes + FNRecovery TP8160/FP4605/FN3107/F1 **67.9094541 %**. FNRecovery a produit **300** NoteOn supplémentaires, soit net **106 TP et 194 FP** ; ses 124 attaques appariées comprennent 18 déplacements d'attribution par rapport aux NoteOn du pipeline original. **32 pistes régressent**, 26 gagnent, 22 inchangées. En solo 17TP/55FP (72 propositions), en accompagnement 89TP/139FP (228 propositions). Funk2 et Rock3 subissent des régressions F1.

Le mécanisme de l'apprentissage précédent est objectivement visible : `train_one` sélectionne les négatifs au maximum à hauteur de `3×N_positifs`, rejetant jusqu'à plus de **80 % des exemples négatifs TRAIN**. Exemple réel d'un des plis : `1135 pos / 19547 neg` dont seulement 3405 neg échantillonnés. **Cette observation ne prouve PAS que la faible diversité entraîne les FP : c'est l'hypothèse soumise à intervention.**

## Hypothèse et prédiction écrites AVANT le test — [GitHub issue #141](https://github.com/Andriamarosoa/note/issues/141)

**H_FN_NEGCOVERAGE_01 :** les FNRecovery FP proviennent au moins partiellement de la faible couverture des *variétés* de négatifs. **Intervention isolée :** pendant l'entraînement des modèles FNRecovery MLP69, prendre **TOUS** les négatifs TRAIN, répéter de façon déterministe les positifs TRAIN `ceil(N_neg/(3N_pos))` pour garder une proportion proche de l'ancienne 1:3. Cela teste la **diversité des erreurs négatives** sans changer l'architecture 69D, les 3 sources de candidats, les poids S613/S701/MLP69, la façon d'émettre NoteOn, la composition des groupes ni la procédure de calibration CAL. L'équilibre de classes est approximativement préservé ; nombre de lignes et diversités effectivement changent, donc la cause reste à attribuer par les scores et erreurs réelles.

15 nouveaux MLP(64,32) doivent être effectivement entraînés, en 5 folds (3 compositions TRAIN, une CAL, une OUTER), 3 spécialistes possibles (sensitive/moderate/cautious). Seul CAL choisit `cut` et `policy`, jamais OUTER. Sauvegarder 5 checkpoints joblib et métadonnées avec SHA256. **La comparaison scientifique est le vrai deuxième replay causal sur les 80 WAV**, 256 nouveaux échantillons à 44.1kHz chaque décision, toutes les 4 têtes interrogées AVANT sélection ; les autres 5 caractéristiques restent gelées et le MIDI officiel n'est pas changé.

### Prévision réfutable et non-régression exigée

- Le vrai F1 de la version quatre têtes **doit dépasser 67.9094541 %**, le meilleur quatre têtes historique, **ET**
- les **FP totaux doivent être strictement inférieurs à 4605** sur le même ensemble,
- avec plus de **8054 TP** (référence des trois têtes). Le script scientifique échoue sinon ; ni ajustement des seuils ni changement de test après coup.
- Toutes les régressions par piste et par composition sont archivées pour audit ; si le score global augmente mais des compositions perdent beaucoup, pas de promotion sans validation supplémentaire.
- Si les FP ne baissent pas et le F1 ne monte pas, la faible couverture négative n'explique pas suffisamment les erreurs : **H_FN_NEGCOVERAGE_01 REFUTED_OR_INCOMPLETE**, rouvrir le diagnostic. Si la précision augmente mais le rappel s'effondre, diagnostic **partiel**, pas de promotion. Règle centrale [issue #140](https://github.com/Andriamarosoa/note/issues/140).

### Limites

- Dataset FIT80 déjà partiellement vu par les enseignants S613 et S701 (20/80, 48/80) : même une amélioration des nouveaux folds ne prouve pas la généralisation sur une population externe.
- La cadence de **5,805 ms** est le 256 hop ; le coût CPU end-to-end p95/p99 n'est pas mesuré. Aucune trame future, hold, latence d'attente ou poids S708 entraîné1024.
- Les trois têtes précédentes doivent reproduire **exactement** TP8054 FP4411 FN3213. Les autres cinq axes et la sortie MIDI publique ne sont pas modifiés.
- Un score « proxy candidats » n'a aucune valeur de promotion sans replay.

## Fichiers expérimentaux

- `scripts/train_onset_fn_recovery256.py` : uniquement les négatifs TRAIN passent de 3×positifs maximum à tous négatifs, positifs répétés (équilibre presque identique), 15 réseaux MLP.
- `scripts/replay_onset_fn_recovery256.py` : critères de réfutation préenregistrés contre **l'ancien meilleur quatre têtes**, pas seulement contre le triple.
- `tests/test_fn_recovery_all_negatives256.py` : vérifie couverture 100%, répétition déterministe, absence de données CAL/OUTER, références inchangées.
- `.github/workflows/onset-fn-recovery-all-negatives256.yml` : provenance signée, entraînement, replay, artifact scientifique.
- `series/ONSET_FN_NEGCOVERAGE256/{resume,resultat}.md` : audit et F1 réel une fois les résultats disponibles.

**Statut au moment du commit : entraînement et replay à exécuter. Aucun nouveau score revendiqué.**
