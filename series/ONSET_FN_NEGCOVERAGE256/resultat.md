# Résultat préenregistré — H_FN_NEGCOVERAGE_01

Hypothèse [issue #141](https://github.com/Andriamarosoa/note/issues/141), règle de réfutation [#140](https://github.com/Andriamarosoa/note/issues/140).

**PREUVES HISTORIQUES**
- Main S613 256 : TP8099 FP4600 FN3168 F1 67.5874155%.
- 3 têtes ErrorPrediction MLP69 no feedback : TP8054 FP4411 FN3213 F1 67.8745997%.
- 4 têtes FNRecovery baseline : TP8160 FP4605 FN3107 F1 **67.9094541%**, 300 candidats ajoutés dont net 106 TP/194 FP. Funk2/Rock3 régressent ; 32 pistes avec baisse F1. En solo 17TP/55FP, en accompagnement 89TP/139FP.
- Seulement ~17% des exemples négatifs d'un des TRAIN étaient effectivement utilisés (3405/19547) par le FNRecovery initial. Ceci est une observation et pas une démonstration de cause.

**INTERVENTION CONTRÔLÉE**
Tous les candidats négatifs TRAIN et répétition des candidats FN positifs TRAIN selon `ceil(N_neg/(3*N_pos))`. Exactement mêmes 69 caractéristiques causales, MLP(64,32), mêmes folds TRAIN/CAL/OUTER, poids trois têtes d'origine gelés, mêmes propositions PCM256, zéro lookahead/hold. 15 véritables modèles doivent être entraînés.

**CRITÈRE ÉCRIT AVANT LES RÉSULTATS** : F1 réel sur 80 WAV strictement > 67.9094541 %, FP réel strictement < 4605, TP réel > 8054, deux premières/ troisième têtes inchangées, zéro autres axes modifiés. Si non validé, pas de promotion et analyse de la réfutation. Aucun score proxy n'est recevable comme résultat.

**NOUVEAU RÉSULTAT** : non encore disponible au moment du commit.
