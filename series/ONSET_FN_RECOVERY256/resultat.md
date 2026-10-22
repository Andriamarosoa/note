# NOTE — série FNRecovery tête 4 native256

## Référence scientifique signée

- S613 natif256 GuitarSet FIT80 : TP8099, FP4600, FN3168, F1 **67,5874155 %**.
- MLP69 troisième tête Onset ErrorPrediction sans feedback (run #39749340403) : TP8054, FP4411, FN3213, F1 **67,8745997 %**.
- Audit FN S613 sur 80 WAV, 306536 PCM256 (run #39754613388) : **2287/3168 (72,19 %) avec P(Onset) insuffisant**, 269 défaut simultané Onset+Sustain, 175 Sustain trop faible, 130 non-armé, 307 budget/timing/appariement.
- Couvertures d'ORACLE, PAS de scores de reconnaissance : sensitive 1923 FN, moderate 1668 FN, cautious 1367 FN parmi les 3168.

## Hypothèse préenregistrée H_FN_RESCO_01

Former un véritable quatrième modèle neuronal indépendant `FNRecoveryHead` qui prédit les nouvelles attaques manquées par le main au PCM256 actuel à partir de 69D causaux. L'expérience 3 TRAIN + 1 CAL + 1 OUTER par composition doit produire 15 MLP véritables ; CAL sélectionne la politique et le seuil sans ouvrir OUTER. Les 5 checkpoints retenus sont SHA256 figés.

**Prédiction falsifiable :** le VRAI flux causal combinant les 4 têtes sur les 80 WAV doit dépasser le F1 du meilleur triple Onset **67,8746 %**, tout en reproduisant à l'identique les trois anciennes sorties. Si le F1 ne s'améliore pas, notre hypothèse d'une récupération utile à partir des propositions disponibles est non démontrée / insuffisante, ne pas promouvoir, rouvrir l'audit suivant GitHub issue #140.

## Statut

Code expérimental créé. Nouveau score **non encore mesuré**. Les scores event-proxy ne peuvent pas être présentés comme performances Live. Les autres cinq caractéristiques et le MIDI officiel demeurent inchangés. Pas de poids S708 entraîné1024, pas de lookahead, pas d'attente d'une trame future. Les poids S613/S701 ont un overlap d'entraînement historique avec FIT80, aucun test externe complet.
