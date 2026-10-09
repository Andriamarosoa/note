# V27.3 — Série 27 : les repassages A/B doivent-ils être complets ?

Protocole fixé AVANT l'exécution. La question est : lorsqu'une tête A a reçu le diagnostic de B, faut-il *recalculer intégralement toutes les têtes à chaque passage*, ou réutiliser certains résultats et recalculer uniquement lorsque l'état de la sélection change ?

**Source :** les 19 checkpoints d'origine S20 (A-first) sur 59 309 événements / 19 morceaux / folds 0,1,2,4, et la référence S18 49 178 corrects global, 2 998 poly (2 934 corrections face à freeze). Les 335 entrées acoustiques, les poids, les scalers et la décision initiale sont figés. Aucune étiquette n'intervient dans les décisions, arrêts ou messages. Tous les seuils ci-dessous sont déclarés *avant* le premier run.

## Référence nécessaire
- `full_4` : 4 évaluations successives de A ; B évaluée après A1/A2/A3 ; le quatrième B est inutile puisqu'il ne sera pas réutilisé. Vérifier sur **chaque morceau** que la distribution finale et tous les passages A concordent avec les distributions S20 archivées à 5×10^-5 près.
- `full_2` et `full_3` : arrêter inconditionnellement après 2 ou 3 A, au lieu de relancer les têtes.
- Comparer **trois choses distinctes** : nombre d'appels *effectifs* A et B par événement, classes prédites, et distributions de probabilités finales. Les coûts sont des évaluations échantillon×tête, pas un temps GPU estimé.

## Variantes de repassage partiel (ordre A-first constant)
1. `B_initial_only` : A fait 4 passes, B uniquement après A1 ; réutiliser le message précédent.
2. `B_if_K_changed` : B après A1, puis seulement si la classe argmax de A diffère depuis la précédente passe de A (sinon réutiliser l'ancien message).
3. `B_if_delta_002`, `B_if_delta_005`, `B_if_delta_010` : B après A1, puis seulement si la différence moyenne absolue de la distribution courante avec la précédente dépasse .02, .05, .10.
4. `msg_only_parent_candidate` : B est réévaluée à chaque étape mais n'envoie à A que deux classes : la classe S18 et la classe actuellement proposée par A ; messages des autres classes à zéro. Ce n'est **pas** une réduction de coût de calcul de B, mais une ablation de largeur de message.
5. `msg_only_top2`, `msg_only_top3` : B est réévaluée à chaque étape, mais son message est masqué aux 2 ou 3 classes ayant la plus forte probabilité *dans la sortie courante de A* (aucun oracle).
6. `early_stable_002`, `early_stable_005`, `early_stable_010` : à partir de A2, arrêter toute réévaluation d'A/B **pour cet événement seulement** si la classe argmax est inchangée et si la différence moyenne absolue p(t)-p(t−1) est inférieure à .02/.05/.10.

Tous les mécanismes doivent préserver un état séparé par événement : un message B non réévalué est **réutilisé**, pas effacé ; un A arrêté conserve sa distribution. Le calcul B est fait uniquement sur les exemples nécessitant sa mise à jour, et A sur les exemples encore actifs. Aucun faux gain de coûts dû à un masque appliqué *après* exécution.

## Décodages à évaluer
Pour chaque politique de la liste, fixer quatre décodages : `raw` argmax K0–K6, et `margin0.25`, `margin0.40`, `margin0.60` qui préservent S18 si le gain de P(proposé) par rapport à P(S18) est insuffisant. C'est une **grille figée**, sans vérité à l'inférence.

## Vérifications et critères
- Tests unitaires : la référence `full_4` reproduit les probabilités S20 originales ; `B_initial_only` utilise moins de B ; `early_stable` saute vraiment des évaluations d'A/B ; `msg_top2` ne contient que 2 composantes non nulles au plus.
- Audit contre S18 et freeze pour chacun des K0–K6, global/poly, fold, événements modifiés, nouveaux corrects / bonnes prédictions détruites / changements neutres. Évaluer distance de probabilité à `full_4`, coût A/B normalisé et qualité Exact-K.
- Une variante peut être « équivalente en qualité » mais **non identique** : ne pas l'appeler équivalente sans vérifier égalité événement par événement.
- Ne jamais promouvoir une variante destructrice des anciennes corrections. Données connues, seuils exploratoires, pas de preuve de généralisation à nouvelle musique. Archiver toutes les sorties, même négatives.
