# V27.3 — arbitre neuronal conditionné par les transitions et les combinaisons

## Pourquoi cette expérience

La dernière version à 14 têtes apprenait le risque par K de destination, mais
son Exact-K poly était **34,2316 %** contre **34,2586 %** pour la référence.
Elle était quasiment équivalente au contrôle consistant à annuler les sorties
K0/K1 non soutenues. Les échecs restants étaient surtout K3↔K4.

L'audit causal des têtes a confirmé des effets contradictoires : supprimer C23
améliorait K2 mais détériorait K3 ; supprimer C34 protégeait K3 mais
détériorait K4. D'où une nouvelle interface apprenable
`K initial → K proposé → sélection de têtes → probabilité de correction /
probabilité de régression → KEEP ou corriger`.

## Ce qui est nouvellement implémenté

Code :
- `scripts/learn_v273_transition_combo_risk.py`
- `scripts/evaluate_v273_transition_combo_risk.py`
- `test/test_v273_transition_combo_risk.py`
- `.github/workflows/v273-transition-combo-risk.yml`

Le réseau dispose d'une banque explicite de sous-ensembles :

1. **H0 reste toujours ouvert**, car il est indispensable au contrat actuel
   de repli du modèle.
2. H1..H5 sont indépendamment sélectionnables ; pour C23/C32/C34/C43,
   l'adaptateur ne devient candidat que dans sa transition légale.
3. Quatre transitions dirigées possèdent **64 sous-ensembles distincts**
   incluant H0 ; les transitions sans adaptateur spécialisé disposent de
   **32 sous-ensembles** (les doublons sont masqués).
4. Chaque sous-ensemble dispose de son **propre audit d'entraînement**
   (nombre de propositions, corrections, régressions et neutres) et
   de caractéristiques du signal. Le routeur apprend **un score par
   événement, par transition et par sous-ensemble** grâce à un réseau
   neuronal avec softmax sur les seuls candidats compatibles.
5. Le risque appris `P(correction)` **et**
   `P(régression)` entrent directement dans les logits d'action. KEEP
   a une tête distincte, alimentée aussi par les anciennes têtes
   F_keep2/F_keep3/F_keep4/F_keep_any.
6. Aucune action K0/K1 n'est permise tant qu'un modèle compétent
   dans ces classes n'a pas été raccordé.

## Contrat de validation

Le vrai K de l'événement à évaluer n'est jamais fourni aux entrées du
routeur ; ses étiquettes servent seulement aux mesures après inférence.
Chaque fold externe entraîne son propre routeur. Les audits de sélections
affectés aux lignes du fold interne courant proviennent seulement des
**autres folds internes**. Pour le fold externe évalué, tous les audits
proviennent exclusivement du jeu externe d'entraînement.

**Limite importante** : les sorties OOF des spécialistes prédites sur un
autre fold peuvent provenir de modèles entraînés sur le fold interne
courant ; la stricte indépendance des représentations au niveau
de chaque ligne n'est pas encore prouvée. Les conditions externes
sont hors entraînement, mais les folds externes 0,1,2,4
ont déjà été inspectés pendant les recherches antérieures. Ce n'est
donc pas une validation entièrement nouvelle. Le fold3/player05 est
absent du cohort natif actuel, et il n'est pas utilisé.

[Évaluation GitHub Actions](https://github.com/Andriamarosoa/note/actions/runs/37755994598).

**Aucune promotion automatique de `freeze_local_combo`.**