# S155 — Correction vérifiée du graphe complet de dépendances S80

La carte S148 représentait **à tort S11 comme parent direct de S18**.
La lecture des sources originales archivées a prouvé le chemin réel :

```text
S5 d'origine → S8 → S9
                  ├→ S13 ─┐
                  └→ S14 ─┴→ S15 safe_OR → S16 → S17 → S18
                  └→ S11 (famille parallèle, PAS ancêtre S18)
S18 + familles parallèles → 16 S43/S44 → S45 risques → S46
S46 + PR16 → S69 → S70/S71/S72 → S79 K1 → S80 K0/K2
                                                → validation indépendante → promotion
```

**28 nœuds**, **16 familles S45**, validés automatiquement en CI. Pas
de nouvelle condition cachée annoncée après coup : toute différence de
source doit modifier explicitement ce registre versionné.

| Étape | État véritable sur nouveau WAV | Source de preuve |
|---|---|---|
| wav_groups | verified_new_audio | S83-S85: WAV to 57 native GOAT audio events |
| PR16 | verified_new_audio | S85: original 36 voting policies + 12 probabilities |
| S4_control111 | adapted_new_audio | S116/S117: original control architecture, 111D input |
| S5_original527 | historical_only | S125: missing original 416D acoustic constructor; 1882/15952 mismatched attempts |
| S5_student135 | rejected_on_exposed_smoke | S133-S137: new student; no gain GOAT 44/57 global, poly down 2/3 -> 1/3 |
| S8_original_parent | missing_new_audio | Original S8 parent depends on S5 global/poly heads and old piece-excluded routing; not restored on GOAT |
| S9_original_parent | missing_new_audio | Original source scripts/audit_v273_series9_posthoc.py veto_P_ge5_and_G_le3 needs original S8 proposal plus original S5 global/poly |
| S13_K0_birth | adapted_new_audio | S153 76 original risk checkpoints S13 executed; 26 policies, S9 adapted, not exact original |
| S14_K0_rollback | adapted_new_audio | S153 76 original risk checkpoints S14 executed; 32 policies, S9 adapted, not exact original |
| S15_safe_OR | adapted_new_audio | S153 exact original nine union formulas; S15 safe_OR gives [45,10,2,0,0,0,0] on GOAT from adapted S9 |
| S16_K12_K23 | adapted_new_audio | S154 152 original trained checkpoint heads and 108 veto policies, stage parent matches original selected S15 policy but S9 adapted |
| S17_K32_K34 | adapted_new_audio | S154 152 original trained checkpoint heads and 108 veto policies, stage parent matches original selected S16 policy but S9 adapted |
| S11_original_votes | adapted_new_audio | S149 original trained S11 76 risk checkpoints and 24 veto policies on GOAT; new S9 adapter p=PR16#22; parallel candidate source, not ancestor of S18 |
| S18_parent | adapted_new_audio | S154 exact original 108 K43/K21 veto policy formulas and 140 original trained checkpoints; needs genuine original S9; 0 eligible GOAT changes |
| S43_S44_16_champions | historical_only | 16 archived S43/S44 champions, including parallel S11 head; some original sources run on GOAT, full 16 not yet equivalent |
| S12_original_risks | adapted_new_audio | S143: 19 original cross-piece logistic checkpoints; S9 eligibility missing |
| S38_experts | adapted_new_audio | S142/S144: 16 original trained model checkpoints |
| S40_temporal | adapted_new_audio | S141/S144: four original 42x49 CNN checkpoints |
| S41_gate | adapted_new_audio | S144: 19 original fusion checkpoints, new parent, S145 57/57 KEEP |
| S45_family_risks | historical_only | S45 historical piece-excluded family ridge; no new-piece model/aggregation policy |
| S46_193 | logic_ready_inputs_missing | S88 label-free exact 193-policy decoder from 16 family proposals, risk and votes |
| S69 | historical_only | S89 exact historical veto rules are keyed by old outer folds |
| S70_S71_S72 | historical_only | S72 risk and poly guard fitted/tested on exposed historical folds |
| S79_K1 | historical_only | Four saved K1 logistic models with fold-specific choices; no novel-piece deployment rule |
| S80_K0_K2_tail | logic_ready_inputs_missing | S147: zero mismatches on 59309 historical rows, 46 K0 and 61 K2 actions |
| S80_full | missing_new_audio | No full end-to-end predictor producing new-wave outputs through all original inputs |
| independent_ExactK | pending_independent | External preregistered IDMT-SMT-GUITAR V2 heldout not performed |
| promote_S80 | pending_independent | Must beat freeze_local_combo in global and poly Exact-K on unseen pieces |

## Récupérations terminées depuis S148

- **S149** 76 poids S11 authentiques sur GOAT (parent S9 adapté).
- **S150** 140 poids S18 authentiques sur GOAT (S11 parent provisoire).
- **S151** archives originales S11/S16/S17/S18 préservées par release GitHub.
- **S152** 444 modèles S16/S17/S18 exécutés séquentiellement, S11 parent provisoire.
- **S153** 76 poids S13 + 76 poids S14 et les neuf règles S15 safe OR/AND.
- **S154** S15 original (adapté S9) alimente les 152 S16, 152 S17 et 140 S18.
  Sortie GOAT : **45 K0, 10 K1, 2 K2** ; aucun veto déclenché.

## Limites et porte de promotion

Le parent **S8/S9/S5 original** est toujours manquant sur un nouveau WAV :
l'adaptation PR16 policy22 à S9 ne constitue pas une preuve d'identité
avec S9 historique. Les sorties S154 sont donc **S80-deploy-v1
expérimentales**, pas un S80 officiel reconstruit.

16 familles S43/S44, leurs risques S45, S46 193, S69/72 et S79
restent à connecter et vérifier *ensemble* sur une musique inconnue.
Les routes finales S80 K0/K2 sont déjà bit-exactes en S147
sur 59 309 décisions historiques. Il faut une évaluation
**indépendante**, préenregistrée, sur de nouveaux morceaux polyphoniques.

**Référence officielle : `freeze_local_combo`; S80 non promu.**

## S156–S159 — Sources originales S8 et S13/S14 archivées

- [S156 — sauvegarde S13/S14 réussie](https://github.com/Andriamarosoa/note/actions/runs/38195953248) :
  152 checkpoints originaux et preuves S153/S154 conservés dans la
  [release de recherche](https://github.com/Andriamarosoa/note/releases/tag/v273-original-S13-S14-trained-S153-S154-cascade-archive).
- [S157 — vrais checkpoints S8 exécutés](https://github.com/Andriamarosoa/note/actions/runs/38196167459) :
  19 modèles originaux de sélection 28D + règle S9 originale ; une seule
  divergence g/p sur 57 groupes, aucune intervention, S9=[45,10,2,0,0,0,0].
- [S158 — comparaison source S9 indépendante des labels](https://github.com/Andriamarosoa/note/actions/runs/38196307750) :
  **0 différence sur 57 IDs** entre le S9 produit par S157 et les vecteurs S9
  gelés de S153/S154. Pas besoin de relancer les 596 forwards S13–S18.
- [S159 — archivage S8 réussi](https://github.com/Andriamarosoa/note/actions/runs/38196392548) :
  19 poids S8 authentiques avec preuves S157/S158 et SHA-256 conservés dans la
  [release permanente de recherche](https://github.com/Andriamarosoa/note/releases/tag/v273-original-S8-weights-S157-S158-GOAT-proof).

**État sur nouvel audio : S8 et S9 passent de « indisponibles » à «
exécutables avec entrées S5 adaptées ».** Ce changement de statut
ne prouve jamais l'équivalence avec les sorties S5 global/poly originales.

Le S80 historique complet reste **non déployé**, non évalué
indépendamment et **non promu**. Le verrou racine 527D S5 et les
16 producteurs/risques S45 sont toujours présents.


## S161 — Vrai parent S18 pour les modèles S41 récupérés

[Run S161 réussi](https://github.com/Andriamarosoa/note/actions/runs/38196667499).

S144 avait pris à tort `freeze_local_combo` comme parent de
conservation de la tête S41. En S161, nous avons conservé les vrais
scores des 16 experts S38 et quatre S40, conservé les 19 checkpoints
S41 originaux, mais remplacé la classe parent par la sortie S18
du pipeline adapté S154.

**Résultat :** 19 réseaux exécutés sur 57 groupes ; **57 KEEP / 57**,
aucune modification déclenchée par S41. **Quatre décisions diffèrent de
S144** uniquement parce que son parent était différent. Le poids moyen
KEEP passe de **0,912681** à **0,914608**.

Ce résultat négatif est conservé. S18 dépend toujours du S9 adapté,
pas du S5 historique avec son constructeur 527D ; il ne permet
aucune promotion S80.
