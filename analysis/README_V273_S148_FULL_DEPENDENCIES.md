# S148 — Contrat unique pour la promotion de S80

**Une seule liste de dépendances**, testée en CI et maintenue dans
`analysis/v273_s80_full_dependency_contract.json`.

Deux chemins clairement séparés : **S80 historique exact** exige les vrais
producteurs S5 527D et votes S18 ; **S80-deploy-v1** pourra substituer des
composants, mais devra être validé comme un nouveau modèle.

| Composant | État sur un nouvel audio | Preuve |
|---|---|---|
| wav_groups | verified_new_audio | S83-S85: WAV to 57 native GOAT audio events |
| PR16 | verified_new_audio | S85: original 36 voting policies + 12 probabilities |
| S4_control111 | adapted_new_audio | S116/S117: original control architecture, 111D input |
| S5_original527 | historical_only | S125: missing original 416D acoustic constructor; 1882/15952 mismatched attempts |
| S5_student135 | rejected_on_exposed_smoke | S133-S137: new student; no gain GOAT 44/57 global, poly down 2/3 -> 1/3 |
| S11_original_votes | missing_new_audio | Genuine original S11 global/poly votes not executable on an unfamiliar recording |
| S18_parent | missing_new_audio | Historical S17/S18 weights and veto parent exist; original new-wave upstream not reconstructed |
| S43_S44_16_champions | historical_only | 16 original champions recorded by S43/S44, only partially functional on GOAT |
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

## Les 16 familles S45 (aucune omission autorisée)

- S1_acoustic : historical_vector_only
- S2_source : historical_vector_only
- S3_repair : historical_vector_only
- S4_repair : historical_vector_only
- historical_repair_coherent : historical_vector_only
- S25_Bfirst : historical_vector_only
- S24_Bfirst : historical_vector_only
- S40_temporal_CNN : adapted_checkpoint_forward_not_original_full_family
- S19_recurrent_ABA : historical_vector_only
- S11_votes : historical_vector_only
- S38_morphology : adapted_checkpoint_forward_not_original_full_family
- S12_logistic : adapted_checkpoint_forward_not_original_full_family
- S20_repeated_ABA : historical_vector_only
- S27_early_stop : historical_vector_only
- S41_fusion : adapted_checkpoint_forward_not_original_full_family
- S35_multiroute : historical_vector_only

## Portes finales non négociables
1. Define final new-wave model version, parents, routing order and policies before opening labels
2. Run fully unseen-audio S4/S5->S18->16 S45->S46->S69->S72->S79->S80
3. Freeze model, code, checkpoints and independent predictions with SHA256
4. Evaluate song-disjoint unseen and unexposed guitar polyphony data against freeze_local_combo
5. Report global, K0-K6, K2plus, paired corrections and regressions with uncertainty
6. Promote only with independently verified better global AND poly Exact-K and no unresolved regressions

**État : NON PROMU.** GOAT (deux courts morceaux) est désormais un jeu
exposé et ne sera pas utilisé pour régler les seuils puis prétendre
à un test indépendant.

**S147 est fait** : K0/K2 reproduisent exactement 46+61 interventions
sur 59 309 événements (sans lire de vérités). Ces règles seules
ne constituent pas le S80 complet.

**S148 restaure la source originale S18** et valide son selftest,
sans prétendre rétablir automatiquement ses parents et checkpoints
sur de nouveaux audios.

## Résultats vérifiés des deux workflows

- [S147 — reproduction exacte de la queue K0/K2](https://github.com/Andriamarosoa/note/actions/runs/38194275444) :
  **59 309 prédictions identiques**, **0 divergence**, 46 actions K0 et 61 actions K2.
  Entrée et sortie d'exécution sans vérités terrain. **Ce n'est pas le S80 complet**.
- [S148 — graphe exhaustif, 21 dépendances et 16 familles](https://github.com/Andriamarosoa/note/actions/runs/38194433107) :
  contrôle de présence des fichiers, absence de cycles, non-promotion,
  restauration et selftest réussis du script S18 d'origine.
- Sur les 21 nœuds du chemin historique, **2** sont confirmés directement
  sur nouvel audio sans remplacement de leurs entrées ; les autres sont
  en état adapté, historique, logique disponible sans entrées, ou bloqué.
  Le nombre de nœuds **n'est pas un pourcentage d'avancement**.
- Les 58 checkpoints déjà sauvés en S146 restent conservés.
- Le statut de référence ne change **pas** : `freeze_local_combo`.
