# S752 — Exécution ici des missions techniques A à F (audit initial)

**Date** : 2026-10-19
**Branche** : `codex/note-s752-six-role-integrity-audit`
**Important** : il s'agit de travaux menés par un seul assistant dans la présente conversation, **pas** de six sessions Work autonomes et indépendantes. Une auto-vérification contradictoire ne remplace pas une revue externe.

## Statut factuel des domaines

| Mission | Travail effectué | Limite |
|---|---|---|
| A — LIVE/latence | Test non-contiguïté `S708StrictLiveStream.push` ; test mutation future PCM pour `sharp_track` | Pas de mesure de latence microphone Windows/iRig ni de p95 machine réelle |
| B — DSP/harmoniques | **Correction** : faux harmoniques détectés dans les deux directions (octave +12 / -12 et partiels ±19/±24). Deux tests de faute d'octave E2/E3 | Indicateur de corrélation, pas preuve causale de confusion de fondamentale |
| C — ML / Exact-K | Test que K6 seul, sans énergie attaque/sustain, ne doit pas créer de faux NoteOn ; lecture du `GuidedSelector` et du `JointSelector` natifs | Aucune nouvelle tête entraînée ni expérimentation CAL32 de sélection dans cette PR |
| D — Validation | `per_true_k["6"]` aligne K≥6 sur la **saturation de S704** ; test à sept notes superposées et comparaison des 6 métriques ; rejet d'Offset orphelin dans `axis_audit` | Protocole S704 d'origine non modifié (préservation des références signées) |
| E — Code/CI | Workflow S752 avec 8 tests contradictoires plus suites préexistantes S743/S746/S747/S750 | CI verte nécessaire mais insuffisante pour prouver la qualité réelle sur guitare |
| F — Red Team | Contre-exemples : faux octave, K7 contre K6 saturé, MIDI NoteOff orphelin, attaque extra qui baisse le F1, mutation PCM futur | Revue menée par le même assistant : non indépendante institutionnellement |

## Anomalies de conception vérifiées par lecture

1. **P1 / diagnostic harmonique** : `scripts/s748_six_axis_failure_audit.py` comptait seulement les fausses **notes plus graves** que la note réelle. Une vraie E2 transcrite comme E3 (octave +12) était invisible à cet indicateur. S752 enregistre les deux sens et garde le compte combiné sans double compter la même note manquée.
2. **P2 / cohérence du protocole** : le rapport `per_true_k["6"]` employait un K réel non saturé alors que `scripts/s704_joint_metrics.py` compare `min(predicted_K,6)==min(true_K,6)`. Un cas synthétique K7→K6 paraissait incorrect dans la strate alors que le protocole officiel Exact-K le compte comme correct ; S752 harmonise les deux. `k_gt6_frames` reste disponible pour signaler cette limitation.
3. **P1 / architect. ML (à investiguer)** : `GuidedSelector.step` utilise toujours `self.active` et `self.armed` communs aux décisions finales, avec deux passes `JointSelector` puis `GuidedSelector`. L'existence de têtes indépendantes en amont n'implique donc pas six chaînes de décision MIDI indépendantes.
4. **P1 / impossibilité de promouvoir** : S749 a terminé ses 12 boucles FIT80/CAL32 et n'a trouvé aucune variante sûre. `depth5_risk40_standard` : Onset 74,7231 %, Exact-K 63,0202 %, Active-K 57,5362 %, Offset 76,7276 %, Full Note 57,0242 %, Frame Pitch 77,0092 %. Ne pas promouvoir.
5. **P1 / faible polyphonie** : S750 reproduit S708 sur GuitarSet CAL32 ; taux exacts des classes vraies : K2=35,684 %, K3=32,904 %, K4=32,672 %, K5=9,391 %, K6=6,397 %. 36 890 trames actives sous-comptées. Corriger seulement les fins de notes ne peut pas résoudre toute cette perte.

## Vérification nécessaire avant toute promotion

- Valider les huit tests S752 et les gardes existants dans une exécution GitHub Actions signée.
- Refaire un parcours CAL32 uniquement si une modification touche les décisions : S752 ne modifie que le **diagnostic post-hoc**, pas les décisions causales ni les six calculs de `scripts/s704_joint_metrics.py`.
- Tester les modifications futures d'apprentissage sur FIT80, sélectionner sur CAL32, figer, confirmer sur HELD non exploité pour le choix ; ne pas ouvrir TEST.
- Conserver S708/S707/S80 et la version LIVE inchangés tant que les six métriques et la latence ne sont pas garanties ; pas de distillation avant seuil >80%.

## Preuves

- S750 CAL32 signed score parité : https://github.com/Andriamarosoa/note/actions/runs/39207493606
- S749 essais Offset : https://github.com/Andriamarosoa/note/actions/runs/39206876051
- S752 PR DRAFT : https://github.com/Andriamarosoa/note/pull/83

**Verdict S752** : correction du **diagnostic** et tests, non évolution du modèle. Aucun score nouveau ne peut être revendiqué sur la simple base de ces changements.
