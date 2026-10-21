---
serie: S825
statut: en_cours
type: experience
reference: S708
branche: codex/note-s825-private-crossfit-pitch-midi256
run_id: en_attente
commit_sha: en_attente
promotion: false
---
# S825 — Boucles de têtes Active Exact-K véritablement indépendantes sur PCM MIDI256

## Objectif

Après l'ablation S824, apprendre une nouvelle représentation probabiliste de la présence des 49 hauteurs **depuis le PCM natif MIDI256**, sans postérieurs S708, sortie Frame Pitch, ni historique d'événements MIDI. Tester **trois modèles de cardinalité entraînés sur des prédictions de hauteurs hors entraînement**, et comparer leurs scores sur les mêmes 249411 trames GT actives. Rejeter les solutions régressives.

## Hypothèse

Un head acoustique `pitch_activity` formé de zéro avec labels FIT et partagé **uniquement à l'intérieur** de l'architecture Active-K peut restaurer une partie des informations de S708. Un deuxième `cardinality` apprend des cartes de probabilité **OOF**, et les variantes d'indices harmoniques et de variations temporelles sont des ablations falsifiables. Aucune tête ne réutilise les poids ou prédictions des cinq autres caractéristiques.

## Modifications

`src/causal_note/s825_crossfit_activek.py` avec `pitch_activity` causal entraîné par `HistGradientBoostingClassifier`, puis classifieurs K0–K6 `StandardScaler+LogisticRegression` utilisant l'activité des 49 hauteurs, leur compétition fondamentale/harmonique et le changement depuis le **précédent hop 256**, sans futur. `scripts/s825_train_crossfit_midi256.py` reconstruit 80 FIT originaux à partir des traces et DSP SHA; effectue 5 splits outer de compositions (48 TRAIN/16 inner-CAL/16 outer-TEST), puis **3 sous-plis internes d'entraînement**: chaque prédiction OOF de hauteur pour une composition TRAIN provient d'un modèle entraîné sur les deux autres compositions TRAIN. Une nouvelle tête privée `pitch_activity` est enfin entraînée sur les 48 TRAIN uniquement pour les sorties CAL et OUTER. Chaque variante cardinalité a ses propres poids; sélecteur issu de CAL exclusivement.

## Protocole

80 pistes GuitarSet, PCM mono original 44,1 kHz, `feature_hop_samples=decision_hop_samples=metric_hop_samples=256`, 306536 frames, 249411 GT K>0, H0 142213 corrects et S815 141946 corrects comme comparateurs. Préenregistrement de trois modes cardinalité `pitch_stats`, `pitch_harmonic`, `pitch_temporal`. Poids d'échantillonnage par vraie classe K conservés après échantillonnage; GT n'entre jamais en inférence. Calibration sur 16 autres pistes exclusivement. Mesurer score, under, over, vrai K1..6, K0 diagnostic et chaque composition. Référence S823 76,8775% SHADOW provient de S708 1024 et n'est **pas une baseline indépendante**.

## Audits obligatoires

Signature 80 enregistrements et cache S813, 5 splits 48/16/16 disjoints + 3× OOF sans composition commune au pitch teacher, invariance préfixe/causalité, entraînement de chaque cardinalité exclusivement sur prédictions crossfit, 3 têtes aux poids distincts. Test de contribution à 0 / 1 / 2 combinaisons supplémentaires dans chaque pli ; aucune sélection à posteriori sur outer. Aucun changement S708, MIDI LIVE, Onset, Offset, Frame Pitch, Exact-K ou Full Note. La résolution 5,805ms ne garantit pas la latence hardware: mesure distincte requise. Les 80 pistes FIT80 sont historiquement explorées, jamais une preuve de généralisation indépendante.

## Statut

En cours: pas de résultat S825 certifié avant les logs GitHub Actions authentifiés. Chaque candidat est SHADOW, promotion=false.
