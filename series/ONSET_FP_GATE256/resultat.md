# Résultat — H_FP_EXPERT_GATE_02, Onset PCM256

**Nouvelle tête d'arbitrage d'experts 69D ↔ 74D : PRÉENREGISTRÉE mais pas encore évaluée en replay au moment de la création de cette série.**

### Référence signée historique

- S613 256 GuitarSet FIT80 : TP8099 FP4600 FN3168 F1 67.5874155%.
- Trois têtes ErrorPrediction69 : TP8054 FP4411 FN3213 F1 67.8745997%.
- Quatre têtes avec FNRecovery : TP8160 FP4605 FN3107 F1 67.9094541%.
- Contrôle FP69 (meilleur F1 expérimental non promu) : **TP8127 FP4519 FN3140 F1 67.9713963%**.
- Contrôle FP70 : TP8137 FP4549 FN3130 F1 67.9413852%.
- Temporal FP74 : TP7948 FP4290 FN3319 F1 67.6281642%, **hypothèse #142 non validée**.

### Audit de la cause présumée

La tête 74D refuse Rock3 **128 TP et 131 FP** dans les annotations des événements d'origine. Seules 27 des 128 vraies attaques perdues avaient un parent harmonique native Main attaqué moins de 80ms avant. Les 101 autres TP ont été refusés alors que le motif harmonique étudié est ABSENT. Contre-exemples montrant que la tête 74D apprenait des erreurs plus générales que l'hypothèse.

### Nouvelle intervention TESTABLE

- La nouvelle tête `FPExpertGate`, vraiment entraînée sur 11D causaux, arbitre seulement les désaccords des 69D/74D, sans autre modification.
- Même original 69/70/74 weights SHA. Entraînement gate sur 8 pistes CAL, validation/choix de cut uniquement sur les 8 autres CAL, OUTER composition jamais utilisée avant gel.
- Si insuffisance d'exemples contradictoires dans CAL, gate désactivé et 69D inchangé pour ce pli.
- Critère écrit AVANT test : **vrai F1 >67.9713963% ET vrais FP <4519** sur les 80 WAV PCM256 ; anciennes références exactement reproduites.
- Si le gate échoue, `REFUTED_OR_INCOMPLETE`, ne pas promouvoir. Le proxy d'événements calibré sur CAL n'est PAS le F1 Live.

**Statut à ce stade : résultat nouveau non encore vérifié, aucune promotion, aucun changement des cinq autres caractéristiques ni du MIDI officiel.**
