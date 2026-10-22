# Résultat — audit des faux négatifs Onset256

**État : non mesuré tant que le workflow scientifique GitHub n'est pas terminé.**

## Références démontrées

- S613 natif256 sur GuitarSet FIT80, 80 pistes, 306536 décisions : **TP8099, FP4600, FN3168**, F1 67,5874155 %.
- Troisième tête ErrorPrediction MLP69 après correction de feedback, mêmes WAV et poids : **TP8054, FP4411, FN3213**, F1 67,8745997 %. **45 FN supplémentaires** par contestation de vrais NoteOn, aucun FN original récupéré.
- Audits précédents S684 de 12 autres pistes DEV : sur FN582, 501 impliquaient un posterior Onset sous le seuil. Ce constat est une piste, **pas** une mesure native256 validée sur les 80 FIT.

## Protocole signé

Hypothèse `H_FN_MODEL_01` pré-enregistrée : >=50% des FN S613 ont P(onset) < seuil S613 dans leur fenêtre annotée ±80 ms. Deux catégories composent ce groupe : `acoustic_onset_below_0_12` et `main_onset_below_calibrated_threshold`. Si non vérifiée : diagnostic **REFUTED_OR_INCOMPLETE**, pas de correction silencieuse de seuil.

Trois stratégies de proposition causales indépendantes `sensitive/moderate/cautious` choisies **avant** lecture des annotations GT de cette expérience, pour mesurer la couverture maximale d'événements candidats correspondant aux 3168 GT-FN originaux. Aucune de ces propositions n'est un vrai nouveau résultat Live; **l'oracle de matching** voit la vérité terrain uniquement après le replay, jamais à l'inférence.

Les artefacts à produire : `report.json`, `false_negatives.jsonl`, `candidate_proposals.jsonl` (avec features 69D sans GT dans l'inférence) et `main_events.jsonl`, tous sur les mêmes 80 WAV/306536 PCM256, SHA vérifiés.

Le main, la tête S701, la tête ErrorPrediction MLP69 et les autres cinq caractéristiques restent strictement inchangés. Aucun poids entraîné1024/S708, aucune attente supplémentaire, aucune promotion ni MIDI officiel.
