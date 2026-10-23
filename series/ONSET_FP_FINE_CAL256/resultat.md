# H_FP_CAL_GRID_04 — résultats scientifiques et référence gelée

**Statut : résultat du replay avec seuils fins non encore vérifié.** Ne jamais confondre la calibration CAL par événements avec le véritable F1 causal.

## Échec confirmé des deux expériences précédentes (replay réussi techniquement)

- [Run #39900882849](https://github.com/Andriamarosoa/note/actions/runs/39900882849): 10 MLP69 entraînés et 2 x 80 WAV originaux PCM256 totalement rejoués, deux garde-fous scientifiques ont échoué volontairement. `track_uniform` et `track_class` chacun TP8160 FP4605 FN3107 F1 67.9094541 %.
- Meilleur contrôle69 historique : **TP8127 FP4519 FN3140 F1 67.9713963 %** ; il reste meilleur et non promu au MIDI.
- POUR CHACUN des deux bras, cinq choix CAL `calibrated_cut = 0.0`, donc aucune des dix têtes discriminantes n'a produit de veto et les anciennes sorties quatre-têtes ont été inchangées.
- **Le seuil non nul initial 0.05 était trop haut par rapport à la limite de 2.5% de TP perdus** ; ce problème de résolution rend l'audit du mécanisme d'échantillonnage INCOMPLET, non scientifiquement réfuté. Nouvelle hypothèse préenregistrée [#145](https://github.com/Andriamarosoa/note/issues/145).

## Critère invariable pré-enregistré pour chaque nouvelle branche CAL-fine

- Utiliser exactement les 5 MLP69 précédemment entraînés par bras, vérifier chaque SHA, zéro nouveau modèle refitté. 70D et 74D inchangés.
- CAL uniquement avec garde inchangée : ≥5 FP évités, ≤2.5% des TP CAL rejetés, F1 CAL strictement amélioré. Grille plus précise `0.00001–0.05` + ancienne grille, fixée avant résultat.
- **Vrai replay causal** 80 WAV /306536 décisions, TP/FP/FN/F1 réel validés. Doit dépasser **67.9713963% F1** et avoir **strictement moins de 4519 FP**.
- Si échec : verdict `REFUTED_OR_INCOMPLETE`, publier la table de compromis et les pistes régressées, ne pas manipuler les seuils après avoir consulté OUTER. Référence gelée, cinq autres axes et production MIDI inchangés.
- Les OUTER FIT80 ont été explorées auparavant historiquement; même un score favorisé n'est pas une preuve de généralisation complètement indépendante.

**Aucun score nouveau promu ni inventé.**
