# S852 — Résumé

Expérience autonome : entraîner deux TCN neuronaux causo-temporels (perte de forme d'onde L1 seule, L1 plus spectrale courte) à générer du vrai audio transposé, sans rejouer la banque à l'inférence. Le teacher est un pitch shift **offline** standard de chaque vraie prise, uniquement pour apprentissage et benchmark.

**Conclusion : échec.** Le modèle conserve la fondamentale originale malgré l'erreur de reconstruction en baisse. L'inférence retourne 256 échantillons en fonction de 256 nouveaux échantillons et de 1 024 échantillons passés, sans futur. Les scores Onset, Exact-K, Active Exact-K, Offset, Frame Pitch, Full Note et MIDI ne sont pas modifiés.

Voir [résultats S852](resultat.md). La prochaine boucle S853 étudie des pertes spectrales sur des séquences de huit blocs.
