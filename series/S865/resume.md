# S865 — Résumé

**But :** Affinage explicite de cohérence harmonique.

**Méthode :** Fine-tuning des poids S863 avec pénalité de contraste des harmoniques GT (uniquement pendant l'entraînement), sans connaissance de la note à l'inférence. Sélection du checkpoint sur CAL.

**Mesures :** CAL 4/6 ; 6 notes déjà examinées 3/6. Log-mel médian CAL 14,622 dB pour la variante entraînée ; ne pas comparer ce chiffre à une mesure OUTER. Aucune amélioration de réussite YIN par rapport à S863.

**Conclusion : Non retenu comme gain validé ; aucune promotion.** Le détail figure dans [resultat.md](resultat.md). Les artefacts de poids/audio/source sont fournis dans l'archive de conversation, pas déposés en binaire dans cette branche. Exécution de recherche locale uniquement ; aucun nouveau run musical GitHub Actions ni gain officiel NOTE.
