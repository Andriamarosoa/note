# S866 — Résumé

**But :** Conversion de phase et fine-tuning causal.

**Méthode :** Les noyaux appris S863 sont reprojetés en approximation de phase minimale, puis affinés exclusivement sur TRAIN/CAL. La transformation ne fournit pas de note ni de méthode de transposition conventionnelle au moteur PCM256.

**Mesures :** CAL 4/6 ; 6 notes déjà examinées 3/6 (50 %). Distance log-mel médiane vs standard sur ces six notes 15,079 dB, contre 15,374 dB pour S863 et 11,707 dB pour S860. Centres d'énergie des filtres entraînés ~1,2–3,9 ms (proxy, pas mesure d'alignement d'attaques).

**Conclusion : Même exactitude de hauteur que S863 et légère amélioration spectrale, mais toujours moins fidèle que S860 ; pas conforme à « zéro retard » sur preuve acoustique, donc non promu.** Le détail figure dans [resultat.md](resultat.md). Les artefacts de poids/audio/source sont fournis dans l'archive de conversation, pas déposés en binaire dans cette branche. Exécution de recherche locale uniquement ; aucun nouveau run musical GitHub Actions ni gain officiel NOTE.
