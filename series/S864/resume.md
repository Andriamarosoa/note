# S864 — Résumé

**But :** Filtres plus longs et phase minimale.

**Méthode :** Banc de filtres appris étendu à 1 025 coefficients, initialisation de phase minimale, sélection par CAL. La consommation CPU et le retard de groupe sont audités séparément de l'hop PCM256.

**Mesures :** CAL 4/6 ; 6 notes déjà examinées 2/6 (33,3 %). Centres d'énergie des coefficients ~3,31–7,20 ms selon filtre après entraînement ; p99 local ≈0,759 ms. Cette mesure de filtre n'est pas une latence audio réelle.

**Conclusion : Régresse par rapport à S863 sur le jeu déjà consulté ; pas de promotion. Les poids restent dans l'archive pour audit.** Le détail figure dans [resultat.md](resultat.md). Les artefacts de poids/audio/source sont fournis dans l'archive de conversation, pas déposés en binaire dans cette branche. Exécution de recherche locale uniquement ; aucun nouveau run musical GitHub Actions ni gain officiel NOTE.
