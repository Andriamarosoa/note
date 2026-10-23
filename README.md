# Onset PCM256 — amélioration du F1

Priorité : améliorer les scores. Chaque décision reste au bloc courant de
**256 échantillons**, sans audio futur ni attente ajoutée.

La nouvelle récupération HGB obtient sur les événements archivés :

| Variante | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|
| Contrôle69 historique | 8127 | 4519 | 3140 | 67,971396 % |
| Run 39915038301, audio vérifié | 8012 | 4264 | 3255 | 68,062694 % |
| Nouvelle récupération HGB, replay audio en cours | 8050 | 4231 | 3217 | **68,370987 %** |

Le nouveau résultat reste à confirmer sur les 80 WAV. Il représente
38 TP supplémentaires et 33 FP de moins que la référence immédiate.
Les modèles, les politiques et les seuils sont déjà figés.

- [Protocole et hypothèse précédente écartée](series/ONSET_SCORE_RECOVERY_HGB256/resume.md)
- [Résultat et état de validation](series/ONSET_SCORE_RECOVERY_HGB256/resultat.md)
- [Rapports et prédictions scellées](series/ONSET_SCORE_RECOVERY_HGB256/evidence/)
- [Nouveaux poids de récupération](model/onset256_recovery_20261023/)
- [Poids principaux et enseignants conservés](model/onset256_research_39915038301/)

Cette branche concerne uniquement Onset. Les scores FIT80 restent exploratoires :
corpus déjà réutilisé, modèles amont historiques, aucune preuve de généralisation
externe. Les cinq modèles sont des plis de recherche ; le MIDI officiel n'est
pas promu.
