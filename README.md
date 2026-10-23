# Onset PCM256 — F1 68,370987 % confirmé

La nouvelle récupération HGB est validée sur **80 WAV originaux** et
**306536 blocs de 256 échantillons**, sans audio futur ni attente ajoutée.

| Variante | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|
| Contrôle69 historique | 8127 | 4519 | 3140 | 67,971396 % |
| Run 39915038301 | 8012 | 4264 | 3255 | 68,062694 % |
| **Nouvelle récupération HGB** | **8050** | **4231** | **3217** | **68,370987 %** |

Gain confirmé sur le replay audio : **+0,308293 point** contre le run précédent,
**38 vraies attaques supplémentaires et 33 fausses attaques de moins**.
Chaque paire `(sample, pitch)` correspond aux prédictions figées avant le replay.

La précision atteint **65,548408 %** et le rappel **71,447590 %**.
50 pistes progressent, 24 régressent et 6 restent identiques en F1.
Quatre compositions sur cinq progressent ; Rock3 perd 0,081706 point.

- [Résultat détaillé](series/ONSET_SCORE_RECOVERY_HGB256/resultat.md)
- [Rapport du replay audio](series/ONSET_SCORE_RECOVERY_HGB256/evidence/fresh_audio_report.json)
- [Protocole et hypothèse précédente écartée](series/ONSET_SCORE_RECOVERY_HGB256/resume.md)
- [Nouveaux poids de récupération](model/onset256_recovery_20261023/)
- [Poids principaux et enseignants](model/onset256_research_39915038301/)
- [81 tests GitHub réussis](https://github.com/Andriamarosoa/note/actions/runs/39923540639)

L'entraînement et le replay audio ont été exécutés localement ; le workflow
GitHub ci-dessus vérifie les tests. Le workflow propose également une
reproduction complète sur demande.

La priorité reste l'amélioration des scores. Cette branche concerne uniquement
Onset. FIT80 est un corpus déjà exploré avec des modèles amont historiques :
ce résultat ne prouve pas une généralisation externe. Les cinq modèles restent
des plis de recherche, sans promotion du MIDI officiel.
