---
serie: S862
statut: termine_non_promu
type: audit
reference: Onset_PCM256_69.2584331_historique_inchange
promotion: false
execution: local_cpu_2026_10_24
---

# S862 — Résumé

**Action exécutée :** Comparaison commune et validation des contrats (audit).

**Mesures et verdict :** À métriques comparables sur les **mêmes six CAL** : log-mel S858=**14,487**, S860=**12,373**, S861=**12,427 dB** ; S860 retenu en labo sur CAL. Sur six OUTER déjà utilisées pour diagnostiquer les modèles : S858=13,702, S860=11,707, S861=11,654 dB. Les ratios pairs/impairs restent <1 dB dans les trois variantes : artefacts et risque sur accords non résolus. Les **4 tests unitaires locaux** de géométrie PCM256, continuité, causalité et indépendance de la banque passent. Démo WAV et scripts/checkpoints dans l'archive locale, pas intégrés à NOTE.

**Piste suivante :** Pour promotion : données sources/cibles réelles parfaitement auditées, test inédit inter-guitaristes, GT multi-note, évaluation d'intermodulation sur vrais accords, p99 Windows/iRig et audit complet de stabilité.

Détails dans [resultat.md](resultat.md).

**Périmètre strict.** Modèles expérimentaux indépendants : exactement 256 nouveaux PCM à 44,1 kHz par hop, contexte passé uniquement, aucun futur ni hold, aucun changement de NOTE (Onset, Offset, Exact-K, Active Exact-K, Frame Pitch, Full Note) ni du MIDI. Métriques de labo sur banque S847 de 39 notes réelles (27 TRAIN / 6 CAL / 6 OUTER distinctes en hauteur, mais de même collection/guitariste) ; les 6 OUTER ont déjà été examinées en itération, donc ne valent pas jeu inédit après conception. Teacher standard +12 est **offline seulement**. Aucun modèle promu et aucun nouveau F1 officiel.
