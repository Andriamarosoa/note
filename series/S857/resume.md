---
serie: S857
statut: rejete
type: entrainement
reference: Onset_PCM256_69.2584331_historique_inchange
promotion: false
execution: local_cpu_2026_10_24
---

# S857 — Résumé

**Action exécutée :** MLP neural à symétrie paire (entrainement).

**Mesures et verdict :** La variante `even` apprend un waveshaping sans règle de transposition DSP fixe. Six notes déjà connues du cycle : dominance médiane seconde harmonique sur fondamentale **+5,53 dB**, erreur médiane YIN ≈**11,88 demi-tons** ; corrélation enveloppe sur vraie phrase 3,5 s 0,947 et RMS/standard 0,304. Insuffisant pour prouver +12.

**Piste suivante :** Ajouter des filtres causaux apprenants à la non-linéarité symétrique et un audit de tous les harmoniques.

Détails dans [resultat.md](resultat.md).

**Périmètre strict.** Modèles expérimentaux indépendants : exactement 256 nouveaux PCM à 44,1 kHz par hop, contexte passé uniquement, aucun futur ni hold, aucun changement de NOTE (Onset, Offset, Exact-K, Active Exact-K, Frame Pitch, Full Note) ni du MIDI. Métriques de labo sur banque S847 de 39 notes réelles (27 TRAIN / 6 CAL / 6 OUTER distinctes en hauteur, mais de même collection/guitariste) ; les 6 OUTER ont déjà été examinées en itération, donc ne valent pas jeu inédit après conception. Teacher standard +12 est **offline seulement**. Aucun modèle promu et aucun nouveau F1 officiel.
