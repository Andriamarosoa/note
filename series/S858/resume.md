---
serie: S858
statut: en_audit_non_promu
type: entrainement
reference: Onset_PCM256_69.2584331_historique_inchange
promotion: false
execution: local_cpu_2026_10_24
---

# S858 — Résumé

**Action exécutée :** MLP pair et filtres causaux appris (entrainement).

**Mesures et verdict :** La variante `evenfilter` possède trois filtres convolutifs appris (dilatations 1/3/9). Six notes : **+26,60 dB** de dominance médiane énergie 2f vs f0, toutes six >10 dB, mais **11,89 demi-tons** d'erreur YIN. Niveau RMS de la phrase réelle **27,6 %** du standard ; log-mel médian sur les six notes **13,70 dB**. Le signal n'est pas un octave musicalement prouvé.

**Piste suivante :** Contrôler la somme des harmoniques impaires/paire, les clics de jonction, le CPU et les artefacts polyphoniques.

Détails dans [resultat.md](resultat.md).

**Périmètre strict.** Modèles expérimentaux indépendants : exactement 256 nouveaux PCM à 44,1 kHz par hop, contexte passé uniquement, aucun futur ni hold, aucun changement de NOTE (Onset, Offset, Exact-K, Active Exact-K, Frame Pitch, Full Note) ni du MIDI. Métriques de labo sur banque S847 de 39 notes réelles (27 TRAIN / 6 CAL / 6 OUTER distinctes en hauteur, mais de même collection/guitariste) ; les 6 OUTER ont déjà été examinées en itération, donc ne valent pas jeu inédit après conception. Teacher standard +12 est **offline seulement**. Aucun modèle promu et aucun nouveau F1 officiel.
