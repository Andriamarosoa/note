---
serie: S861
statut: rejete_sur_cal
type: entrainement
reference: Onset_PCM256_69.2584331_historique_inchange
promotion: false
execution: local_cpu_2026_10_24
---

# S861 — Pénalité harmonique impaire renforcée

## Résultats observés
Fine-tuning de S858 avec pénalité **2,0**, même source, même splits. Six CAL : log-mel médian **12,427 dB**, donc moins bon que S860 sur CAL (12,373 dB). Six OUTER déjà inspectées : log-mel **11,654 dB**, RMS/standard **47,1 %**, dominance cible **+27,08 dB**, paire/impaire **+0,15 dB**, YIN ≈11,89 demi-tons. Sur phrase réelle : log-mel 10,589 dB, corrélation d'enveloppe 0,790, RMS 68,1 %. Les valeurs OUTER meilleures ne justifient pas une promotion ni un changement de sélection CAL.

## Limites, séparation et contraintes


**Périmètre strict.** Modèles expérimentaux indépendants : exactement 256 nouveaux PCM à 44,1 kHz par hop, contexte passé uniquement, aucun futur ni hold, aucun changement de NOTE (Onset, Offset, Exact-K, Active Exact-K, Frame Pitch, Full Note) ni du MIDI. Métriques de labo sur banque S847 de 39 notes réelles (27 TRAIN / 6 CAL / 6 OUTER distinctes en hauteur, mais de même collection/guitariste) ; les 6 OUTER ont déjà été examinées en itération, donc ne valent pas jeu inédit après conception. Teacher standard +12 est **offline seulement**. Aucun modèle promu et aucun nouveau F1 officiel.

## Verdict
**Aucune promotion.**

## Suite
Conserver S860 uniquement comme meilleur candidat de laboratoire selon la comparaison CAL commune.
