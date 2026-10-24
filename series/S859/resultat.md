---
serie: S859
statut: termine_non_promu
type: audit
reference: Onset_PCM256_69.2584331_historique_inchange
promotion: false
execution: local_cpu_2026_10_24
---

# S859 — Audit multi-métriques causal et sonore

## Résultats observés
S858 : médiane **+26,60 dB** de 2f/f0 mais seulement **+0,44 dB** de somme harmoniques paires/impaires, signe de produits parasites ; distance log-mel sur phrase externe 14,24 dB. P99 Python CPU local ≈**1,17 ms** sur 220 blocs de 256 ; ceci ne prouve pas Windows/iRig. Tests : 256 strict, pas de futur. L'estimateur YIN et la puissance 2f restent contradictoires et ne permettent pas une promotion.

## Limites, séparation et contraintes


**Périmètre strict.** Modèles expérimentaux indépendants : exactement 256 nouveaux PCM à 44,1 kHz par hop, contexte passé uniquement, aucun futur ni hold, aucun changement de NOTE (Onset, Offset, Exact-K, Active Exact-K, Frame Pitch, Full Note) ni du MIDI. Métriques de labo sur banque S847 de 39 notes réelles (27 TRAIN / 6 CAL / 6 OUTER distinctes en hauteur, mais de même collection/guitariste) ; les 6 OUTER ont déjà été examinées en itération, donc ne valent pas jeu inédit après conception. Teacher standard +12 est **offline seulement**. Aucun modèle promu et aucun nouveau F1 officiel.

## Verdict
**Aucune promotion.**

## Suite
Apprendre explicitement à réduire les harmoniques impaires, sans traitement de fréquence codé en dur au runtime.
