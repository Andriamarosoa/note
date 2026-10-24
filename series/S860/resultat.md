---
serie: S860
statut: meilleur_labo_non_promu
type: entrainement
reference: Onset_PCM256_69.2584331_historique_inchange
promotion: false
execution: local_cpu_2026_10_24
---

# S860 — Pénalité harmonique impaire modérée

## Résultats observés
Fine-tuning indépendant de S858 avec pénalité spectrale d'harmoniques impaires **0,5** dans la perte TRAIN. Six CAL : log-mel médian **12,373 dB**, contre **14,487 dB** pour S858. Six notes OUTER déjà inspectées : log-mel **11,707 dB**, RMS/standard médian **46,0 %**, dominance 2f/f0 **+26,98 dB**, paire/impaire seulement **+0,13 dB**, erreur YIN ~11,89 demi-tons. Sur phrase externe 3,5 s : log-mel **10,655 dB**, corrélation enveloppe **0,789**, RMS **65,8 %** du standard. Amélioration sonore, pas de transposition musicale validée.

## Limites, séparation et contraintes


**Périmètre strict.** Modèles expérimentaux indépendants : exactement 256 nouveaux PCM à 44,1 kHz par hop, contexte passé uniquement, aucun futur ni hold, aucun changement de NOTE (Onset, Offset, Exact-K, Active Exact-K, Frame Pitch, Full Note) ni du MIDI. Métriques de labo sur banque S847 de 39 notes réelles (27 TRAIN / 6 CAL / 6 OUTER distinctes en hauteur, mais de même collection/guitariste) ; les 6 OUTER ont déjà été examinées en itération, donc ne valent pas jeu inédit après conception. Teacher standard +12 est **offline seulement**. Aucun modèle promu et aucun nouveau F1 officiel.

## Verdict
**Aucune promotion.**

## Suite
Comparer avec une pénalisation renforcée sur CAL avec des métriques communes ; refuser toute sélection fondée sur OUTER.
