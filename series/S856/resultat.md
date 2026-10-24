---
serie: S856
statut: rejete
type: entrainement
reference: Onset_PCM256_69.2584331_historique_inchange
promotion: false
execution: local_cpu_2026_10_24
---

# S856 — Waveshaper neural spectral et harmonique

## Résultats observés
Deux architectures neuronales sans transposition DSP en inférence. Sur six notes OUTER, médiane erreur YIN environ **11,89 demi-tons**; la source reste dominante. Sortie RMS de la phrase externe environ **19,3 %** du standard pour spectral et **6,2 %** pour harmonique ; corrélations enveloppe ≈0,932 et 0,929. Ces enveloppes ne mesurent pas la justesse de hauteur. Causal PCM256 conforme dans le test local.

## Limites, séparation et contraintes


**Périmètre strict.** Modèles expérimentaux indépendants : exactement 256 nouveaux PCM à 44,1 kHz par hop, contexte passé uniquement, aucun futur ni hold, aucun changement de NOTE (Onset, Offset, Exact-K, Active Exact-K, Frame Pitch, Full Note) ni du MIDI. Métriques de labo sur banque S847 de 39 notes réelles (27 TRAIN / 6 CAL / 6 OUTER distinctes en hauteur, mais de même collection/guitariste) ; les 6 OUTER ont déjà été examinées en itération, donc ne valent pas jeu inédit après conception. Teacher standard +12 est **offline seulement**. Aucun modèle promu et aucun nouveau F1 officiel.

## Verdict
**Aucune promotion.**

## Suite
Essayer une architecture non linéaire explicitement symétrique, tout en auditant séparément hauteur harmonique et YIN.
