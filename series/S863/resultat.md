---
serie: S863
statut: termine_local_non_promu
type: experience
date: 2026-10-24
promotion: false
---

# S863 — Première amélioration de hauteur par banc de filtres neuronal causal

## Expérience
Le réseau apprend sept filtres FIR de 513 coefficients (initialisés par passe-bas), une porte de sélection dépendant du signal PCM et une transformation neuronale paire. Toutes les sorties restent 256 PCM natifs, sans banque de clips ni shift classique à l'inférence.

## Résultats réels
TRAIN 27 notes ; CAL 4/6 ; 6 notes déjà examinées 3/6 (50 %). Sur celles-ci, distance log-mel médiane vs standard 15,374 dB, contre 11,707 dB pour S860. P99 calcul réseau seul CPU local ≈0,434 ms ; centres d'énergie des filtres ~254–258 samples (~5,8 ms).

## Décision
**Hausse de justesse sur les notes réutilisées, mais dégradation de fidélité audio et retard de phase probable. Non promu.**


## Cadre méthodologique
Données : 39 vraies notes physiques GuitarSet de la banque S847, 27 TRAIN, 6 CAL et 6 notes OUTER déjà consultées dans les boucles antérieures. Les références +12 sont produites par transposition standard **hors ligne** ; jamais pendant l'inférence de l'IA. Les notes MIDI vraies servent aux pertes d'apprentissage et aux mesures, jamais au réseau de sortie. La réussite est définie par une erreur YIN absolue ≤0,5 demi-ton face à MIDI source+12. Ce petit score n'est pas une estimation fiable de la réussite musicale en général.

Exactement 256 échantillons PCM nouvellement reçus à 44,1 kHz ; uniquement le passé et le bloc acquis, sans futur ni file d'attente ajoutée. Ne pas confondre cela avec une latence acoustique nulle : les filtres causaux peuvent introduire du déphasage ou un retard de groupe. Les six têtes officielles NOTE et le MIDI ne sont pas modifiés ; **aucune promotion**.

