---
serie: S867
statut: termine_local_non_promu
type: audit
date: 2026-10-24
promotion: false
---

# S867 — Audit comparatif homogène S860/S863/S866

## Expérience
Replay sur exactement les mêmes six notes CAL et six OUTER réutilisées, même pitch tracker YIN et même distance log-mel, ainsi que même phrase guitare originale 3,5 s vs standard offline +12. Démo audio normalisée pour écouter, sans toucher aux chiffres.

## Résultats réels
S860 CAL 0/6, OUTER 0/6, log-mel OUTER 11,707 dB. S863 CAL 4/6, OUTER 3/6, log-mel OUTER 15,374 dB. S866 CAL 4/6, OUTER 3/6, log-mel OUTER 15,079 dB. Sur phrase externe log-mel S860 10,655 dB, S863 15,270 dB, S866 15,146 dB.

## Décision
**Compromis objectivé : réussite de hauteur sur certaines notes vs fidélité sonore inférieure. Aucune promotion ; vrai test sur guitaristes/chansons/accords inconnus nécessaire.**


## Cadre méthodologique
Données : 39 vraies notes physiques GuitarSet de la banque S847, 27 TRAIN, 6 CAL et 6 notes OUTER déjà consultées dans les boucles antérieures. Les références +12 sont produites par transposition standard **hors ligne** ; jamais pendant l'inférence de l'IA. Les notes MIDI vraies servent aux pertes d'apprentissage et aux mesures, jamais au réseau de sortie. La réussite est définie par une erreur YIN absolue ≤0,5 demi-ton face à MIDI source+12. Ce petit score n'est pas une estimation fiable de la réussite musicale en général.

Exactement 256 échantillons PCM nouvellement reçus à 44,1 kHz ; uniquement le passé et le bloc acquis, sans futur ni file d'attente ajoutée. Ne pas confondre cela avec une latence acoustique nulle : les filtres causaux peuvent introduire du déphasage ou un retard de groupe. Les six têtes officielles NOTE et le MIDI ne sont pas modifiés ; **aucune promotion**.

