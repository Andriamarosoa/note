---
serie: S865
statut: termine_local_non_promu
type: experience
date: 2026-10-24
promotion: false
---

# S865 — Affinage explicite de cohérence harmonique

## Expérience
Fine-tuning des poids S863 avec pénalité de contraste des harmoniques GT (uniquement pendant l'entraînement), sans connaissance de la note à l'inférence. Sélection du checkpoint sur CAL.

## Résultats réels
CAL 4/6 ; 6 notes déjà examinées 3/6. Log-mel médian CAL 14,622 dB pour la variante entraînée ; ne pas comparer ce chiffre à une mesure OUTER. Aucune amélioration de réussite YIN par rapport à S863.

## Décision
**Non retenu comme gain validé ; aucune promotion.**


## Cadre méthodologique
Données : 39 vraies notes physiques GuitarSet de la banque S847, 27 TRAIN, 6 CAL et 6 notes OUTER déjà consultées dans les boucles antérieures. Les références +12 sont produites par transposition standard **hors ligne** ; jamais pendant l'inférence de l'IA. Les notes MIDI vraies servent aux pertes d'apprentissage et aux mesures, jamais au réseau de sortie. La réussite est définie par une erreur YIN absolue ≤0,5 demi-ton face à MIDI source+12. Ce petit score n'est pas une estimation fiable de la réussite musicale en général.

Exactement 256 échantillons PCM nouvellement reçus à 44,1 kHz ; uniquement le passé et le bloc acquis, sans futur ni file d'attente ajoutée. Ne pas confondre cela avec une latence acoustique nulle : les filtres causaux peuvent introduire du déphasage ou un retard de groupe. Les six têtes officielles NOTE et le MIDI ne sont pas modifiés ; **aucune promotion**.

