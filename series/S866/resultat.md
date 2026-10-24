---
serie: S866
statut: termine_local_non_promu
type: experience
date: 2026-10-24
promotion: false
---

# S866 — Conversion de phase et fine-tuning causal

## Expérience
Les noyaux appris S863 sont reprojetés en approximation de phase minimale, puis affinés exclusivement sur TRAIN/CAL. La transformation ne fournit pas de note ni de méthode de transposition conventionnelle au moteur PCM256.

## Résultats réels
CAL 4/6 ; 6 notes déjà examinées 3/6 (50 %). Distance log-mel médiane vs standard sur ces six notes 15,079 dB, contre 15,374 dB pour S863 et 11,707 dB pour S860. Centres d'énergie des filtres entraînés ~1,2–3,9 ms (proxy, pas mesure d'alignement d'attaques).

## Décision
**Même exactitude de hauteur que S863 et légère amélioration spectrale, mais toujours moins fidèle que S860 ; pas conforme à « zéro retard » sur preuve acoustique, donc non promu.**


## Cadre méthodologique
Données : 39 vraies notes physiques GuitarSet de la banque S847, 27 TRAIN, 6 CAL et 6 notes OUTER déjà consultées dans les boucles antérieures. Les références +12 sont produites par transposition standard **hors ligne** ; jamais pendant l'inférence de l'IA. Les notes MIDI vraies servent aux pertes d'apprentissage et aux mesures, jamais au réseau de sortie. La réussite est définie par une erreur YIN absolue ≤0,5 demi-ton face à MIDI source+12. Ce petit score n'est pas une estimation fiable de la réussite musicale en général.

Exactement 256 échantillons PCM nouvellement reçus à 44,1 kHz ; uniquement le passé et le bloc acquis, sans futur ni file d'attente ajoutée. Ne pas confondre cela avec une latence acoustique nulle : les filtres causaux peuvent introduire du déphasage ou un retard de groupe. Les six têtes officielles NOTE et le MIDI ne sont pas modifiés ; **aucune promotion**.

