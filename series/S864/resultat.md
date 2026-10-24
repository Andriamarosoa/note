---
serie: S864
statut: termine_local_non_promu
type: experience
date: 2026-10-24
promotion: false
---

# S864 — Filtres plus longs et phase minimale

## Expérience
Banc de filtres appris étendu à 1 025 coefficients, initialisation de phase minimale, sélection par CAL. La consommation CPU et le retard de groupe sont audités séparément de l'hop PCM256.

## Résultats réels
CAL 4/6 ; 6 notes déjà examinées 2/6 (33,3 %). Centres d'énergie des coefficients ~3,31–7,20 ms selon filtre après entraînement ; p99 local ≈0,759 ms. Cette mesure de filtre n'est pas une latence audio réelle.

## Décision
**Régresse par rapport à S863 sur le jeu déjà consulté ; pas de promotion. Les poids restent dans l'archive pour audit.**


## Cadre méthodologique
Données : 39 vraies notes physiques GuitarSet de la banque S847, 27 TRAIN, 6 CAL et 6 notes OUTER déjà consultées dans les boucles antérieures. Les références +12 sont produites par transposition standard **hors ligne** ; jamais pendant l'inférence de l'IA. Les notes MIDI vraies servent aux pertes d'apprentissage et aux mesures, jamais au réseau de sortie. La réussite est définie par une erreur YIN absolue ≤0,5 demi-ton face à MIDI source+12. Ce petit score n'est pas une estimation fiable de la réussite musicale en général.

Exactement 256 échantillons PCM nouvellement reçus à 44,1 kHz ; uniquement le passé et le bloc acquis, sans futur ni file d'attente ajoutée. Ne pas confondre cela avec une latence acoustique nulle : les filtres causaux peuvent introduire du déphasage ou un retard de groupe. Les six têtes officielles NOTE et le MIDI ne sont pas modifiés ; **aucune promotion**.

