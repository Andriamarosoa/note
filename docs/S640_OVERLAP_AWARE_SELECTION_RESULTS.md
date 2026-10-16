# S640 — Tête de sélection des notes superposées / Bass ×2

## Objectif : distinguer une nouvelle fondamentale d'une harmonique

L'audit S639 avait identifié 243 faux NoteOn nouveaux lors du Bass ×2
GuitarSet DEV, avec 98 fuites depuis des attaques d'autres notes,
72 réattaques/doublons et 64 relations octave/harmoniques suspectes.

S640 ne transforme plus aveuglément toute nouvelle détection transposée
en NoteOn. Il évalue chaque candidat avec des **preuves simultanées**
de la fenêtre originale, de la fenêtre ×2 et de la tête native K.
Les arguments comprennent :
- probabilité native d'attaque, sustain et release ;
- probabilités bass ×2 d'attaque, sustain et release ;
- posterior K=0..6, utilisé comme **feature**, pas comme veto global ;
- fondamentales spectrales natives aux trois résolutions et leur
  relation à la bande complète ;
- dérivées de l'énergie native et flux spectraux (nouvelle attaque) ;
- attaque et sustain d'autres hauteurs, notamment pitch+12,
  afin de détecter compétition harmonique et fuites inter-hauteurs.

Un classifieur linéaire régularisé apprend à **s'abstenir**.
L'apprentissage utilise uniquement 18 enregistrements GuitarSet TRAIN
appartenant à neuf groupes de composition; la validation est menée sur
les 12 enregistrements DEV connus (BN3, Funk3, Rock1).
Les annotations ne sont jamais fournies au système à l'inférence.
TEST jamais chargé. S614 et le LIVE officiel sont inchangés.

## Protocole et preuve

- GitHub CI réussi :
  https://github.com/Andriamarosoa/note/actions/runs/38940351211
- Artefact `S640_TRAIN_ONLY_OVERLAP_RESCUE_GUITARSET_DEV/report.json`.
- Code :
  `scripts/s640_learned_overlap_rescue.py`,
  `test/test_s640_learned_overlap_rescue.py`.
- 388 événements d'entraînement; seulement 19 nouveaux vrais NoteOn
  récupérables par Bass ×2 (369 exemples négatifs).
- `StandardScaler + LogisticRegression(C=0.35, class_weight=balanced)`.
- Seuils exploratoires fixes : 0,35/0,50/0,70/0,85,
  primaire 0,70 défini avant la validation.
- Même sample clock, pas de future lookahead, NoteOn ±80ms,
  exact checkpoint et K natifs inchangés.

## Comparaison réelle GuitarSet DEV

| Méthode | TP | FP | FN | F1 | TP notes superposées /1147 | TP K3+ /910 |
|---|---:|---:|---:|---:|---:|---:|
| S614 référence | 998 | 585 | 582 | **63,1046%** | 697 | 550 |
| Bass ×2 naïf | 1044 | 834 | 536 | 60,3817% | 740 | 586 |
| Tête apprise seuil 0,35 | 1026 | 694 | 554 | 62,1818% | 723 | 571 |
| Tête apprise seuil 0,50 | 1020 | 664 | 560 | 62,5000% | 718 | 567 |
| Tête apprise seuil 0,70 (primaire) | 1014 | 645 | 566 | 62,6119% | 712 | 562 |
| Tête apprise seuil 0,85 (exploratoire) | 1011 | 616 | 569 | **63,0496%** | 709 | 559 |

À seuil 0,85 (déclaré exploratoire) :
- +13 TP, +31 FP relativement à S614 ;
- le rescue naïf était +46 TP, +249 FP ;
- 48 propositions approuvées par la tête ;
- +12 NoteOn corrects en contexte de notes superposées,
  dont +9 sur accords K3+ ;
- pas de régression des TP déjà corrects dans cet essai
  (13 récupérations, 0 perdue).
- F1 encore **−0,055 point**, donc **PAS PROMU**.

## Pourquoi pas encore devant S614 ?

Le train strict n'a produit que 19 exemples vraiment positifs pour
388 événements candidats; class-weighting et régularisation évitent
une sur-adaptation grossière, mais la généralisation est fragile.
L'ajout de features de compétition aide, mais le seuil 0,85 conserve
un ratio +31 FP / +13 TP, légèrement trop haut (2,38 FP par TP).
Pour dépasser S614, avec +13 TP, il faudrait <=28 FP additionnels.

## Étapes possibles, non encore validées

- Étendre les enregistrements TRAIN GuitarSet (sans utiliser DEV
  ni TEST comme labels d'entraînement) pour augmenter la proportion
  de cas de vrais accords récupérables ;
- ajouter un garde-fou de nouveauté physique et de compétition harmonique,
  ou sélectionner par contexte K prédit sans faire regresser S614 ;
- évaluer sur compositions et instruments réellement indépendants
  (notamment URMP pour les notes aiguës) ;
- profiler la charge de trois fenêtres sur le microphone. S640 est
  OFFLINE et ne prétend pas produire un flux MIDI strict LIVE.

Le score F1 à 63,05% est un réel progrès par rapport au Bass Rescue naïf,
mais n'est PAS une amélioration officielle ni un score nouveau validé.
