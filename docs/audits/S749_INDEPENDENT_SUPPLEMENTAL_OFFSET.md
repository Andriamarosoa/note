# S749 — Heads Offset supplémentaires indépendants, 12 boucles complètes

Date : 2026-10-19. Statut : expérimental, **non promu**.

## Décision fondée sur S748 (run 39205146884)

12 boucles complètes FIT80 / GuitarSet CAL32 : workflow vert, référence
S708 strictement reproduite, **0 candidat admissible**.

Source S708 signée (même protocole CAL32) :

| Mesure | Référence |
| --- | ---: |
| Onset F1 | 74.2638623 % |
| Exact-K | 62.0455409 % |
| Active Exact-K | 56.2762487 % |
| Offset accuracy | 77.3291925 % |
| Full Note F1 | 57.1319312 % |
| Frame Pitch F1 | 76.4626354 % |

La meilleure Active-K S748 (depth3_risk55_tail20) :
+0.2523867 point Active-K, +0.1971434 point Exact-K,
+0.0553028 point Onset, +0.1538666 point Frame Pitch,
+0.0072900 point Full Note, **−0.0486418 point Offset**.
13 événements supplémentaires acceptés ; aucun ancien Offset
correct perdu, mais dénominateur d'Offset modifié par des attaques
acceptées dont la fin n'est pas suffisamment précise.

S748 `depth5_risk55_tail20` : 19 événements ajoutés,
+0.1257613 point Onset et +0.2374231 point Active-K,
**−0.0517776 point Offset**, et 1 Offset source correct perdu.
Les branches acceptant 0 nouvel événement reproduisent S708 ;
elles ne prouvent pas une amélioration.

## Hypothèse causale à vérifier

Le classificateur d'acceptation de nouvelles notes estime, au moment
NoteOn, la probabilité de créer une instance complète correcte.
Cependant l'Offset S707 provient d'un modèle entraîné sur les notes
initiales ; le transfert à une attaque supplémentaire peut avoir
une distribution différente. **Hypothèse, pas cause démontrée.**

Même sans perdre un ancien Offset, l'Offset accuracy peut diminuer
car le nombre de notes correctement appariées avec fin observée
augmente sans gain de fins correctement estimées en proportion.

## Boucles S749 exécutables

Les cinq têtes H1–H5 et leur routeur acoustique dynamique de S748
proposent des attaques. Les trois spécialistes de relâchement
indépendants reçoivent le **même** plan de hazard causal S707 :

- sensible : `gate=.45, persistence=1`
- standard : `gate=.80, persistence=1` (contrôle S748)
- persistant : `gate=.60, persistence=2`

Deux profondeurs HGB FIT80 × deux risques d'admission NoteOn
(`.40` / `.55`) × trois stratégies Offset = **12 flux MIDI réels**
mesurés sur CAL32 avec annotations ouvertes seulement après
prédiction. L'apprentissage d'admission est identique entre
variantes, basé sur features disponibles au NoteOn et les seules
annotations FIT. Chaque spécialiste Offset est indépendant du flux
de NoteOff signé S708.

## Verrous absolus

- Conserver bit à bit la source S708/S707, son identité SHA et ses
  events / métadonnées ; refaire les six mesures sur le même flux.
- Ne pas retenir un essai si baisse d'Onset F1, Exact-K,
  Active Exact-K, Offset accuracy, Full Note F1 ou Frame Pitch F1.
- Refuser si une ancienne instance Offset correcte disparaît,
  si un véritable Onset long disparaît ou si le nombre de TP Onset
  source diminue, même quand l'agrégat semble stable.
- TEST fermé ; aucun réglage sur HELD ni déploiement microphone LIVE.
- S80 officiel inchangé ; aucun réseau distillé/étudiant avant
  validation >80 % de toutes les caractéristiques.
- CI verte ne signifie pas promotion.

## Reproduction

Workflow GitHub :
https://github.com/Andriamarosoa/note/actions/runs/39206876051

PR expérimentale :
https://github.com/Andriamarosoa/note/pull/75

Les conclusions d'efficacité seront ajoutées uniquement après
l'écriture de `report.json` et la réussite des contrôles.
