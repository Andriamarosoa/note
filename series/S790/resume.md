---
serie: S790
statut: en_cours
type: experience
reference: S708
branche: codex/note-s790-independent-framepitch-causal-expert
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S790 — Résumé

## Objectif

Améliorer prioritairement **Frame Pitch F1** en conservant S708/S707 et le flux MIDI LIVE. Former une tête indépendante par hauteur avec sélection causale par fragment (et par pitch), analyser très profondément les erreurs des accompagnements, registres graves et voisinages harmoniques.

## Hypothèse

Frame Pitch est à 76,46 % sur CAL32 et 80,96 % sur HELD déjà exposé. Une tête de sustain par pitch, utilisant les indices spectraux, l'historique causal et les notes réellement actives dans la sortie MIDI S708, pourrait séparer les notes réellement présentes des harmoniques et augmenter le score Frame Pitch. Le score HELD étant déjà exposé, il ne doit pas servir au réglage.

## Modifications

- Tête de classification Frame Pitch indépendante et **sélecteur par fragment**.
- Entraînement FIT80 avec validation par composition laissée à l'écart.
- Détection d'erreurs par composition et par registre, mesure des confusions harmonic-neighbor.
- Six audits : données, acoustique, architecture, apprentissage, métriques, LIVE.
- Rapport systématique de chaque résultat dans `series/S790/resultat.md`, en respectant les règles de l'audit documentaire.

## Protocole

Signé S708 #39094682419 ; S707 #39059008484 ; GuitarSet FIT80 pour entraîner et choisir la politique, CAL32 uniquement pour diagnostic. HELD et TEST protégés. L'expert produit une **prédiction de sustain par hauteur à 1024 échantillons (SHADOW)**, et NON un nouveau flux MIDI final ; il ne peut donc pas être promu sur ce seul score. Aucun post-traitement non causal ni inférence avec labels.

Comparaison des valeurs Frame Pitch sur la même horloge et les mêmes labels pour base et candidat ; écart global et par composition, précision/recall, K et impact hypothétique. Il faudra ensuite intégrer un candidat retenu et valider les six métriques sur le même MIDI 256 sans dégradation avant toute promotion.

## Audits obligatoires

- Données : contrôle PCM/annotations, FIT80/CAL32, noms des enregistrements, distributions des classes/hauteurs, graves et polyphonie.
- Acoustique : spectre causal, notes fondamentales, harmoniques et ambiguïté de fondamentale.
- Architecture : tête autonome par hauteur, sélection par fragment, pas de réemploi trompeur des métriques MIDI.
- Apprentissage : OOF par cinq compositions, choix FIT-only et garde anti-régression.
- Métriques : TP/FP/FN frame pitch, recomptage, comparaison base/candidat, texture/composition.
- LIVE : invariants S708/S707, information disponible au temps de décision, aucune modification en production.

## Statut

**En cours.** Nouveau résultat Frame Pitch : **non mesuré**. Aucun candidat promu.
