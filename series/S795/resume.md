---
serie: S795
statut: en_cours
type: experience
reference: S708
branche: codex/note-s795-activek-multihead-immediate-midi
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S795 — Cinq experts candidats Active Exact-K, vrais événements MIDI sans attente

## Objectif

Récupérer les voix K2–K6 manquantes identifiées par S794, tout en conservant les six métriques du MIDI réel et la règle S793 de zéro latence supplémentaire.

## Hypothèse

Une nouvelle attaque acoustiquement justifiée peut être ajoutée au moment de la décision causale existante sans déplacer le moindre événement S708. Cinq modèles indépendants de familles d'erreurs différentes peuvent être comparés, mais le nombre final retenu résulte exclusivement de FIT. Risque : faux NoteOn, NoteOff non précis, surcharge CPU.

## Modifications

- src/causal_note/s795_activek_experts.py : cinq spécialistes candidats (général, graves, polyphonie, transitoires, harmoniques), 23 entrées causales, 13 masques et deux seuils. Une nouvelle voix n'utilise jamais de signal futur.
- scripts/s795_train_activek_midi256.py : entraînement FIT80, cinq folds disjoints par composition, source S708 et S707 signées, sélection avec six métriques issues des véritables événements MIDI. CAL32 uniquement après gel de la politique.
- test/test_s795_activek_experts.py : causality, NoteOn immédiat, NoteOff à première preuve, retrigger, maximum six voix, protection des événements et labels FIT.
- Contrat S793 : aucune attente imposée ni événement source décalé ; les nouvelles notes sont horodatées immédiatement. La latence physique réelle et le coût CPU supplémentaire ne sont pas encore établis.

## Protocole

FIT80 GuitarSet et CAL32 par composition, HELD/TEST non ouverts. Source S708 par fonctions originales, vérification stricte de la signature et des métriques. Chaque tête est séparément entraînée sur son domaine ; les sous-ensembles sont sélectionnés sur FIT uniquement, aucune régression des cinq autres métriques. Si aucun candidat admissible : S708 identité. Pas de promotion automatique.

## Audits obligatoires

- Données : empreinte et séparations FIT/CAL.
- Acoustique : rôles complémentaires et absence de futur.
- Architecture : source MIDI intouchée et maximum six voix.
- Apprentissage : entraînement et sélection hors-composition FIT seulement.
- Métriques : six axes MIDI256, parité des compteurs.
- LIVE : S793, zéro attente de décision ; mesure physique exigée avant promotion.

## Statut

En cours. Aucune tête entraînée ni score nouveau validé jusqu'au run. Production LIVE inchangée.
