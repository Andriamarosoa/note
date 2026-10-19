---
serie: S795
statut: en_cours
type: experience
reference: S708
branche: codex/note-s795-activek-multihead-immediate-midi
run_id: 39339203180
commit_sha: eb2f0ecec7e3bba6ec598705a82a6108abdd97c0
promotion: false
---

# S795 — Active Exact-K specialists

## Objectif

Améliorer la détection des voix polyphoniques manquantes sans délai supplémentaire.

## Hypothèse

Les accords K2 à K6 exigent potentiellement plusieurs têtes spécialisées plutôt qu'une seule correction globale.

## Modifications

Branche expérimentale dérivée de S794. Aucun modèle ou flux MIDI modifié à ce stade. Documentation initiale ajoutée après un échec de validation documentaire.

## Protocole

Entraînement et sélection sur GuitarSet FIT80, validation par composition. CAL32 reste un diagnostic exposé. HELD et TEST restent fermés. S708 et S707 sont immuables. Les six scores doivent être calculés depuis un même MIDI256 causal.

## Audits obligatoires

- Données : splits et empreintes S708.
- Acoustique : graves et collisions harmoniques.
- Architecture : têtes et sélection par fragment.
- Apprentissage : FIT seulement, validation hors composition.
- Métriques : MIDI256 officiel distinct de shadow 1024.
- LIVE : zéro attente supplémentaire et protection S707/S708.

## Statut

En cours. Aucun entraînement S795 ni score nouveau mesuré. Six audits indépendants requis pour chaque nouveau SHA avant promotion.