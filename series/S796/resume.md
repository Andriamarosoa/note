---
serie: S796
statut: en_cours
type: experience
reference: S708
branche: codex/note-s796-retrigger-forensics-no-extra-latency
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S796 — Garde anti-réarmement synthétique au même instant

## Objectif

Interdire une extinction et une nouvelle attaque synthétique de la même hauteur au cours de la même décision, sans déplacer un NoteOn ou NoteOff natif S708 et sans augmenter la fenêtre d'attente.

## Hypothèse

Avec sustain entre 0,16 et 0,30, la logique S795 pouvait libérer une voix puis l'émettre de nouveau au même timestamp. Cet état peut créer des attaques parasites et de fausses paires. L'ensemble `closed_this_frame` empêche uniquement la réattaque simultanée ; une attaque d'une décision ultérieure demeure possible.

## Modifications

`src/causal_note/s795_activek_experts.py` ajoute un ensemble temporaire de hauteur libérées et rejette leur réarmement dans cette décision. `test/test_s795_activek_experts.py` contient le scénario régressif exact. `scripts/s796_s795_forensic.py` et le workflow S796 contrôlent les six rôles indépendants à partir de l'archive S795 existante.

## Protocole

Pas de nouvel entraînement : réutiliser l'archive FIT80/CAL32 du run S795 39340511467. Les cinq spécialistes étaient entraînés, mais une seule des 25 politiques était admissible : la référence inchangée. Aucune précision CAL32 validée, car le replay source affiche k_correct=78367 au lieu de 78366 signé. CAL ne sera jamais utilisé pour sélectionner. HELD et TEST restent fermés. Aucun modèle LIVE changé.

## Audits obligatoires

Données, acoustique, architecture, apprentissage, métriques, LIVE ; six exécutions séparées sur les mêmes preuves gelées, consensus indépendant et échec strict si une pièce manque. Un audit réussi valide la vérification d'une erreur S795, pas une amélioration S796. Non-régression MIDI256 des six axes et comparaison de latence physique toujours indispensables avant toute promotion.

## Statut

En cours. Score S796 non mesuré, latence matérielle et coût CPU non mesurés ; aucune nouvelle promotion. Ne pas dupliquer le run d'entraînement S795 pour un audit logique.
