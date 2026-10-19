---
serie: S793
statut: en_cours
type: infrastructure
reference: S708
branche: codex/note-s793-zero-added-latency-hard-gate
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S793 — Aucun délai additionnel en LIVE

## Objectif

Rejeter toute architecture NOTE qui ajoute un temps d'attente, des frames tamponnées, un lookahead, une confirmation tardive, une rétention de NoteOff ou du temps de calcul réellement supplémentaire. Le but est zéro latence ajoutée à la chaîne S708, non une latence physique absolue nulle.

## Hypothèse

Une amélioration du Frame Pitch SHADOW peut masquer une régression temporelle Offset/Full Note. Le veto de latence doit rester indépendant de l'exactitude et refuser une proposition même si elle dépasse 80 % sur toutes les métriques.

## Modifications

- src/causal_note/s793_zero_added_latency.py : contrat scheduling zéro buffer, lookahead, confirmation, hold et hop supplémentaires ; provenance par événement S708 ; rejet de toute émission native retardée.
- Mesures comparables répétées sur le même ordinateur et périphérique : capture vers NoteOn, capture vers NoteOff, délai total et durée CPU, p50, p95 et maximum ; aucune augmentation admissible.
- scripts/s793_zero_added_latency_gate.py : rejeu S792 figé, rejet 1, 2 et 4 blocs.
- test/test_s793_zero_added_latency.py : treize tests de veto incluant gain de F1 avec latence, mesures absentes, coût CPU et six axes.
- .github/workflows/note-s793-zero-added-latency.yml : preuves CI sans nouvel entraînement ni changement de production LIVE.

## Protocole

Toute promotion future devra réussir simultanément le contrat de scheduling, la preuve que les NoteOn et NoteOff ne sont pas émis plus tard, la protection des six métriques et des tests physiques couplés de latence sur le même matériel. Une absence de données bloque l'autorisation, elle ne vaut jamais pour zéro latence. Pour une note supplémentaire, la première date d'éligibilité doit être établie par une preuve indépendante.

Les prolongations S792 sont interdites quelle que soit leur performance. Il n'existe aucune exception pour les modèles multi-têtes, sélecteurs ou architectures distillées.

## Audits obligatoires

Données : mêmes paramètres et même matériel. Acoustique : aucun futur audio. Architecture : aucune file supplémentaire. Apprentissage : aucune sélection d'attente par CAL/HELD/TEST. Métriques : six axes et délais non régressifs. LIVE : preuves matérielles p50/p95/max, sinon refus.

## Statut

Infrastructure en cours de validation sur GitHub. Aucun nouveau modèle scientifique ni mesure du microphone matériel revendiqués. Le contrat logiciel ne constitue pas une preuve de latence physique du LIVE.
