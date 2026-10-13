# S405 — Présence harmonique ≠ nouvelle note : K0 est une absence d'attaque

Date : 13 octobre 2026. Nouveau contrat vérifié dans le code. **La présence d'harmoniques n'est pas K0 à elle seule** ; **K0 signifie exactement 0 nouvelle attaque**, que le signal contienne du silence, une décroissance ou les harmoniques d'une fondamentale déjà jouée.

## Principe

| Situation | Frame (notes présentes) | Harmoniques | S80 K | Nouveaux NoteOn |
|---|---|---|---:|---|
| Premier C3 joué, harmoniques C4/G4 | C3 actif | Oui | K1 | C3 |
| C3 continue à résonner sans nouvelle attaque | C3 actif | Oui | **K0** | **aucun** |
| C3 résonne, E3 vient d'être jouée | C3/E3 actifs | Oui | K1 | **E3 seulement** |
| Résonance déjà terminée | aucune | éventuellement bruit | **K0** | **aucun** |
| Même note E3 frappée à nouveau alors qu'elle résonne | E3 actif | Oui | K1 | **retrigger E3** |

### Changements réellement appliqués

1. `src/causal_note/k_first_multihead_notes_s404.py` : la sortie alias `note_logits` correspond désormais aux **logits de nouvelles attaques** `onset_logits`, et non à `frame_logits+onset_logits`. La sortie `frame_logits` et les sorties harmoniques demeurent entièrement accessibles, même pour K0. Le K prédictif S80 reste **une entrée de l'encodeur**.
2. `src/causal_note/k0_sustain_harmonic_onset_decoder_s405.py` : décodeur d'événements MIDI chronologiques qui lit **onset, frame et harmoniques séparément**. Seuls K et les sorties `onset_logits` peuvent créer un `NoteOn`. `frame_logits` suit la durée des notes déjà créées, avec délai de relâchement ; les harmoniques servent de données explicatives, pas de preuves de nouvelles attaques. Une réattaque d'une même hauteur produit `NoteOff+NoteOn`. L'état MIDI est réinitialisé entre morceaux ; sans réinitialisation il pourrait fuir d'un morceau à l'autre.
3. `KOnsetStateDecoder.step_from_s404(...)` permet d'utiliser directement les trois sorties authentiques de S404, sans recombiner la présence en attaques.

### Contrôle reproductible

[Workflow S405](https://github.com/Andriamarosoa/note/actions/workflows/v273-s405-K0-harmonic-sustain-onset.yml), sur la branche `codex/s276-idmt-song-disjoint-candidate-audit`.

Tests synthétiques déterministes :
- K0 et harmonique C4 très forte + note C3 encore active : aucune nouvelle note MIDI ; C3 reste active.
- K1 et C3 déjà actif : seule E3 est déclenchée, même avec harmoniques C4.
- K0, décroissance en plusieurs événements : NoteOff autorisé ; NoteOn interdit.
- K1 sur une hauteur déjà active : réattaque correcte en NoteOff puis NoteOn.
- K2 : exactement deux nouvelles notes déclenchées malgré la résonance.
- K0 au début d'une nouvelle composition : aucune note fantôme héritée d'un morceau précédent.
- Intégration : les sorties de la classe de modèle S404 alimentent directement le décodeur via `step_from_s404`.

**Limites majeures :** S404 n'a toujours pas été entraîné, les seuils de maintien/libération de S405 ne sont pas calibrés, aucune nouvelle évaluation d'Exact-K, d'Exact-accord MIDI ni de fausses notes n'a été réalisée. Ne pas transformer le succès des tests unitaires en amélioration de précision. Les sorties S80 S397 et les métriques 90,25 % global / 81,98 % poly sont **inchangées**, tout comme S402 (36,76 % d'accords polyphoniques complets en validation par composition de sa tête de notes).

Suite de recherche : entraîner séparément le `frame` (présence de fondamentale) et le `onset` (nouvelle attaque) sur annotations GuitarSet, y compris de nombreux K0 avec résonance, préserver masquage des harmoniques qui ne proviennent pas d'une source physiquement séparable ; comparer avant/après sur compositions disjointes avec scores **K0 faux NoteOn**, réattaques et K1–K6 notes manquées.
