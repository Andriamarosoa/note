---
serie: S830
statut: en_cours
type: experience
reference: S708
branche: codex/note-s830-private-musical-evidence256
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S830 — Exact-K indépendant, reconstruction musicale S809 à PCM256

## Objectif
Tester, sans aucune interaction entre les caractéristiques, si reconstruire les signaux musicaux **Onset / Sustain / Release** dans une architecture propre à Exact-K améliore le résultat S829 57,5247 % sur FIT80. Chaque axe reste isolé ; aucune fusion ou influence croisée ne sera tentée durant cette expérience.

## Hypothèse
S829 a remplacé les indices prédictifs par des simples résumés spectraux et deux modèles K. Trois têtes acoustiques supervisées supplémentaires capables d'estimer séparément attaque, maintien et relâchement de chaque hauteur peuvent mieux distinguer K2–K6. Ces têtes sont internes à Exact-K : elles ne sont PAS les modèles publics Onset et Offset ni leurs sorties.

## Modifications
Nouveaux modules seulement : `src/causal_note/s830_private_musical_evidence256.py`, `scripts/s830_train_private_musical_evidence256.py`, les tests et le workflow S830. Extraction causal PCM via la copie fonctionnelle S829 de DSP propre à Exact-K, [49,72] features par hauteur. Trois modèles HistGradientBoostingClassifier indépendants estimant Onset/Sustain/Release internes à Exact-K, plus deux anciens experts de cardinalité S829 entraînés de nouveau sur chaque fold. Sélection dynamique par fragment sur les candidats K de ces experts, calibrée uniquement sur inner-CAL. Aucun modèle d'autre caractéristique, poids partagé, note future, NoteOn/Off de référence comme feature ou sortie MIDI.

## Protocole
Même 80 pistes GuitarSet FIT80, cinq compositions, grille 256 et 306 536 trames exactement identiques au test S829. 3 compositions TRAIN, 1 inner-CAL, 1 OUTER (48/16/16 pistes) dans chacun des cinq folds. Ré-entraînement de 5 têtes dans chaque fold, sélection CAL exclusivement, mesure Exact-K sur K0–K6 y compris silence. Comparaison S830 vs S829 **recalculé sur les mêmes frames** et H0 historique. Comptage par vrai K, sous- et surcomptages, composition et temps d'inférence. Le protocole FIT exploré ne prouve pas la généralisation HELD.

## Audits obligatoires
- Données : provenance S800 SHA, 80/80 WAV, labels disponibles seulement TRAIN/CAL ou mesure finale OUTER.
- Acoustique : flux mono PCM44,1 kHz, nouvelle fenêtre à chaque hop256, invariance changement du futur.
- Architecture : code de nouvelles têtes propre à Exact-K ; cinq autres caractéristiques intouchées.
- Apprentissage : cinq véritables têtes indépendantes, 3 TRAIN/1 CAL/1 TEST, aucun pseudo-label d'une autre architecture.
- Métriques : scorings appariés 306 536 frames, K0–K6, score par composition, veto régression.
- LIVE : no extra hold/lookahead, parity offline/online synthétique ; CPU p95/p99 iRig à prouver avant toute promotion.
- Agents NVIDIA et six agents Work : non invoqués ; aucune approbation simulée.

## Statut
En cours. Aucun pourcentage S830 annoncé avant rapport FIT80 vérifié. S829 57,5247 % et S708 H0 61,6972 % historiques demeurent inchangés. PR S830 DRAFT / `promotion:false`.

Demande scientifique explicite après smoke PASS GitHub #39632139950 : [run-fit80]. Ce déclencheur exécute les 80 pistes, les cinq folds et la mesure appariée S830/S829/H0. Les scores restent non mesurés jusqu'au rapport final.
