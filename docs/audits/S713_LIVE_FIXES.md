# S713 — Correctifs de code S708/S712 sur branche d'audit

**Branche** `codex/note-s713-independent-code-audit-agents` · **PR draft** [#62](https://github.com/Andriamarosoa/note/pull/62) · **aucune promotion officielle**.

## Ce qui est corrigé et reproduit

1. **AUD-001, couverture d'événements** : fixture d'entrée PCM causal via un modèle injecté, véritable `GuidedSelector` S708 non simulé, NoteOn MIDI 52 à 1024 échantillons puis NoteOff à 3072 échantillons. Invariant par tailles de callbacks 128/256/333/2048/5120 échantillons, fermeture normale et shutdown; horodatages jamais antidatés. Le smoke WAV avec le checkpoint réel a produit deux événements, et la CI vérifie maintenant explicitement au moins un NoteOn et un NoteOff apparié (contrairement au test zéro événement de S712).
2. **AUD-002, CPU** : `IncrementalS708TemporalHead` utilise une mémoire bornée par couche convolutive (tap actuel, t-d, t-2d) plutôt que de recalculer les 30 décisions passées. `latest_s708_acoustic_features` calcule uniquement la ligne 192D courante à partir des 9 derniers instants et des taps 8/32/64, au lieu de reconstruire les 65 lignes. Les poids, le modèle K, les scores et les seuils S708 sont INCHANGÉS. Il existe `--s708-temporal-engine reference` pour comparaison indépendante.
3. **AUD-004, opération** : `--status --s708-model ...` indique maintenant SHA256 du vrai modèle, vérification contre `report.json`, politique figée, mode incrémental ou référence, pas natif 256 vs pas décision 1024 et absence S707 LIVE. Le status natif sans `--s708-model` n'annonce aucune promotion.
4. **AUD-005 partiel, définition des métriques** : tests manuels NoteOn-pitch±80ms, Offset conditionnel, EOF non crédité, note fausse au bon instant, réattaque à la même hauteur, NoteOn doublon et horodatage incohérent. Pas encore de reproduction de tous les 6 scores sur un nouveau corpus de guitare.

## Mesures CI vérifiées

- [Premier benchmark S713](https://github.com/Andriamarosoa/note/actions/runs/39106655134) : deux moteurs sur **exactement le même PCM synthétique 2 secondes** et le véritable `s708_selected.pt`, événements identiques. p95 callback : référence 14,7635 ms, incrémental 8,3766 ms ; ratio temps CPU/audio : 0,7485 → 0,5113.
- [Optimisation acoustique supplémentaire](https://github.com/Andriamarosoa/note/actions/runs/39106844639) : **19 tests réussis**, équivalence 192D complète sur 82 frames (y compris une FIFO 65), même sortie événementielle et probabilités à tolérance flottante 3e-6. p95 callback : référence 15,4651 ms, optimisé **5,1891 ms** ; ratio CPU/audio : 0,7526 → **0,3841**.
- 256 échantillons/44,1 kHz couvrent **5,805 ms** d'audio. Un p95 inférieur à 5,805 ms sur runner GitHub n'est **pas** une garantie de cadence réelle sur iRig/Windows. La file peut encore déborder sous des pics p99, driver ou SoundFont.
- Le test PCM synthétique vérifie intégration/sémantique et effort CPU, **pas** le F1 sur vraie guitare. Les six métriques de transcription restent **non mesurées** sur microphone réel.

## Ce qui reste ouvert

- **AUD-003** : S707 causal hazard NoteOff hors ligne n'est pas intégré à S712; le live utilise toujours le sélecteur joint. Les résultats de 73,73 % Offset et 50,31 % F1 notes complètes de S708 DEV NE SONT PAS les scores S713 LIVE. Ne pas raccorder naïvement S675 sans comparer chaque libération et la capacité active.
- **AUD-002 sur Windows** : vérifier worker p95/p99, files, interruptions, reconfiguration périphériques Realtek/iRig, sortie MIDI et piano FluidSynth.
- **AUD-005/006** : contre-audit des splits et du matching, notes longues et accords, harmoniques basses E2/B2, K2–K6, faux positifs, résultats HELD et DEV simultanés.
- **AUD-007** : inventaire/déduplication déclencheurs workflows historiques avant toute suppression ; une archive de 848 YAML n'est pas à elle seule un bug de parallélisme.

## Commandes PowerShell de reprise

À la racine de `note`, après récupération des dernières modifications :

```powershell
git fetch origin codex/note-s713-independent-code-audit-agents
git switch codex/note-s713-independent-code-audit-agents
$env:PYTHONPATH = ".;src"
python -m scripts.live_strict_notes --status --s708-model ".\s708_model\s708_selected.pt"
python -m scripts.live_strict_notes --mic --device 15 --hop-size 256 --block-size 256 --s708-model ".\s708_model\s708_selected.pt" --s708-temporal-engine incremental --soundfont ".\SalamanderGrandPiano-V3+20200602.sf2" --out ".\s713_micro.jsonl"
```

Le numéro `15` doit être confirmé par `--list-devices`. Ne pas combiner avec `--s675-hazard-model`, lookahead ou les branches dual-hop. Pour A/B isolé, remplacer `incremental` par `reference` avec un AUTRE fichier `--out`.

## Critères avant promotion

Un vrai WAV GuitarSet/IDMT de référence et une prise micro annotée doivent passer les six métriques et les guards, en comparant deux variantes sur mêmes samples et seuils. Aucune promotion automatique, aucune mise à jour des checkpoints natifs.
