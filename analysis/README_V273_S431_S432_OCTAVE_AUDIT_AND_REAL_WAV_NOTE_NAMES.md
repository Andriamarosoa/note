# S431–S432 — Son → E3 / [E3, G3], correcteur d'octave et vrai WAV

**14 octobre 2026.** Objectif utilisateur : identifier les **noms de notes musicales**, pas générer de fichier MIDI. **Live et durées (Offsets) mis de côté.** Toutes les décisions sur le nombre de nouvelles notes `K0..K6` demeurent sous la responsabilité du S80 original.

## Référence de développement toujours retenue

**S428 : 3 025 / 7 385 = 40,9614 % d'accords K2–K6 entièrement reconnus**, +250 accords nets contre S419 ; rappel des hauteurs poly 15 855/20 894 = 75,8830 %. Scores GuitarSet de **développement** avec exposition historique en amont : pas encore de validation indépendante complète.

S80/S397 Exact-K **53 529 / 59 309 global (90,25 %), 6 054 / 7 385 poly (81,98 %)** sur le même benchmark, préservé exactement ; K0 = aucune nouvelle attaque et **pas** absence de son.

## S431 — Test d'un modèle de substitution d'octave (non retenu)

Code [S431](https://github.com/Andriamarosoa/note/blob/codex/s276-idmt-song-disjoint-candidate-audit/scripts/s431_octave_K23_pitch_identity_guard.py) ; [run #38518847905](https://github.com/Andriamarosoa/note/actions/runs/38518847905), **success**, artefact `11872703323`.

- Construire les propositions **octave -12/+12** pour une des notes prédites en K2 ou K3, toujours en remplaçant une hauteur par une autre, jamais en ajoutant/supprimant une note.
- Caractéristiques : morphologie harmonique source (S428, S416) pour la hauteur remplacée, la hauteur proposée et la différence physique d'énergie.
- Modèle binaire des cas où le changement rend l'accord entièrement correct contre les cas où il le rend incorrect.
- Apprentissage et sélection du seuil **sur d'autres compositions seulement**, avec abstention si gain interne insuffisant.

**Résultats mesurés :**

| Vrai K | S428 exacts | S431 exacts | Net |
|---:|---:|---:|---:|
| K0 | 38 432 | 38 432 | 0 |
| K1 | 8 232 | 8 232 | 0 |
| K2 | 1 495 | 1 493 | **−2** |
| K3 | 1 084 | 1 086 | **+2** |
| K4 | 339 | 339 | 0 |
| K5 | 93 | 93 | 0 |
| K6 | 14 | 14 | 0 |
| **Poly K2+** | **3 025** | **3 025** | **0** |

**10 corrections / 10 régressions, zéro gain net.** Trois folds n'activent pas le correcteur, un fold apporte des gains et pertes compensés. Pas de promotion. Le décodeur **S428 reste la meilleure référence expérimentale de hauteurs**.

Les sorties de S431 ont été exportées comme **vrais noms musicaux** par ligne puis relues ; les 59 309 comptages K sont inchangés.

## S432 — Chaîne WAV réelle jusqu'aux noms musicaux (K S80 externe requis)

Code [S432](https://github.com/Andriamarosoa/note/blob/codex/s276-idmt-song-disjoint-candidate-audit/scripts/s432_WAV_original_harmonic_input_to_note_names_with_S80_K.py) ; [run corrigé #38519266231](https://github.com/Andriamarosoa/note/actions/runs/38519266231), **success**, artefact `11873141340`.

Il réutilise **exactement** l'extracteur acoustique historique `scripts/extract_v273_harmonic_trajectory.py` : FFT 4096, saut de 256, 6 partiels harmoniques, reconstruction non négative, `sequence42x49` à partir des échantillons WAV réels. Aucun nouveau faux extracteur de propriétés ou oracle de vérité n'est substitué aux poids entraînés.

**Interface :**
```bash
PYTHONPATH=.:src python -m scripts.s432_WAV_original_harmonic_input_to_note_names_with_S80_K \
  --wav ma_guitare.wav \
  --decisions evenements_predict_S80.jsonl \
  --s402-models model/S402 \
  --s416-models model/S416 \
  --s428-models model/S428 \
  --s419-report model/S419/report.json \
  --s428-report model/S428/report.json \
  --out model/noms_de_notes
```

**Format des décisions prédites par le système S80 en amont**, une entrée par événement, ordonnées, avec `row_id` unique :

```jsonl
{"row_id":"son-1","start_sample":19845,"end_sample":20101,"s80_K":1}
{"row_id":"son-2","start_sample":66150,"end_sample":66406,"s80_K":2}
{"row_id":"son-3","start_sample":97020,"end_sample":97276,"s80_K":0}
```

Les événements sont des **nouvelles attaques**, pas les fréquences actuellement audibles. Une note E3 qui continue pendant l'attaque de G3 ne compte pas à nouveau dans K ; pour K2, il faut **réattaquer E3 en même temps que G3**.

Le résultat consolidé est `recognized_note_names_from_WAV.jsonl`, par exemple (sortie réelle du test S432) :

```jsonl
{"source_row_id":"new-WAV-onset-1","start_sample":19845,"K":1,"notes":["E3"]}
{"source_row_id":"new-WAV-onset-2","start_sample":66150,"K":2,"notes":["E3","G3"]}
{"source_row_id":"new-WAV-K0","start_sample":97020,"K":0,"notes":[]}
```

**Vraie expérience WAV synthétique** : le premier événement contient une nouvelle attaque E3, le second **une réattaque E3 plus G3**, et le troisième aucune nouvelle attaque. Le vrai extracteur original et les modèles préentraînés S402/S416/S428 donnent **les trois sorties attendues** sous ces décisions K fournies. La première tentative avait K2 au second événement sans réattaque E3, ce qui violait le protocole d'attaques ; l'audio du test a été **corrigé**, le workflow relancé et validé. Ne pas utiliser la première tentative comme benchmark musical.

**Attention, S432 ne calcule toujours PAS Exact-K depuis un WAV inconnu.** Il reçoit les instants d'événements et leur valeur K du système S80 extérieur ; la reproduction complète des **16 producteurs S45 / 193 politiques S46 / S79 / S80** sur WAV arbitraire doit être connectée avant de dire « WAV seul → E3 ».

L'entrée WAV requiert **mono, 44 100 Hz, PCM16** ; un format différent est refusé explicitement pour protéger la compatibilité de l'extracteur. S430 utilise un ensemble de quatre checkpoints pour les nouveaux sons ; **son taux de reconnaissance sur musique inconnue n'est pas encore mesuré**. L'expérience de quelques sinusoïdes polyphoniques ne démontre **aucune précision générale** sur guitares acoustiques, harmoniques compliquées, bruits, sons réels.

## État des références et suite

- **S80 reste original et verrouillé** : aucune variation K introduite par S431 ou S432.
- **S428 reste meilleur challenger note/noms** à **40,96 % accords poly exacts** sur GuitarSet développement. S431 refusé (zéro gain).
- **S425 reste meilleur challenger NoteOff complet** mesuré précédemment ; cette étude n'a pas recomposé ni rescored la chaîne des Offsets.
- **S402 hauteur officielle de développement, S408 Offset officiel choisi**, inchangés. **Live préservé**, travail temps réel différé.
- **Prochaine étape bloquante** : connecter la véritable inférence S80 Original S79/S45/S46 à un WAV jamais vu pour obtenir automatiquement ses événements et ses K sans oracle/annotation. Puis **valider les notes sur des morceaux indépendants** et entraîner de meilleurs mécanismes conjoints de sélection de fondamentales/harmoniques (S431 n'est pas une solution gagnante).
