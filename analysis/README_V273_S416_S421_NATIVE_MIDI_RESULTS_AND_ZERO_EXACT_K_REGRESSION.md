# S416–S421 — Architecture native de notes MIDI, gain mesuré et **zéro régression Exact-K**

**13 octobre 2026.** Résultat : **S420 = 47,17298 % de F1 note MIDI complète** (hauteur+Onset+Offset ≤50ms), entraîné **sans YourMT3+** et utilisant seulement nos modèles à l'inférence, contre **S412 = 41,83314 %** et **S417 = 46,81349 %**. **Zéro régression Exact-K** sur **59 309 / 59 309 groupes d'événements**, prouvé aussi sur les notes MIDI véritablement exportées. Ni le comptage S80 ni la référence Offset S408 choisie par l'utilisateur n'ont été modifiés.

Tous les chiffres sont des **résultats de développement sur GuitarSet**, pas un benchmark externe indépendant : S80 a déjà été entraîné/sélectionné avec une partie de ces compositions.

## 1. Verrou dur S80 — aucun compromis

Référence Exact-K de développement : `S80_S397_K5_GUARDED`, 53 529 / 59 309 = **90,25443 % global**, 6 054 / 7 385 = **81,97698 % poly K2..K6**.

| Vrai K | Nombre événements | S80 exacts | S420 exacts | Régressions |
|---:|---:|---:|---:|---:|
| K0 | 39 652 | 38 432 | 38 432 | **0** |
| K1 | 12 272 | 9 043 | 9 043 | **0** |
| K2 | 3 445 | 2 999 | 2 999 | **0** |
| K3 | 2 289 | 1 897 | 1 897 | **0** |
| K4 | 1 207 | 806 | 806 | **0** |
| K5 | 355 | 301 | 301 | **0** |
| K6 | 89 | 51 | 51 | **0** |

À chaque `source_row_id`, le `K` original est passé en entrée du réseau de notes et chaque sortie de notes compte **exactement K**. K0 = **aucun nouveau NoteOn**, MAIS la résonance, les harmoniques et les NoteOff précédents peuvent continuer.

Deux contrôles :
1. Contrat intermédiaire : `(midi_pitch_slots>=0).sum(1)==K` sur toutes les 59 309 lignes originales.
2. Audit indépendant [S421](https://github.com/Andriamarosoa/note/actions/runs/38384577930) : **parcours des réels fichiers JSONL MIDI exportés de S417 et S420**, recompte des `source_row_id` et comparaison à la source S80, incluant K0. Résultat : **0 écart**, total 33 475 NoteOn produits.

La performance Exact-K n'augmente PAS de cette opération ; elle est **exactement protégée**.

## 2. S416 — notre propre réseau MIDI, apprentissage sans YourMT3+

[Code S416](https://github.com/Andriamarosoa/note/blob/codex/s276-idmt-song-disjoint-candidate-audit/scripts/s416_native_Kfirst_temporal_MIDI_note_decoder.py) ; [run #38381449092](https://github.com/Andriamarosoa/note/actions/runs/38381449092), **success**, artifact `11803643782`.

- Source native S258 : `sequence42x49` (42 trames x MIDI 40..88) + **K prédit S80**.
- Propre PyTorch CNN 2D temps/fréquence -> information temporelle de croissance/décroissance -> embeddings par hauteur et K (modulation FiLM **avant** la prédiction des notes) -> attention entre 49 pitches -> 49 logits MIDI ; sortie `top-K` contrainte.
- Entraînement sur 4 folds distincts par **composition**, 14 epochs par fold, JAMS utilisés uniquement pour les autres compositions, **aucun poids, score ou note YourMT3+** même en phase d'apprentissage.
- Résultat note exacte sur les 7 385 vrais groupes poly : **S402 2 715 (36,76%)**, S416 **2 608 (35,31%)**, soit **−107 accords exacts**. S416 seul est **rejeté comme nouvelle référence des hauteurs** : gains K1/K4/K5 mais régressions identitaires K2/K3.

## 3. S417 — vraie durée de note individuelle sur l'audio temporel complet

[Code S417](https://github.com/Andriamarosoa/note/blob/codex/s276-idmt-song-disjoint-candidate-audit/scripts/s417_native_S416_MIDI_pitch_plus_continuous_WAV_note_off.py) ; [run #38382271141](https://github.com/Andriamarosoa/note/actions/runs/38382271141), **success**, artifact `11804187427`.

- S416 prédit le MIDI à partir du K de S80, S407 prévoit son instant d'attaque, puis S417 extrait sur le vrai MIC GuitarSet une **trajectoire continue STFT 10 ms**, même pendant K0, et entraîne 4 modèles HistGradientBoosting de log-durée/NoteOff sur les **autres compositions**.
- Il apprend la durée pour chaque note identifiée, ne compte jamais K et ne ferme pas systématiquement les notes à la fin de la fenêtre d'attaque.
- Deux objectifs pré-déclarés comparés : `absolute_error` (retenu expérimentalement) et `squared_error` (moins performant). Pas d'optimisation de seuil sur les compositions testées.
- L'audio MIC continu diffère des caractéristiques pickup mix initiales de S80, et utilise jusqu'à 3,5 s de futur : **mode hors ligne**.

| Même population de 190 pistes, 34 954 vraies notes | S408 | S412 | **S417 natif** |
|---|---:|---:|---:|
| NoteOn+pitch corrects à ±50ms | 25 120 | 25 120 | **25 125** |
| NoteOff correct à ±50ms parmi notes appariées | 13 050 | 14 313 | **16 017** |
| Précision Offset conditionnelle | 51,95 % | 56,98 % | **63,75 %** |
| **F1 note complète** | 38,14 % | 41,83 % | **46,81 %** |
| MAE NoteOff sur notes appariées | 105,43 ms | 95,87 ms | **89,69 ms** |

Le modèle S417 génère réellement les séquences `(source_row_id, MIDI, onset_samples, offset_samples)` pour toutes les 33 475 nouvelles notes sous K protégé.

## 4. S418 — interface de prédiction sur de nouvelles entrées S80

[Code S418](https://github.com/Andriamarosoa/note/blob/codex/s276-idmt-song-disjoint-candidate-audit/scripts/s418_native_Kfirst_MIDI_head_any_new_S80_audio.py) ; [run #38383180884](https://github.com/Andriamarosoa/note/actions/runs/38383180884), **selftest réussi**.

- Charge 4 checkpoints natifs S416 et produit MIDI K0..K6 à partir de `row_id/sequence42x49` et `row_id/s80_K`.
- Vérifie la provenance de chaque checkpoint puis revalide le nombre K **en relisant le CSV exporté**.
- **Ne fournit pas un WAV->S80 complet**, ni directement le NoteOff (cette fonction est assurée par S417). L'ensemble des 4 réseaux **ne doit pas** être noté sur GuitarSet : certains réseaux ont vu les compositions du test lors de l'entraînement ; les chiffres S416 proviennent des prédictions *song-held*.

## 5. S419 — sauver les gains de S416 sans ses régressions K2/K3

[Code S419](https://github.com/Andriamarosoa/note/blob/codex/s276-idmt-song-disjoint-candidate-audit/scripts/s419_S402_S416_songheld_Kconditional_native_head_gate.py) ; [run #38383639848](https://github.com/Andriamarosoa/note/actions/runs/38383639848), **success**.

- Pour chaque fold, sélection d'un modèle de hauteur selon **le K prédit**, avec bilan de gain de S416 vs S402 sur **les autres compositions seulement**. Jamais d'étiquette `trueK` à l'inférence.
- S402 est conservé pour les accords K2/K3 risqués, S416 utilisé quand les preuves TRAIN montrent suffisamment d'avantages (K1, K4 et certains K5).
- S419 prédit **2 775/7 385 accords poly exacts = 37,58 %**, contre 2 715 de S402 : **195 corrections / 135 régressions = +60 accords**.
- Limite : la sélection est crossfit par composition pour le gate, mais les sorties amont S402/S416 des folds d'apprentissage ne sont **pas complètement réentraînées de façon imbriquée**, d'où possible biais de méta-sélection. La mesure est expérimentale.

## 6. S420 — meilleure chaîne native de notes complètes

[Code S420](https://github.com/Andriamarosoa/note/blob/codex/s276-idmt-song-disjoint-candidate-audit/scripts/s420_native_S419_pitches_S417_noteoff_Klocked_combination.py) ; [run #38384016298](https://github.com/Andriamarosoa/note/actions/runs/38384016298), **success**, artifact `11804871016`.

Applique les **quatre véritables poids S417 figés** aux notes MIDI choisies par S419 ; aucun réentraînement de modèle et aucune correction du comptage S80. Les sorties de note sont gelées **avant ouverture des annotations** JAMS de scoring.

| Note MIDI complète à ±50ms | S412 | S417 | **S420** |
|---|---:|---:|---:|
| N notes prédites | 33 475 | 33 475 | 33 475 |
| Vraies notes annotées | 34 954 | 34 954 | 34 954 |
| Note+Onset appariés | 25 120 | 25 125 | **25 395** |
| F1 MIDI+Onset | 73,42 % | 73,43 % | **74,22 %** |
| Offsets complets corrects | 14 313 | 16 017 | **16 140** |
| Offset correct **conditionnel** | 56,98 % | **63,75 %** | 63,56 % |
| **F1 note complète** | 41,83 % | 46,81 % | **47,17 %** |

S420 est donc meilleur que S417 sur le **F1 global note complète**, même si sa précision Offset **conditionnelle** est légèrement plus faible, car il associe plus de notes correctes à des Pitch+Onset (25 395 contre 25 125).

S420 reste largement sous YourMT3+ original (**75,29 %**) et sous l'hybride assisté directement par le décodeur YourMT3+ S415 (**67,56 %**). S420 n'utilise pourtant aucun poids ni aucune prédiction YourMT3+ pour sa propre inférence.

## 7. S421 — audit indépendant sur les mêmes vraies notes

[Code S421](https://github.com/Andriamarosoa/note/blob/codex/s276-idmt-song-disjoint-candidate-audit/scripts/s421_audit_S420_vs_S417_pairwise_same_true_MIDI_notes_Klocked.py) ; [run #38384577930](https://github.com/Andriamarosoa/note/actions/runs/38384577930), **success**, artifact `11805076211`.

**34 954 notes de vérité identiques** ont été comparées deux fois, en réévaluant les **fichiers MIDI exportés S417/S420**, sans réentraînement ni changement de sélection.

- Notes complètes corrigées S420 vs S417 : **1 248**.
- Notes complètes régressées : **1 125**.
- **Gain net : +123 notes complètes**.
- Par fold composition : **+34 / +74 / −1 / +16** (gain global positif mais une petite régression dans un fold).
- Par durée annotée :
  - <100ms : S417 578/2 263, S420 564/2 263 (−14).
  - 100–499ms : S417 14 899/28 212, S420 15 043/28 212 (**+144**).
  - 500–999ms : S417 511/3 579, S420 504/3 579 (−7).
  - >=1000ms : **29/900** correctes dans les deux versions, encore très insuffisant.
- **Les 59 309 K de S80 ont tous été respectés** après relecture effective des deux exports `NoteOn`.

Le nouveau gain se concentre donc principalement dans les notes **100–499 ms**. Il reste beaucoup de corrections possibles sur les notes longues, et les régressions de petites notes ne sont pas cachées.

## 8. Décision et prochaine direction

- **S80 S397 reste la référence K utilisateur** : 0 régression, 0 modification ; 90,25 % global, 81,98 % poly sur GuitarSet développement.
- **S402 reste la référence de hauteurs active enregistrée** ; S416 seul est rejeté ; S419 est un challenger positif de +60 accords poly.
- **S408 reste la référence Offset active expressément choisie par l'utilisateur** ; S417 constitue une nouvelle tête Offset expérimentale positive.
- **S420 est le meilleur challenger de transcription MIDI complète purement native**, avec **47,17 % de F1** et **0 régression Exact-K**.
- **Aucune promotion scientifique officielle**, car S80 possède un entraînement recoupant GuitarSet, et la méta-sélection S419 n'est pas entièrement imbriquée.

Pour avancer vers YourMT3+ sans sacrifier Exact-K : remplacer les entrées limitées 42×49 autour de l'attaque par un encodeur multi-secondes sur spectre continu, mémoire par voix/corde et décodage causal ou offline lié `NoteOn, Pitch, NoteOff` ; conserver S80 K comme contrainte ferme en sortie et entrée du réseau ; séparer les pertes `pitch, onset, duration/survival, harmonics` et tester **par composition jamais vue et sans recoupement d'entraînement de S80**. Les notes >=1000ms (29/900 seulement) sont l'une des prochaines faiblesses principales.
