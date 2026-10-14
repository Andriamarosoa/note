# S428–S430 — Reconnaissance directe des notes musicales **son → E3 / [E3, G3, B3]**

**14 octobre 2026.** L'objectif utilisateur est la **reconnaissance des notes et des accords**, pas le fichier MIDI ni le traitement Live. Ces expériences ne modifient **ni le comptage S80, ni les références sélectionnées, ni le Live ni les détecteurs d'Offsets**.

## Résultat de précision des hauteurs

S428 a réellement entraîné, sur les 42×49 caractéristiques acoustiques GuitarSet déjà reconstruites, **quatre nouveaux modèles de classement de hauteurs par composition tenue à l'écart**. Chaque modèle combine la fondamentale estimée, la naissance temporelle de l'attaque, les harmoniques à 12/19 demi-tons, les notes voisines, les relations de quinte, les probabilités préentraînées des réseaux S402/S416 et **K prédit S80 injecté dans les caractéristiques**.

Le choix de faire confiance au nouveau modèle par K est pris **uniquement à partir d'autres compositions** (sélection interne, sinon on conserve S419). Le nouveau modèle émet exactement les **K noms de notes musicales**, K0 = liste vide.

| Identité musicale exacte | S419 ancien meilleur | S428 nouveau modèle | Écart |
|---|---:|---:|---:|
| Vrais événements poly K2..K6 | 7 385 | 7 385 | — |
| Accords entièrement exacts | 2 775 | **3 025** | **+250** |
| Précision des accords poly | 37,58 % | **40,96 %** | **+3,39 points** |
| Vraies notes poly individuelles retrouvées | 15 444/20 894 | **15 855/20 894** | **+411** |
| Rappel des notes poly individuelles | 73,92 % | **75,88 %** | **+1,97 point** |
| Accords poly corrigés | — | **345** | |
| Accords poly régressés | — | **95** | |
| Exact-K global original | 53 529/59 309 | **identique** | **0 régression** |
| Exact-K poly | 6 054/7 385 | **identique** | **0 régression** |

Résultats mesurés, pas estimés : [run S428 #38517551029](https://github.com/Andriamarosoa/note/actions/runs/38517551029), **success**, artefact `11871374161`. Code : `scripts/s428_train_sound_to_note_names_harmonic_ranker.py`.

### Audit de précision par vrai K

| Vrai K | N événements | S419 accords corrects | **S428 corrects** | Net |
|---:|---:|---:|---:|---:|
| 0 | 39 652 | 38 432 | **38 432** | 0 |
| 1 | 12 272 | 8 100 | **8 232** | **+132** |
| 2 | 3 445 | 1 338 | **1 495** | **+157** |
| 3 | 2 289 | 1 009 | **1 084** | **+75** |
| 4 | 1 207 | 321 | **339** | **+18** |
| 5 | 355 | 93 | **93** | 0 |
| 6 | 89 | 14 | **14** | 0 |

Le nouveau modèle n'est activé que lorsqu'il est positif sur les compositions internes de développement. K5 et K6 **ne régressent pas**, mais ne progressent pas non plus en accords entiers. Ces catégories rares demandent un apprentissage plus spécifique.

## S429 — Validation indépendante des *sorties gelées*, avec confusion musicale

La [validation S429 #38517797080](https://github.com/Andriamarosoa/note/actions/runs/38517797080), **success**, artefact `11872831104`, exécute un **autre script d'audit** sur les 59 309 listes réelles de noms exportées par S428 et retrouve exactement : 3 025 accords poly, +345 corrections / −95 régressions, +250 net, **zéro modification des 59 309 K prédit**.

Sur les événements vrais K2 dont l'accord n'est pas entièrement reconnu, l'audit relève :
- **357** confusions de hauteur séparées d'une octave,
- **126** confusions de demi-ton ou ton,
- **209** confusions classées quintes ou harmoniques à 19 demi-tons,
- d'autres erreurs d'intervalles et de cardinalité provenant d'un K prédit erroné.

Ce diagnostic distingue le comptage K correct de l'identité des notes. Attention : ce sont des catégories d'erreurs calculées par appariement des hauteurs restantes, pas un diagnostic causal prouvé de la physique du son.

Code : `scripts/s429_audit_actual_E3_G3_B3_note_names_59309_S80_Klocked.py`.

## S430 — Inférence utilisable sur **nouveaux événements acoustiques prétraités**

Code : `scripts/s430_infer_note_names_from_new_S80_sound_events.py`.

Ce programme charge les vrais modèles entraînés dans S402/S416/S428 et ne lit **aucune annotation**. Il prend :
1. un fichier `.npz` contenant `row_id` et `sequence42x49` (caractéristiques du son) ;
2. un autre `.npz` contenant les mêmes `row_id` et **`s80_K` déjà prédit** ;
3. les quatre modèles entraînés S402, S416, S428 et les politiques de sélection documentées S419/S428.

Il ressort `notes_from_new_sound.jsonl`, par exemple :

```json
{"source_row_id":"nouvel-evenement-1","K":1,"notes":["E3"],"MIDI_pitches":[52]}
```

Ou plusieurs hauteurs :

```json
{"source_row_id":"nouvel-evenement-2","K":3,"notes":["E3","G3","B3"],"MIDI_pitches":[52,55,59]}
```

**Ce sont des exemples de format et NON des prédictions vérifiées pour un WAV particulier.**

Le runtime agrège quatre modèles sur des événements entièrement nouveaux ; **la politique de vote de cet ensemble n'a pas encore de score indépendant**. Il ne faut jamais appeler ses sorties sur un morceau GuitarSet déjà vu une évaluation hors entraînement.

**Important : S430 n'est PAS encore WAV brut → E3.** La première transformation du WAV en `42×49` et le K S80 doivent venir de la chaîne acoustique d'amont. Le développement et la validation d'un point d'entrée **WAV directement en noms musicaux**, avec reconstruction non ambiguë de toutes les entrées historiques, restent à terminer.

## Conservation des travaux antérieurs

- **S80 S397** : K0..K6 inchangés ; 53 529/59 309 (90,25 %) global et 6 054/7 385 (81,98 %) poly sur GuitarSet développement.
- **S402** : référence de notes officielle de développement conservée ; **S419** reste l'ancien challenger, **S428** devient le meilleur challenger pour **l'identité des notes musicales**.
- **S408** : référence utilisateur Offset toujours inchangée.
- **S425** : meilleur challenger **transcription complète avec Offset**, à 48,35 % F1 et sans régression Exact-K. **Ne pas lui attribuer le gain de S428** sans réentraîner/retester la chaîne correspondante.
- **S424 Live** : moteur historique et pont Live MIDI conservés, **non modifiés**. Priorité actuelle = notes musicales hors ligne.
- Comparaison directe avec YourMT3+ sur le même GuitarSet n'est **pas encore indépendante** : S80 a déjà vu des données GuitarSet dans le passé, et les modèles auxiliaires crossfit S402/S416 ne sont pas entièrement imbriqués au méta-sélecteur S428.

## Suite recommandée

La priorité est de réduire les erreurs d'octaves, de fondamentales faibles et de notes K5/K6, en testant les corrections contre S428 par classe K et en ne gardant que les gains vérifiés. Il faudra ensuite une **validation totalement indépendante sur des sons et morceaux non exposés** à S80 ni aux modèles de notes, puis connecter la reconstruction des caractéristiques depuis WAV brut à la nouvelle interface de prédiction de noms de notes. **Live et NoteOff ne sont pas les priorités actuelles.**
