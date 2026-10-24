---
serie: S848
statut: termine
type: experience
reference: S708
branche: codex/note-s848-ia-vs-standard-transpose256
run_id: 40095759550
commit_sha: 61e1d4132fcbfa895a13194f0dc03de7bc69b31d
promotion: false
---

# S848 — Audio réel transposé avec méthode STANDARD vs méthode IA S847

## Exécution
A/B/C sur **le même enregistrement original** : (1) vraie guitare originale, (2) transposition standard offline +12 par `librosa.effects.pitch_shift`, (3) sortie sélection par réseau neural S847 +12 avec vraies notes GuitarSet. Même échantillonnage de restitution 44100Hz, même durée. Le troisième son n'est pas une vraie transposition de la performance mais la **sélection par IA** de clips enregistrés physiquement et rejoués.

## Mesures finales du vrai test A/B/C

[GitHub Actions #40095759550](https://github.com/Andriamarosoa/note/actions/runs/40095759550) **SUCCESS** : même segment original physiquement joué pendant **3,500 s**, 154350 échantillons à 44100 Hz, comparé à une référence `librosa.effects.pitch_shift` +12 demi-tons et au checkpoint réellement entraîné S847 à entrée 256. Les tests logiciels S848 et S847, le checksum Git du WAV et les 39 vraies notes de banque ont été vérifiés.

**Artefact audio réel :** `S848_REAL_GUITAR_STANDARD_VS_AI_PLUS12_AB_AUDIO` (GitHub Artifact ID **12561195961**), contenant :
- `01_entree_guitare_reelle.wav` — entrée originale,
- `02_transposeur_standard_plus12.wav` — transposition standard offline +12,
- `03_transposeur_IA_S847_plus12.wav` — IA S847 réelle sans DSP,
- `04_comparaison_entree_standard_IA.wav` — les trois passages enchaînés avec séparateurs,
- `comparaison_resultat.json` — toutes les mesures reproductibles.

| Propriété mesurée sur le MÊME audio | Original | Standard +12 | IA S847 +12 |
|---|---:|---:|---:|
| Durée | 3,500 s | 3,500 s | 3,500 s |
| RMS brut | **0,18057** | **0,14460** | **0,03588** |
| Médiane spectrale | **199,18 Hz** | **403,75 Hz** | **554,48 Hz** |
| Centroïde spectral | 621,57 Hz | 1091,02 Hz | 620,23 Hz |
| Énergie >10kHz | 1,570 % | 2,274 % | 0,00087 % |
| Corrélation d'enveloppe avec la guitare originale | 100 % | **90,384 %** | **10,541 %** |
| Corrélation d'enveloppe avec standard +12 | — | 100 % | **11,643 %** |
| Distance IA vs standard log-mel absolue | — | — | **18,3773 dB** |

**Échec net : l'IA choisit exclusivement le MIDI 73 (C#5) sur 603/603 blocs** (aucun bloc de silence), alors qu'une performance de guitare possède une évolution temporelle et une enveloppe. Le son vient d'un clip réel de guitare, mais il ne transpose PAS fidèlement cette performance ; il rejoue continuellement le même échantillon. Le standard préserve beaucoup mieux la dynamique du morceau. La médiane spectrale du standard passe de 199 à 404 Hz (×2,027), cohérente avec +12 ; une médiane spectrale n'est toutefois pas une mesure indépendante de fréquence fondamentale, donc cela ne suffit pas à établir la justesse de chaque note.

**La CI SUCCESS signifie que la comparaison s'est déroulée intégralement, PAS que S847 est compétitif. Verdict musical : S847 échoue.** La sortie IA est réelle (pas une onde sinusoïdale synthétique), mais la sélection est catastrophique. Ce test met en évidence le défaut qui était caché derrière l'existence de vrais WAV source. Le standard est un benchmark offline non causal et ne saurait être déclaré conforme à la contrainte Live PCM256.

## Métriques
| Cible | Résultat |
|---|---|
| Méthode standard +12 | **MESURÉ** : médiane spectrale 403,75 Hz et 90,384 % corrélation d'enveloppe |
| Sortie IA vs standard (timbre et dynamique) | **ÉCHEC** : 11,643 % corrélation d'enveloppe, 18,3773 dB de distance log-mel |
| Audio original vs sorties | **MESURÉ** sur le même enregistrement : IA sélectionne MIDI73 pendant 603/603 blocs |
| S847 +12 pour E2→E3 | **14,74365 % F1** de présence de note active (TP3123, FP35352, FN766) : insuffisant |
| S847 -12 pour E3→E2 | 18,307 % F1 de présence de note active (TP4945, FP44125, FN8) |
| Référence Onset travail PCM256 | **69,2584331 % F1**, inchangée |
| Exact-K | inchangé |
| Active Exact-K | inchangé |
| Offset | inchangé |
| Frame Pitch F1 | inchangé |
| Full Note F1 | inchangé |
| MIDI | inchangé |

## Comparaison à S708
S708 historique entraîné à 1024 non repris ; modèle S847 et nouvelle évaluation S848 restent des études indépendantes en PCM256. Le DSP du standard est strictement isolé dans le benchmark offline.

## Audits
| Catégorie | Statut | Contrôle |
|---|---|---|
| Données | PASS | même guitariste WAV origine SHA Git et checkpoint S847 authentique, 39 notes physiques |
| Acoustique | FAIL IA | même entrée et durée, la sortie IA est une unique note répétée, sans enveloppe comparable |
| Architecture | PASS code | IA note et échantillons réels, transposition standard indépendante |
| Apprentissage | PASS S847 | checkpoints réels existants ; pas de nouvel apprentissage |
| Métriques | FAIL IA | comparaison spectre, dynamique, choix de notes réalisée et archivée, aucune transposition correcte |
| LIVE | IA PCM256 ; standard non causal | pas de comparaison en latence, pas de modifications MIDI |

## Verdict
**TERMINÉ, NON PROMU : IA S847 très inférieure à la référence standard.** Source commune, WAVs et mesures confirmés. Répétition MIDI73 (603/603 blocs), corrélation dynamique 11,6 % vs standard : l'IA ne transpose pas la phrase. Le succès technique du run n'est pas une preuve musicale de la méthode.
