---
serie: S848
statut: en_cours
type: experience
reference: S708
branche: codex/note-s848-ia-vs-standard-transpose256
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S848 — Audio réel transposé avec méthode STANDARD vs méthode IA S847

## Exécution
A/B/C sur **le même enregistrement original** : (1) vraie guitare originale, (2) transposition standard offline +12 par `librosa.effects.pitch_shift`, (3) sortie sélection par réseau neural S847 +12 avec vraies notes GuitarSet. Même échantillonnage de restitution 44100Hz, même durée. Le troisième son n'est pas une vraie transposition de la performance mais la **sélection par IA** de clips enregistrés physiquement et rejoués.

## Métriques
| Cible | Résultat |
|---|---|
| Méthode standard +12 | **NON MESURÉ** (référence benchmark à exécuter) |
| Sortie IA vs standard (timbre et dynamique) | **NON MESURÉ** |
| Audio original vs sorties | **NON MESURÉ** |
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
| Données | UNKNOWN | source guitare réelle SHA et checkpoint S847 véritable vérifiés |
| Acoustique | UNKNOWN | mêmes entrée/durée/hauteur cible pour référence et IA |
| Architecture | PASS code | IA note et échantillons réels, transposition standard indépendante |
| Apprentissage | PASS S847 | checkpoints réels existants ; pas de nouvel apprentissage |
| Métriques | UNKNOWN | comparer sorties WAV, profils acoustiques, dynamique et choix de notes |
| LIVE | IA PCM256 ; standard non causal | pas de comparaison en latence, pas de modifications MIDI |

## Verdict
**Non promu** avant la comparaison effective. Le simple fait que les notes échantillonnées proviennent de vrais enregistrements n'est pas une preuve qu'une IA sait transposer la même phrase jouée. Une référence standard permet de le démontrer clairement si l'écart de timbre/pitch/dynamique est significatif.
