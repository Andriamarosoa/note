---
serie: S836
statut: en_cours
type: experience
reference: S708
branche: codex/note-s836-live-sharp-cache256
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S836 — Résultat de la première boucle : cache de phase spectral PCM256

## Exécution
Une implémentation `IncrementalSharp256` facultative a été ajoutée au chemin SHADOW S834. La voie référence reste le mode par défaut ; les six modèles d'origine, leurs décodeurs et le Live/microphone/SoundFont actuel ne sont pas changés. Les premiers runs de CI ont signalé un défaut de métadonnées du rapport et un workflow mal formé : corrections d'infrastructure en cours. **Pas de nouveau score ni gain CPU déclaré avant succès des runs.**

## Métriques
Référence signée S834 FIT80 sur 80 vraies pistes, **pas un unique MIDI** :
- Exact-K : 75,780659 % accuracy SHADOW.
- Active Exact-K : 65,193596 % accuracy SHADOW (S834 ; meilleur privé S828 73,831146 %).
- Onset F1 : 71,382285 % S834, inférieur à S708/H0 73,252019 %.
- Offset : 38,677437 % F1 événement S834, **incomparable** à l'accuracy H0 de 69,886229 %.
- Frame Pitch F1 : 84,998850 %.
- Full Note F1 : 71,108037 %.
- CPU S834 p95 Linux original par bloc : 6,61–7,75 ms contre budget 5,805 ms ; CPU Windows/iRig inconnu.
- Cache S836 : parité spectrale, parité musicale et p95 optimisé **non mesurés au moment de ce rapport**.

## Comparaison à S708
S708 officiel CAL32 reste la référence, inchangée, sur sa propre cohorte et son horloge historique 1024. Cette expérience utilise le même S708 figé en entrées 256 et cherche à réduire **uniquement** le coût CPU. Aucun score FIT80 ne prouve un meilleur CAL32 ou meilleur Live. Aucune tête Onset/Offset historique n'est remplacée ; résultat musical hypothétique = strictement identique si cache bitwise exact.

## Audits
| Rôle | État | Preuve attendue |
|---|---|---|
| Données | UNKNOWN | SHA vrais WAV et splits |
| Acoustique | UNKNOWN | exact 96D / feature [49,288] |
| Architecture | UNKNOWN | six décisions et états inchangés |
| Apprentissage | PASS | code seulement, checkpoints source figés |
| Métriques | UNKNOWN | mêmes TP, FP, FN par axe |
| LIVE | UNKNOWN | CPU p95/p99 global + réel Windows/iRig |

## Verdict
**NON PROMU — EXPÉRIENCE EN COURS**. Un gain de la seule FFT synthétique ne suffit pas à prouver un meilleur moteur MIDI Live. Le code n'écrit pas dans le MIDI officiel. Ne pas introduire d'attente de données au-delà du bloc de 256.

## Prochaine action
Corriger les contrôles CI, mesurer la parité bit-à-bit et le budget CPU isolé, comparer cinq vrais WAV signés dans les cinq compositions avec mesures appariées de chaque axe et contre-exemples, puis décider d'une extension FIT80 complète. Toute régression ou gain insuffisant déclenchera un audit d'explication causale avant une autre variante.
