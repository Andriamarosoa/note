---
serie: S834
statut: termine
type: experience
reference: S708
branche: codex/note-s834-s809-six-axes-native256
run_id: 39658093988
commit_sha: 4d5c445c6f44d1c956f781f85764fe12bfba4983
promotion: false
---

# S834 — Portage fidèle S809 six axes natif256, analyse scientifique terminée

## Objectif
Porter fidèlement les six architectures S809 HEAD→SELECTOR→DECODER à une nouvelle décision toutes les 256 valeurs PCM44,1 kHz, sans attendre quatre hops, sans partager les états/sorties des têtes ni modifier le MIDI officiel. Exiger non-régression des cinq autres caractéristiques avant toute promotion.

## Hypothèse
Les têtes/arbres de décision indépendants de S809 peuvent recevoir des postérieurs harmoniques S708 **fraîchement recalculés** à chaque nouvelle frontière PCM256 et reconstruire les bonnes caractéristiques de S809 à 1024 sans réutiliser les snapshots S800 as-of-1024. La séparation des six axes se vérifie par mutation adversariale de chaque sélecteur.

## Modifications
- `src/causal_note/s834_native256_decoders.py` : port isolé S763 1024→256 sans mutation de l'original. Suppression du hold additionnel Full Note ; pas de nouveau lookahead.
- `src/causal_note/s834_s809_six_axis_pcm256.py` : copie fidèle de la famille de six HEAD/SELECTOR/DECODER S809 avec copies immuables et identités d'axes.
- `src/causal_note/s834_s809_native256_audio.py` : vraie acoustique S708 288D, origine audio PCM et poids gelés, chaque hop actuel 256 produit un nouveau postérieur `[49,3]` et une distribution K `[7]` disponibles aux six axes seulement en lecture.
- `scripts/s834_s809_native256_fit80_replay.py` : vérification des sources signées et du MIDI H0, six diagnostics SHADOW comparés sur 80 pistes, six contre-exemples par fenêtre (5 axes étrangers), 64 premières fenêtres de chaque piste.
- `test/test_s834_s809_native256_axes.py` : tests adversariaux S809 portés à PCM256.

## Protocole
80 originaux GuitarSet FIT80 S800 signés, 5 compositions ×16 pistes, **306 536 frontières PCM256** à 44,1 kHz, aucune entrée de vérité lors de l'inférence, et 80×64×6×5 = **153 600 invariances inter-axes**. Même MIDI H0 historisé en témoin. Toutes les mesures des six axes SHADOW sont DISTINCTES des métriques officielles CAL32 et d'un MIDI final fusionné.

## Audits obligatoires
- Données : PASS, 80 WAV authentiques, checkpoint S708 identifié SHA.
- Acoustique : PASS pour le recalcul par hop et l'absence de futur, MAIS modèle S708 appris historiquement à 1024 et repris gelé à 256.
- Architecture : PASS, 41 tests initiaux, 153 600/153 600 interactions sans influence sur les cinq autres.
- Apprentissage : PASS limité — **pas de réentraînement** 256 des poids S708 et pas de split OOF garanti de son apprentissage initial.
- Métriques : PASS pour calcul exact des six scores SHADOW, mais Onset baisse de **73,2520 % H0 FIT80 à 71,3823 % S834** ; Offset F1 n'est pas comparable à Offset accuracy H0.
- LIVE : FAIL comme **preuve du budget strict**, pas comme algorithme nécessairement non causal : sur Linux GitHub, les 80 p95 CPU de traitement bloc sont entre 6,61 et 7,75 ms, tous supérieurs à la période 5,805 ms ; p99 entre 6,81 et 9,50 ms. Ce ne sont ni p95 physiques iRig ni tests de production, la latence matérielle reste UNKNOWN.
- Aucun agent NVIDIA / Work réellement appelé.

## Statut
**TERMINÉ, NON PROMU.** [GitHub Actions #39658093988](https://github.com/Andriamarosoa/note/actions/runs/39658093988) SUCCESS, code scientifique `4d5c445c6f44d1c956f781f85764fe12bfba4983`, artifact GitHub `S834_REAL_S809_SIX_AXES_PCM256_FIT80` id `12376305671`. Six scores S834 : Exact-K 75,7807 %, Active Exact-K 65,1936 %, Onset F1 71,3823 %, Offset F1 38,6774 %, Frame Pitch F1 84,9989 %, Full Note F1 71,1080 %. Exact-K améliore S829 57,5247 % de 18,2566 points sur les 306 536 mêmes fenêtres. **Aucune régression d'interférence technique** entre architectures ; cela NE garantit PAS la non-régression musicale de chaque caractéristique. Onset est en régression sur le H0 signé, et coût CPU >5,805 ms. Le témoin S708 officiel CAL32 et S828 sont intacts. S834 doit rester research SHADOW.
