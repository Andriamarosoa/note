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

# S836 — Accélérer le vrai LIVE NOTE à décisions PCM256 sans régression

## Objectif
Rendre la voie de recherche S834/S835 à six caractéristiques compatible avec le budget temporel réel strict de NOTE. Garder le moteur Live S613/S614 actuel et le MIDI officiel inchangés. Une intervention de cache spectral ne doit modifier ni notes ni scores. Préparer la validation appariée de tous les axes et la mesure sur Windows/iRig sans prétendre qu'une unification est déjà réalisée.

## Hypothèse
La voie S834 calcule à chaque nouveau 256 des données `sharp_track(audio,[n-256,n])` en refaisant les trois FFT de l'intervalle précédent. Ces FFT ont déjà été calculées lors du hop passé. Conserver leurs spectres complexes interpolés et leurs constantes devrait accélérer les entrées spectrales 96D, **sans modifier leurs valeurs numériques ni la prédiction S708**. C'est une hypothèse à mesurer, pas un score validé.

## Modifications
- `src/causal_note/s836_incremental_sharp256.py` : objet `IncrementalSharp256`, cache des spectres du hop précédent uniquement, aucune lecture du futur.
- `src/causal_note/s834_s809_native256_audio.py` : argument optionnel `sharp_engine="incremental"` ; le défaut `"reference"` préserve intégralement S834.
- `scripts/s834_s809_native256_fit80_replay.py` : option `--sharp-engine` pour les mêmes WAV/poids/truth FIT80, sorties scientifiques conservées.
- `test/test_s836_incremental_sharp256.py` : vérification stricte des 96 features, du rolling 8192, de l'audio futur, de la continuité temporelle et des six décodeurs privés.
- `scripts/s836_sharp_cpu_benchmark.py` : comparaison par p50/p95/p99 du seul calcul spectral sur synthétique.
- Nouveau workflow S836, aucune modification des têtes Exact-K, Active Exact-K, Onset, Offset, Frame Pitch, Full Note et aucun NoteOn/NoteOff officiel modifié.

## Protocole
- Signal PCM mono 44 100 Hz ; **256 nouveaux échantillons** et une décision au même front : 5,80499 ms.
- Aucun audio futur, zéro lookahead, aucune attente de confirmation, aucune file ajoutée, horodatage non antidaté.
- Les six axes utilisent leurs HEAD→SELECTOR→DECODER et états indépendants ; aucun axe ne se sert des sorties d'un autre.
- La référence H0 S708 et les modèles S828/S834/S835 restent figés. Onset S834 inférieur au H0 et Offset F1 non comparable à Offset accuracy H0 : veto inchangé.
- Évaluation appariée sur mêmes 5 compositions GuitarSet en premier, puis 80 pistes si validé ; poids vérifiés SHA, source signée, aucune donnée CAL/HELD/TEST utilisée.
- Comparaison stricte ABI 96D `np.array_equal`, décisions et comptages par axe, TP/FP/FN inchangés, p50/p95/p99 CPU et budget global p95/p99 5,805 ms.
- Si amélioration nulle ou régression : autopsie du mécanisme et des contre-exemples, **ne pas changer simplement de méthode**.

## Audits obligatoires
| Rôle | Contrôle |
|---|---|
| Données | provenance et SHA source PCM identiques pour les deux runs |
| Acoustique | égalité bit-à-bit de la banque S706 96D et source [49,288] |
| Architecture | 6 têtes isolées, autres cinq axes inchangés |
| Apprentissage | poids S708 gelés, aucun entraînement contaminant |
| Métriques | six paires GT, même FIT80, ne pas confondre CAL32 |
| LIVE | cadence 256 et CPU réel mesuré ; Windows iRig reste nécessaire |

## Statut
En cours. Aucune amélioration musicale ni vitesse du chemin complet revendiquée ; six axes unifiés non validés et aucune promotion. Aucun agent NVIDIA/Work externe n'a été appelé dans cette série.
