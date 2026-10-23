---
serie: S836
statut: en_cours
type: experience
reference: S708
branche: codex/note-s836-live-sharp-cache256
run_id: 39989523624
commit_sha: dad2ac7ff454014a8a2896d5fac7a4b407e7a9f7
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

## Résultats contrôlés et protocole des boucles
Boucle CPU initiale [#39986203235](https://github.com/Andriamarosoa/note/actions/runs/39986203235) : 22 tests PASS, FFT S706 médiane 0,811 → 0,258 ms sur signal identique; 5 vrais enregistrements (1 accompagnement dans chacune des 5 compositions), 20 062 hops; scores égaux, 9 600 invariances. S836 p95 5,288–5,459 ms et p99 5,500–5,792 ms sur ce runner Linux uniquement.

Boucle de contrôle SHA [#39988295532](https://github.com/Andriamarosoa/note/actions/runs/39988295532) : 23 tests PASS ; empreintes source 288D, spectre 96D et chaque décision/sélection de six axes identiques **sur chacun des 20 062 hops**. Ce runner trouve p95 S836 5,692–5,760 ms (<5,805), mais p99 S836 5,849–6,079 ms (>5,805). **Veto LIVE p99 effectif**, ne pas sélectionner seulement le meilleur run.

Boucle d'audit de reproductibilité [#39989523624](https://github.com/Andriamarosoa/note/actions/runs/39989523624) : 23 tests PASS, triple replay signé de mêmes cinq vrais WAV avec même S708 gelé et même H0 : référence S834 SHA, cache S836 SHA, cache S836 sans SHA. Les métriques restent identiques ; S836 p95 3,288–3,670 ms, p99 3,800–4,710 ms avec SHA, et p99 3,757–4,346 ms sans SHA. Le témoin S834 lui-même varie de 3,845–4,166 ms p95 sur ce runner, contre 6,4–6,9 ms sur les autres. Ainsi la **plateforme CI/mesure** est un facteur confondant considérable, et le coût SHA n'explique pas systématiquement chaque variation. Par exemple le p99 du cache sans SHA est plus grand que celui du cache avec SHA sur Jazz1. L'explication thermique/ordonnanceur reste hypothétique.

Le [rapport d'échec intermédiaire](audit_ci_failure.md) explique aussi le défaut d'instrumentation réel `NameError nn_post` de la boucle #39987476143, corrigé avec test d'appel réel au modèle. Ce n'était ni une régression musicale du cache, ni une amélioration scientifique.

**Blocage de promotion non levé** : pas de contrôle FIT80 intégral 80 WAV pour cette optimisation ; les cinq morceaux testés sont les premiers accompagnements `_comp` et ne couvrent pas les 75 autres pistes ; aucune donnée de vraie performance microphone/FluidSynth Windows/iRig, aucune garantie p99 sur une autre machine et aucune tête S708 véritablement entraînée sur une décision native 256. Il faut mesurer p95/p99 robuste sur la machine cible, protéger chaque tête, puis aborder séparément la régression Onset S834 et la faiblesse Offset.
