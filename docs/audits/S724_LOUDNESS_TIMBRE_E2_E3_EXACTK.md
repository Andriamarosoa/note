## Résultat final S724 — 18 octobre 2026 (GuitarSet original, sans gain artificiel)

[Run S724 SUCCESS, 21 tests réussis](https://github.com/Andriamarosoa/note/actions/runs/39118797590), 80 FIT pour l'entraînement ; 32 CAL pour la sélection ; 32 HELD/12 DEV uniquement pour rejouer le **modèle sélectionné** (référence historique S708).

| Score CAL | S708 figé | S724 1 époque + K | S724 2 époques + K | S724 2 époques sans guide faibles-K |
|---|---:|---:|---:|---:|
| Onset F1 | **74,2639%** | 74,2481% | 74,2494% | 73,0446% |
| Exact-K | 62,0455% | 62,3892% | **62,4454%** | 59,8500% |
| Exact-K actif | 56,2762% | 56,8479% | **56,9845%** | 53,4062% |
| Offset | **77,3292%** | 76,5657% | 76,6115% | 76,9313% |
| F1 notes complètes | **57,1319%** | 56,5250% | 56,5591% | 55,8759% |
| F1 hauteurs / frame | 76,4626% | **76,8728%** | 76,7524% | 73,9989% |

**Conclusion** : nouvelles caractéristiques de timbre + volume relatif = gains exploratoires d'Exact-K et des hauteurs sur CAL, mais **pas de gain Onset**, et dégradation Offset/full-note. Toutes les variantes échouent aux gardes S708 sur les six scores. Seul `S708_frozen` est éligible et les six métriques HELD/DEV du choix sont identiques à S708. `promoted=false`. Cela ne prouve pas la nullité des indices timbraux : 1–2 époques de FIT, leur architecture et la composition du corpus limitent la conclusion.

L'écart de performance `Kguided` vs `KweakOFF` démontre la valeur du **guidage K des attaques faibles** pour cette architecture, et non l'utilité spécifique des features de timbre (les deux modèles utilisent les mêmes données/features). La tête K neuronale reste active dans les deux.

Le gain stress réel `x0.5/x1/x2` est une **seconde épreuve**, distincte de ce résultat S724 : [GitHub Actions](https://github.com/Andriamarosoa/note/actions/runs/39120010839). Ne pas affirmer une invariance au volume avant son rapport définitif.

---

# S724 — Intensité de jeu, timbre harmonique et E2 → E3, avec Exact-K

## Question scientifique

Une augmentation du gain d'enregistrement ou du volume de pincement
doit-elle créer une *nouvelle note* ? Non. Le système doit reconnaître
la hauteur déjà active, tolérer des intensités variables et distinguer
une nouvelle excitation indépendante (E3) du renforcement des
harmoniques de E2. Cela se fait conjointement avec Exact-K, jamais à
la place d'Exact-K.

## Diagnostic de code initial

- `strict_live.py` extrait **à la fois** 24 amplitudes spectrales
  absolues, 24 relatives, les deltas temporels, trois flux spectraux
  globaux et deux niveaux RMS; la détection **n'est pas** strictement
  invariante au volume.
- `scripts/live_input_level.py` dispose de `--auto-level`
  (facultatif, plafonné +18 dB par défaut). C'est un gain de
  micro, **pas** une solution qui sépare E2 et E3; l'algorithme
  évite volontairement de monter le gain pendant les notes en
  décroissance pour ne pas fabriquer de faux Onsets.
- `s706_sharp_spectra.py` expose déjà 96D (énergie/harmoniques
  et phases), mais les rapports *paires / impaires du parent E2*
  ne sont pas une feature dédiée.
- E2 (82.41 Hz), E3 (164.81 Hz) : les partiels d'E3
  correspondent idéalement aux partiels **pairs** d'E2.
  Une nouvelle E3 peut donc faire croître les partiels pairs
  du parent E2 davantage que les impairs. Une augmentation
  **globale et uniforme** du volume d'E2 ferait aussi
  croître ses partiels impairs. Ceci est un indice
  probabiliste, pas une preuve universelle (attaque, spectre,
  inharmonicité, EQ, niveau, bruit, même corde vs autre corde).

## La nouvelle ABI append-only S724 320D

Le modèle reçoit les 288D S708 d'origine **intacts** + 32D :
- 8 croissances positives logarithmiques de partiels de la
  hauteur considérée (moyenne sur 3 FFT);
- 8 évolutions de répartition spectrale **centrées par pitch**
  (normalisation relative à la variation de niveau moyen des
  8 harmoniques); cela vise à séparer gain commun et changement
  de timbre;
- 8 formes de timbre courantes relatives
  (log-amplitude – moyenne des 8 partiels);
- 3 différences d'évolution harmoniques paires/impaires sur
  la note située **12 demi-tons** plus bas (3 résolutions);
- 3 différences d'évolution des harmoniques multiples de 3
  sur la note située **19 demi-tons** plus bas;
- 2 indicateurs d'existence des hauteurs parentes;
  ne pas confondre E2 et son voisin fictif sous MIDI40.

Le nouveau `src/causal_note/s724_loudness_timbre.py` utilise
uniquement le bloc décisionnel courant et le bloc reçu
précédemment. Pas de labels/oracle Exact-K, pas d'anticipation,
pas de normalisation des queues de notes à volume constant.
**Limitations** : les transformations `log1p` et la dynamique
physique du pincement ne sont pas parfaitement invariantes
en dB, surtout à faible SNR.

## Validation

- `test/test_s724_loudness_timbre.py` : absence de look-ahead ;
  aucune réponse E3 du contraste octave lorsque **tous**
  les partiels E2 montent uniformément (contrôle volume);
  croissance du contraste lorsque seuls les harmoniques
  pairs E2 augmentent (modèle simplifié d'attaque E3),
  indépendance des nouvelles caractéristiques de timbre
  vis-à-vis d'un gain uniforme *dans le domaine log* ;
  registre graves masqué, test de parité initiale 288→320D.
- `scripts/s724_timbre_exactk_training.py` : reprise de la
  procédure S723 éprouvée, **vrai entraînement** des
  3 têtes Onset/Sustain/Release + la tête Exact-K K0–6
  ensemble, après 1 et 2 époques sur 80 sources FIT,
  checkpoint originel S708 reproduit à epoch 0.
- Sélecteur historique K **actif**, ablation du guidage
  faible Onset K uniquement (ne désactive pas le
  conditionnement neuronal par Exact-K).
- Comparaison finale CAL (32 sources) sur **six scores
  du même flux d'événements** : Onset F1, Exact-K,
  Exact-K actif, Offset, F1 notes complètes, F1 hauteurs
  par trame. Interdiction de perte d'Offset déjà correct
  ou d'attaque longue reconnue. HELD et DEV évalués
  après choix CAL seulement, splits historiquement
  exposés. Pas de TEST inédit.
- `promoted=false` sans audit; aucune modification
  des fichiers de pilotage microphone et LIVE S708+S707.

## Résultat

[Workflow S724 — véritable entraînement / CAL / HELD / DEV](https://github.com/Andriamarosoa/note/actions/workflows/note-s724-volume-timbre-exactk.yml).

**Aucune performance sur vraie guitare ne peut être déduite des
tests synthétiques**. Ne pas annoncer une amélioration de Onset,
Offset ou Exact-K avant résultat du rapport `report.json`.

## Prochaine validation à réserver

Un véritable balayage de gain PCM sur les **mêmes WAV annotés**,
avec niveaux prédéfinis –6 dB / 0 dB / +6 dB, en reportant le
taux d'échantillons écrêtés, permettra de mesurer
l'invariance des **événements émis** au gain, sans modifier
la façon dont les guitaristes pincent les cordes. Le jeu
avec doigts plus doux / plus forts exige en plus des prises
physiques dédiées; multiplier une forme d'onde ne
reproduit pas le changement réel de timbre de pincement.


## Mesure de robustesse au volume sur de vrais fichiers CAL

[Workflow de contrôle gain réel S724](https://github.com/Andriamarosoa/note/actions/runs/39119279340) ; script `scripts/s724_gain_robustness_cal12.py`. Cet audit attend le checkpoint S724 attesté par GitHub Actions puis recalcule **depuis le PCM** toutes les caractéristiques natives et sharp, les probabilités et les vrais NoteOn/NoteOff (pas simplement les features déjà mémorisées) pour 12 pistes CAL fixes, six accords et six solos.

- Gain `0.5` (environ –6.02 dB), `1.0` (original), `2.0` (+6.02 dB), sur les **mêmes prises annotées**.
- Les deux modèles S708 et S724 époque 2 gardent exactement le même décodeur guidé par Exact-K et le même S707 causal Offset.
- Le code reconstruit les 288D originaux depuis le PCM, et impose leur égalité au chemin historique pour les exemples originaux à gain 1, avant d'ouvrir les annotations.
- Rapporte les six métriques, les TP/FP/FN, la différence de chaque score par rapport à la condition gain 1, **le nombre d'échantillons écrêtés** après la multiplication. Une variante +6 dB avec clipping ne prouve pas une robustesse à un gain propre ; elle doit être signalée séparément.
- Le test **n'intervient pas dans la sélection CAL du modèle S724** et n'est pas une validation indépendante HELD/DEV.
- **Ne pas confondre** cette multiplication PCM avec la force physique de pincement (l'intensité peut aussi modifier le spectre et la phase). Les enregistrements physiques faible/moyen/fort devront être ajoutés ultérieurement pour vérifier cette généralisation.
