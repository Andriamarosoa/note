# NOTE S753→S754 — Audit des deux causes du sous-comptage

**Statut au 19/10/2026** : S753 mesure achevée, S754 branche de recherche non promue.

## Preuve S753 — véritable GuitarSet CAL32

GitHub run SUCCESS :
https://github.com/Andriamarosoa/note/actions/runs/39209836870

25 tests réussis. Les **31 568 décisions** sont à l'horloge S708
4×256 échantillons, pas à l'horloge native 256 des six scores
signés. Les valeurs ci-dessous sont des statistiques auxiliaires,
**pas une nouvelle Exact-K officielle**.

| K réel | Décisions | Tête Exact-K correcte | MIDI Exact-K correct | Trames MIDI sous-comptées | Dont tête assez haute mais MIDI sous-compté |
|---|---:|---:|---:|---:|---:|
| K1 | 13 267 | 84,887 % | 79,423 % | 1 997 | 1 264 |
| K2 | 3 716 | 45,963 % | 36,518 % | 2 036 | 987 |
| K3 | 2 967 | 39,535 % | 33,468 % | 1 583 | 828 |
| K4 | 4 180 | 51,818 % | 33,278 % | 2 602 | 1 233 |
| K5 | 750 | 21,733 % | 9,733 % | 653 | 121 |
| K6 | 189 | **0 %** | 6,878 % | 176 | 0 |

Par construction `both_under + head_enough_but_midi_under =
emitted_k_under` pour chaque K, vérifié avec tests synthétiques.

- **P1 Tête** : pour K6 (189 fenêtres), modèle K0..6 n'a jamais
  sélectionné K6. La probabilité K6 moyenne aux vraies fenêtres K6
  était seulement 4,11 %.
- **P1 Sélecteur** : une part significative des sous-comptages K2-K4
  survient même si la tête prédit au moins le vrai nombre (respectivement
  48,48 %, 52,31 % et 47,39 % de ces sous-comptages). L'arbitrage
  d'attaques, le sustain et les NoteOff doivent donc être audités
  séparément.
- La corrélation à la décision n'est pas une preuve qu'augmenter
  artificiellement K corrige la transcription : la tête peut attribuer
  un compte correct aux mauvaises hauteurs.

**Parité S708 signée** confirmée dans S753 sur le protocole natif
CAL32 : Onset 74,263862 %, Exact-K 62,045541 %, Active Exact-K
56,276249 %, Offset 77,329193 %, Full Note 57,131931 %,
Frame Pitch 76,462635 %.

## S754 — Expérience pour isoler la tête K

- FIT80 seulement : quatre classificateurs K0..K6 indépendants
  (deux profondeurs × deux pondérations de classes).
- Entrée inférence strictement causale : 104 données acoustiques et
  postérieures NOTE au fragment courant et historique antérieur.
- Les quatre classificateurs proposent des K ; interpolation avec
  l'ancienne tête signée à 15 / 35 / 55 % pour 12 architectures
  sorties différentes.
- CAL32 : **les mêmes** six métriques après émission de flux MIDI
  complets et les mêmes règles de fermeture S707 que S708.
- Aucune promotion si régression d'une des six métriques, diminution
  TP Onset, perte d'un Offset précédemment correct ou d'un long Onset
  auparavant reconnu.
- HELD/TEST fermés, aucun changement LIVE, ni de S80/S707/S708, aucune
  distillation.

Run FIT80 + 12 CAL32 :
https://github.com/Andriamarosoa/note/actions/runs/39210823815

PR DRAFT :
https://github.com/Andriamarosoa/note/pull/85

La validation CI et les scores de S754 doivent être lus du rapport
`report.json` correspondant à ce run. **Aucun score n'est attribué
à S754 ici avant la fin de l'exécution.**

## Deux prochaines étapes conditionnelles

1. Si la tête K améliore K5/K6 mais fait régresser la note MIDI,
   auditer `GuidedSelector` (têtes Attack/Sustain et états actifs)
   sans remplacer les anciennes notes.
2. Si la tête K n'améliore pas la classification sur CAL32,
   revenir à la morphologie spectrale/fondamentales des accords dans
   des têtes acoustiques indépendantes et évaluer vrais K2-K6.

Cela constitue une investigation coordonnée menée dans cette
conversation, **pas six agents Work autonomes**.
