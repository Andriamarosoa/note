# S656 — Pourquoi le Tone aide seulement certaines superpositions

## Preuve mesurée

GitHub Actions réussi :
https://github.com/Andriamarosoa/note/actions/runs/39008024307

- `scripts/s656_tone_true_octave_pairs.py`
- `test/test_s656_tone_octave_pairs.py`
- `S656_GUITARSET_REAL_E3_E4_CAUSAL_TONE_OVERLAP/report.json`.

**8 paires de vraies prises GuitarSet TRAIN isolées**, E3
(MIDI52, ~164.8Hz) et E4 (MIDI64, ~329.6Hz), mixées
numériquement. La **même prise E3** sert de condition
single et de composante E3 de tous les accords.
E4 est une prise instrumentale distincte. Ces accords sont
**construits**, pas des morceaux naturellement enregistrés,
ni des tests Stratocaster.

Trois vues causalement filtrées à 44,1kHz :
original, première ordre `sosfilt` 2800Hz, première ordre
`sosfilt` 1100Hz ; aucun pitch shifting ni delay.
Le modèle S614 et ses têtes Exact-K sont strictement inchangés.

### Onsets ciblés (tolérance ±80ms)

| Événement attendu | Original | Ton doux | Ton fort |
|---|---:|---:|---:|
| E3 seul détecté | 8/8 | 8/8 | 8/8 |
| E4 seul détecté | 8/8 | 8/8 | 8/8 |
| E3 de E3+E4 simultané | **6/8** | **7/8** | **8/8** |
| E4 de E3+E4 simultané | 8/8 | 8/8 | 8/8 |
| E3 de E3 puis E4 +120ms | 8/8 | 8/8 | 8/8 |
| E4 ajouté à +120ms | **5/8** | **5/8** | **2/8** |

Donc le filtre Tone fort rétablit **2 vraies notes
graves E3** qui étaient masquées par une octave aiguë
simultanée, mais **fait perdre 3 attaques E4** lorsqu'elles
arrivent après l'attaque de E3.

Ce compromis est le mécanisme qu'il faut exploiter par
les **têtes et sélections contextuelles** :
la vue Tone forte peut être utile pour résoudre une
superposition qui attaque ensemble ; elle doit être
écartée lorsqu'une vraie nouvelle note arrive tardivement.

### Garder les faux positifs explicites

Sur les 8 mélanges E3+E4 simultanés :
- S614 : TP **14** /16 attendus, **61 FP**
  sur les segments 1,65s ;
- Tone léger : TP **15**, **58 FP** ;
- Tone fort : TP **16**, **53 FP**.

Les 53 faux NoteOn ne sont pas le score de vrais morceaux :
les segments contiennent des résidus d'autres notes hors
de la fenêtre de cible et une annotation construite
volontairement limitée. Il est donc interdit de prétendre
avoir atteint un F1 global de 100%. Même 16/16
est un **rappel sur deux hauteurs explicitement attendues**,
pas la précision totale de transcription.

### Architecture candidate suivante

**Ne jamais filtrer uniformément l'audio en LIVE.**

Une nouvelle sélection `coincident_octave_repair`
pourrait activer la vue Tone forte seulement si :
1. le modèle original émet une attaque de hauteur aiguë
   E4 ou comparable ;
2. sa fondamentale grave une octave plus bas est
   plausible dans la même fenêtre d'attaque
   (support spectral *indépendant* de l'harmonique
   commune et contexte multi-F0) ;
3. le sélecteur de note grave n'est ni déjà actif
   ni en période réfractaire ;
4. l'argument supplémentaire existe au **présent/past**
   uniquement, aucun samples futurs ni timestamp antidaté ;
5. le K natif/compétition ne bloque pas une véritable
   note de l'accord sans justification.
Les offsets de 120ms déclencheraient une autre tête,
`independent_late_attack`, et jamais la même règle
de filtrage radical.

La tête devra être apprise sur des exemples naturels
GuitarSet TRAIN avec leurs vrais faux positifs/attacks,
puis évaluée sur compositions indépendantes ;
ne pas entraîner sur les 8 paires construites pour
revendiquer l'amélioration de DEV.

**Verdict : piste spectrale utile pour cas simultanés,
pas encore de promotion ni de F1 autonome supérieur.
Le LIVE original demeure intact.**
