
## Résultats vérifiés — run S718 du 18/10/2026

[Run S718 (succès)](https://github.com/Andriamarosoa/note/actions/runs/39111634032) : **16 tests réussis**, replay des 44 pistes (32 HELD et 12 DEV), aucun changement de la chaîne source, six métriques S708 reproduites, `DEV TP=1055 FP=434 FN=525`, `HELD FN=834`.

### Diagnostic de 525 faux négatifs DEV

| Catégorie exclusive | DEV | HELD |
|---|---:|---:|
| `insufficient_joint_neural_evidence` | **321** | **445** |
| `network_onset_below_weak_threshold` | **106** | **147** |
| `high_onset_but_sustain_or_state_insufficient` | 29 | 50 |
| `weak_evidence_blocked_by_K_gate` | 19 | 75 |
| `source_predicted_but_match_conflict` | 19 | 12 |
| `weak_acoustic_rearmed_or_refractory` | 21 | 50 |
| `weak_attack_without_rising_evidence` | 6 | 46 |
| `selector_rearming_blocked` | **3** | **3** |
| `weak_evidence_not_emitted` | 1 | 5 |
| `source_onset_removed_downstream` | 0 | 1 |
| **Total** | **525** | **834** |

Les deux premières catégories regroupent **427/525=81,33%** DEV,
et **592/834=70,98%** HELD. Elles indiquent l'absence d'une
combinaison d'indices neuronaux exploitable **dans la fenêtre ±80ms**,
pas nécessairement la destruction de la fondamentale physique dans
le signal. **Supprimer le verrou `armed` ne corrigerait directement
que 3/525 (0,57%) des attaques classées sur DEV.**

Les drapeaux **non exclusifs** (ne pas additionner aux catégories)
montrent DEV : 57 cas avec `onset >= .75` à un moment,
22 cas avec Onset **et** Sustain au seuil simultanément,
19 avec preuve d'éligibilité forte au moins sur un bloc,
et 35 avec preuve faible réarmée. Ces diagnostics expliquent
pourquoi abaisser un seul seuil peut échanger de nombreux FN contre
des FP supplémentaires et dégrader les autres scores.

Répartition des FN par registre DEV (nombre de vraies attaques manquées) :
MIDI 40–52 : **191** ; MIDI 53–64 : **265** ;
MIDI 65–88 : **69**.
Ces nombres ne sont pas des taux par registre et ne doivent pas être
comparés directement aux proportions de frames-note de S714.

### Verdict prioritaire

1. **Cible principale (P1)** : améliorer la **tête acoustique
   Onset/Sustain conjointe** sur notes manquantes en polyphonie,
   particulièrement graves et médiums. Plan FIT/CAL : nouveaux
   contrastes de fondamentales indépendantes et reweight accords
   + onset faibles, avec comparaison mêmes morceaux.
2. **Cible complémentaire (P2)** : expert K pour les 19
   attaques refusées par la branche faible sur DEV, sans dure
   contrainte K (la tête K sous-estime déjà les accords).
3. **Cible secondaire (P3)** : l'armement fort seul (3 cas DEV).
   S719 le teste uniquement en shadow à titre de contrôle, et
   aucune promotion ne sera faite si les six scores régressent.
4. **Invariant** : ne pas dégrader les 1055 TP existants,
   ne pas augmenter aveuglément les 434 FP, maintenir S707 Offset
   et Exact-K; tous les gains exigent CAL puis HELD/DEV,
   et un nouveau jeu externe indépendant pour promotion.

**Attention** : l'analyse de seuils et de postérieurs n'est pas une
preuve étiologique unique ; certains FN peuvent avoir plusieurs
blocages et la catégorie exclusive reflète une priorité de règles.

---

# S718 — Audit des 525 attaques manquées, sélection versus têtes neuronales

## Objectif
Identifier la catégorie de chacun des **525 FN Onset S708 DEV** et des
**834 FN S708 HELD** sur les vrais enregistrements GuitarSet, sans retoucher
les règles. Une fausse attaque est comptée avec le protocole officiel
`event_score` : même hauteur MIDI et disponibilité d'événement à ±80 ms.
Le chiffre final vient des *NoteOn réellement émis* après S707.

## Instrumentation reproductible
`scripts/s718_onset_false_negative_audit.py` recharge le vrai checkpoint
`s708_selected.pt`, la tête S613 et le S675 hazard original. Il calcule
les prédictions S708 depuis le PCM **avant d'ouvrir les annotations**.
À chaque bloc causal de 1024 échantillons, il enregistre pour chaque pitch :
onset, sustain, release, armement, note déjà active, temps depuis dernière
attaque, tendance, confiance K, K cible et origine des NoteOn émis.
La liste d'événements instrumentée doit être **identique** au décodage
historique S708. La sortie S707 offline, les six métriques et les compteurs
TP/FP/FN sont reconstruits et comparés aux valeurs figées GitHub.

Quand les annotations sont ensuite ouvertes uniquement pour l'audit, chaque
FN examine les blocs admissibles à ±80 ms, sans les utiliser pour réévaluer
la sélection. Les catégories sont **diagnostiques**, exclusives et ne prouvent
pas une cause physique ; les drapeaux transversaux peuvent se recouper.

### Catégories à confirmer expérimentalement
- `source_onset_removed_downstream` : Source S708 avait un bon NoteOn
  apparié mais le plan S707 l'a perdu.
- `selector_rearming_blocked` : score Onset+Sustain assez fort,
  mais aucune possibilité de réarmer la même hauteur.
- `selector_refractory_65ms` : attaque forte dans le délai natif.
- `full_six_note_capacity` : preuve de seuils correcte mais six notes déjà
  actives dans le sélecteur.
- `weak_evidence_blocked_by_K_gate` : preuve acoustique faible admissible
  mais le K estimé est insuffisant.
- `weak_attack_without_rising_evidence` : indices faible-onset/sustain
  présents mais pas de montée détectée.
- `high_onset_but_sustain_or_state_insufficient` :
  forte tête Onset sans Sustain ou autres conditions simultanées.
- `network_onset_below_weak_threshold` : tous les scores Onset
  admissibles sont sous 0.15.
- Les cas `source_predicted_but_match_conflict`,
  `weak_evidence_not_emitted`, `weak_acoustic_rearmed_or_refractory`,
  `eligible_strong_not_emitted`, `insufficient_joint_neural_evidence`
  sont enregistrés au lieu d'inventer une causalité.

## Sécurité scientifique
- Jamais de label utilisé dans la sélection ou le calcul du modèle.
- Protocole GuitarSet strictement figé S708, HELD/DEV précédemment exposés,
  pas un nouveau TEST indépendant.
- Pas de correction ou hyperparamètre choisi sur DEV.
- Si le problème vient du réseau plutôt que de la sélection, ablation
  d'une nouvelle tête entraînée seulement sur FIT, seuils sélectionnés
  seulement sur CAL.
- Si le problème vient de `armed` ou de la période réfractaire,
  expérimenter une tête de réattaque en shadow avec épreuve de séparation
  harmonique, déduplication et protection des six scores, au lieu de
  supprimer ces gardes.

## Runs / provenance
- [S708 source](https://github.com/Andriamarosoa/note/actions/runs/39094682419)
- [S718 audit](https://github.com/Andriamarosoa/note/actions/workflows/note-s718-onset-fn-forensics.yml)
- Données archivées : `report.json`, `held_missing.jsonl`,
  `dev_missing.jsonl`, avec motifs et probabilités extraites pour
  **chaque FN**, pour empêcher qu'un gain local masque des régressions.
- **Aucune modification aux checkpoints, architecture LIVE officielle ou
  à la sélection S708 originale.**
