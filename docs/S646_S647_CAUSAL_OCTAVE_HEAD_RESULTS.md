# S646–S647 — tête d'octave branchée aux *vrais* candidats NoteOn de NOTE

## Verdict du cycle

**Aucune promotion.** S646 et S647 sont les premières évaluations
comparables de cette idée comme **véritables événements MIDI NoteOn causaux**
produits par une seconde instance du sélecteur NOTE, et non comme une
classification offline à partir de la vérité terrain.

Le meilleur F1 NOTE officiel S614 reste **63,104647%**
(TP 998, FP 585, FN 582) sur les 12 pistes GuitarSet DEV de
284,294 secondes. Le meilleur résultat **expérimental** de la
sélection S641 reste **63,399309%**, toujours non promu.

### Références auditées
- S646 : https://github.com/Andriamarosoa/note/actions/runs/38999448394
  artefact `S646_NATIVE_NOTE_OCTAVE_RESCUE_CAUSAL/report.json`
- S647 : https://github.com/Andriamarosoa/note/actions/runs/39000029948
  artefact `S647_NATIVE_NOTE_OCTAVE_RESCUE_CAUSAL/report.json`
- Les scripts `scripts/s646_native_note_octave_head.py` et
  `scripts/s647_high_to_low_octave_head.py` ont des tests unitaires
  de vrai temps causal, horodatage, labels et immutabilité du modèle.

## Pourquoi S645 ne pouvait pas encore être intégré tel quel

S645 avait produit 62,30% de **F1 binaire note seule vs octave**
(19 TP, 14 FP, 9 FN parmi 28 duos réels). Ce n'était pas un F1
de transcription NOTE : le modèle recevait l'attaque exacte de la
note sous supervision annotée et pouvait lire jusqu'à 190ms de
signal après l'attaque.

S646–S647 suppriment cet oracle au moment de l'inférence :
le seul déclencheur est une **attaque émise par le décodeur original**.
Les probas de hauteur, la prédiction K, la logique de sélection et le
checkpoint original restent identiques. Une tête supplémentaire peut
seulement injecter une proposition de pitch dans une **copie**
`CausalSelection`, à la date de décision réelle.

Les décalages sont arrondis au hop causal de 256 échantillons :
- 0ms : émission au même hop que l'attaque source ;
- 40ms demandé : 40,635ms physiquement ;
- 80ms demandé : 81,270ms physiquement ;
- 190ms demandé : 191,565ms physiquement.
Aucune sortie n'est antidatée. Ces durées ne prouvent pas la latence CPU
réelle ni la compatibilité microphone temps réel.

## S646 — déclenchement note grave -> octave aiguë

36 pistes GuitarSet TRAIN, 9 compositions distinctes ; labels en
apprentissage seulement ; CV sur compositions entières.
- 2814 NOTE low candidates (0ms), dont 183 avec vraie octave
  supérieure simultanée annotée.
- La CV TRAIN choisit **abstention totale à 0ms** : confiance insuffisante
  pour ajouter une octave.
- A 40/80ms seuil 0,98, à 190ms seuil 0,95.

| Modèle | TP | FP | FN | F1 | Gain F1 points |
|---|---:|---:|---:|---:|---:|
| S614 original | 998 | 585 | 582 | 63,104647% | 0 |
| S646 0ms | 998 | 585 | 582 | 63,104647% | 0 |
| S646 40ms | 998 | 587 | 582 | 63,064771% | -0,039877 |
| S646 80ms | 999 | 587 | 581 | **63,108023%** | **+0,003375** |
| S646 190ms | 999 | 609 | 581 | 62,672522% | -0,432126 |

S646 80ms = +1 TP, +2 FP, sans perdre de TP initial. Sa
minuscule hausse F1 n'est pas une amélioration robuste et est bien
inférieure à S641.

## S647 — déclenchement note aiguë -> octave grave masquée

Correction d'une erreur de direction expérimentale : S642 montrait
sur E3 vs E3+E4 que E4 était souvent retrouvée, alors que E3 grave
pouvait disparaître de l'accord. S647 essaye de retrouver ce grave
à partir d'une attaque aiguë NOTE déjà émise.

- 4229 NOTE high candidates (0ms) sur TRAIN GuitarSet réel,
  dont 382 avec une vraie note inférieure une octave sous
  l'attaque haute.
- CV TRAIN (5 plis par compositions) : seuil 0,85 à 0/40/80ms,
  seuil 0,90 à 190ms.
- Précision OOF du classifieur pour 0ms : **42,36%**
  (169 duos corrects / 230 faux duos prédits) ; le diagnostic
  a donc déjà un risque structurel de faux positifs.

| Modèle | TP | FP | FN | F1 | Gain F1 points |
|---|---:|---:|---:|---:|---:|
| S614 original | 998 | 585 | 582 | **63,104647%** | 0 |
| S647 0ms | 1000 | 594 | 580 | 63,011972% | -0,092675 |
| S647 40ms | 1000 | 593 | 580 | 63,031831% | -0,072816 |
| S647 80ms | 998 | 619 | 582 | 62,433531% | -0,671116 |
| S647 190ms | 1009 | 617 | 571 | 62,944479% | -0,160168 |

Le gain instantané 0ms : **+2 TP pour +9 FP**, insuffisant.
Il faudrait éliminer au moins cinq nouveaux faux positifs en
conservant les deux bonnes corrections pour dépasser S614.

Le cas 190ms : **+11 événements appariés mais +32 FP**.
Ce résultat ne prouve PAS que les onze corrections sont toutes
les fondamentales appartenant à l'accord de départ :
une injection retardée peut coïncider avec une attaque ultérieure
différente. Il faut auditer les associations attaque-source →
proposition → annotation cible avant de revendiquer onze vraies
corrections d'accord.

### Répartition DEV S647 des corrections/FP 0ms

- `04_Rock1-90-C#_comp` : **+2 TP / +1 FP**.
- Plusieurs autres pistes subissent de petits gains de FP
  sans aucun TP, notamment `Funk3-98-A_comp` (+3 FP).
- Les deux corrections viennent donc **d'un seul enregistrement** :
  elles ne démontrent pas la robustesse inter-morceaux.

### Décision de conception

**L'octave physique est utile comme feature, pas comme décision
autonome universelle.** Une simple régression logistique binaire sur
le spectre ne présente pas assez de précision pour insérer des NoteOn
dans des accords complexes.

Le prochain correcteur devra tenir compte conjointement :
1. de la probabilité d'attaque **native propre à la note grave** et
   de sa variation, non seulement de la présence de sa fondamentale ;
2. des deux hypothèses concurrentes note seule / accord réel ;
3. de l'état armé/actif et des candidats déjà émis (anti-doublon) ;
4. du K probabiliste, mais sans activer le veto K global qui avait
   régressé dans S639 ;
5. des familles d'erreurs observées en S639 : attaque croisée,
   sustain/réattaque, confusion octave et harmonique ;
6. d'une **sélection avec abstention**, entraînée sur les corrections
   et régressions de NoteOn réels, pas uniquement sur la vérité binaire
   « deux notes sont actives ».

N'utiliser ni GuitarSet TEST ni les fichiers IDMT/URMP TEST pour
optimiser la règle. Après apprentissage TRAIN et contrôle DEV,
figer le modèle avant une validation externe indépendante.

Aucun changement du réseau NOTE/Exact-K de référence, de S641 ou
du script de microphone LIVE n'a été effectué.
