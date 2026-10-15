# S648 — Nouvelle tête *naissance de note* (NOTE LIVE causal)

## Constat qui motive la refonte

[S647](https://github.com/Andriamarosoa/note/actions/runs/38819248595)
a prouvé que la tête S642 `q` ne reconnaît pas proprement les
nouvelles attaques : parmi 17 204 fortes montées q GuitarSet DEV,
24 observations trame/hauteur coïncident avec une fenêtre de
vraie attaque ; sur URMP DEV, 1 sur 7 390. L'état actif de NOTE
n'est pas responsable de ces rejets. La tête S642 produit un score
d'état acoustique sans sémantique d'événement suffisamment alignée.

**Ne pas relâcher directement q ni recopier le décodeur MIDI.**

## Nouvelle solution S648

Deux sélecteurs indépendants :
- **ombre S614 originale** : interprète les sorties du modèle d'origine
  exactement comme auparavant, reçoit uniquement l'historique audio
  disponible, et fournit l'état actif, armé, âge de note, nombre de
  notes actives et compétition des hauteurs ;
- **sélecteur expérimental S648** : reçoit le même flux original S614,
  et n'admet *au plus une hauteur additionnelle par trame* si la
  nouvelle tête d'événement considère une attaque possible.
  Les règles originales de confirmation, réfractaire,
  restriction de polyphonie et NoteOff restent applicables.

La nouvelle tête n'est pas S642. Les caractéristiques sont construites
uniquement depuis les trames présentes et passées : 49×126 acoustique
natif, variation spectrale des bins sélectionnés, les trois probabilités
S614, leurs différences temporelles, l'état actuel du sélecteur ombre,
l'âge depuis la précédente NoteOn, concurrence en hauteurs voisines
et représentation de compte Exact-K native. Pas de audio future ni
d'annotations de référence pour l'inférence.

**Supervision TRAIN** : cible positive uniquement sur la première
trame observée à partir d'une vraie attaque annotée et sur la suivante
(maximum 2 hops), au lieu de considérer 40ms d'occupation comme une
succession de nouvelles attaques. Négatifs difficiles : autres hauteurs
durant une attaque polyphonique, notes réellement soutenues, faux
pics de l'ancien réseau, et arrière-plan aléatoire. Aucune correction
fondée directement sur le label DEV.

## Protocole d'apprentissage vérifiable

- 35 sources originales TRAIN : 17 URMP + 18 GuitarSet.
- Séparation TRAIN par **composition**, ordre SHA256 figé : 24
  enregistrements pour entraîner les poids et 11 pour calibrer
  la décision sans chevauchement de composition.
- Modèle entraîné : HistGradientBoostingClassifier CPU,
  graine 648, 105 itérations maximum, 16 feuilles maximum,
  poids positifs 3, régularisation L2=8.
- Stratégies de score prédéfinies **avant ouverture DEV** :
  seuils q = 0,70 / 0,80 / 0,85 / 0,90 / 0,95,
  avec hausse de score minimale prédéfinie 0,10 ou 0,15 ;
  garder une durée de 15 hops entre deux nouvelles attaques
  assistées de même hauteur, plafonner à 1 hauteur ajoutée
  par trame, sans retarder/antidater les émissions.
- La politique n'est verrouillée que sur les 11 CALIBRATION :
  F1 historique symétrique +80ms amélioré dans GuitarSet
  **et** URMP, au plus 8 FP supplémentaires par corpus,
  au plus une source en régression par corpus. Sinon conserver
  S614 original.
- DEV original **17 enregistrements** (12 GuitarSet, 5 URMP),
  compositions disjointes de TRAIN, ouvert seulement
  *après* verrouillage. Comparer TP/FP/FN, F1 historique,
  métrique causale [0,+80ms] sur MIDI 40–76, nombre de
  nouvelles NotesOn réellement émises, accords, NoteOff
  équilibrés et régressions par morceau.

**S80, S614, S632 restent inchangés.** Pas de test physique
microphone→MIDI, ni de score TEST nouveau. Même un gain DEV
ne suffit pas à promouvoir S648.

## Code et validation

- `src/causal_note/s648_event_birth.py` : cible de
  *naissance de note*, features causales, négatifs, décodage.
- `scripts/s648_train_birth_event.py` : préparation de
  TRAIN, entraînement, verrouillage CALIBRATION, DEV.
- `test/test_s648_event_birth.py` : no-op exact S614,
  invariance aux trames futures, une nouvelle note par trame,
  tests ciblés avec faux positifs.
- [GitHub Actions S648](https://github.com/Andriamarosoa/note/actions/workflows/v273-s648-actual-train-birth-head.yml).

**Résultats mesurés à ajouter uniquement après la fin réussie
de l'entraînement et des évaluations.** Toute promotion exclue
sans test indépendant et contrôle hardware LIVE.


## S649 — Parité STRICT LIVE validée

[GitHub Actions S649 : tests réussis](https://github.com/Andriamarosoa/note/actions/runs/38821458185).

`src/causal_note/s649_birth_live_runtime.py` contient deux
sélecteurs temporels strictement causaux (ombre S614 et variante
S648) : chacun reçoit uniquement l'audio déjà arrivé.
Chaque bloc PCM de 256 échantillons, 44,1kHz,
fournit les sorties natives et le vecteur événementiel
de l'instant. Les événements ne sont jamais antidatés ;
la fermeture produit NoteOff pour toutes les notes
actives.

`test/test_s649_birth_live_runtime.py` vérifie que le
vrai checkpoint S614 produit les mêmes événements
que le mode batch causal pour les politiques original,
0,70/0,10 et 0,85/0,15 sur PCM identique, et que
modifier l'audio futur ne peut changer les événements
des premiers blocs. Ces tests utilisent une tête de
score synthétique pour la preuve logicielle : ils
**ne valident pas encore les poids entraînés S648**
ni la vitesse matérielle Windows.

## S650 — Preuve future des effets sur les sources gelées

`scripts/s650_audit_event_birth_deltas.py` reprend uniquement
le rapport S648 (une fois terminé) et prouve que TP/FN
évoluent par symétrie sur les vérités figées, puis
extrait les changements TP/FP/FN et nombres de notes
assistées par morceau, corpus et politique, avec
vérification des agrégats de la CI initiale.

Aucun paramètre ne peut être réoptimisé par S650.


## S648 TRAIN/CALIBRATION/DEV — vrai résultat vérifié

[Exécution S648 réussie et reproductible](https://github.com/Andriamarosoa/note/actions/runs/38821104076)
[Checkpoint S648 + rapports JSON](https://github.com/Andriamarosoa/note/actions/runs/38821104076/artifacts/12013881637).

La tête S648 a été **réellement entraînée** sur 186 196
exemples acoustiques (dont **6 196 trames positives**)
provenant des **24 sources FIT**.
SHA256 du checkpoint appris :
`7613dca4aa9bc5992c96a75ab55f714d8720c55b6880ee9e223e39ea880eb4e9`.

Le verrouillage CALIBRATION utilise les 11 autres TRAIN
de compositions distinctes (aucun échantillon DEV au
moment de choisir). Les cinq variantes prédéfinies
échouent à améliorer simultanément les deux corpus
sans trop de faux positifs : **S614 original a été
verrouillé** avant ouverture de DEV.

### 11 CALIBRATION (métrique d'origine NoteOn ±80ms)

| Politique | GuitarSet TP/FP/FN | GuitarSet F1 | URMP TP/FP/FN | URMP F1 |
|---|---|---:|---|---:|
| S614 original | 844/360/350 | **70,39 %** | 401/149/146 | **73,11 %** |
| q≥0,70 / hausse 0,10 | 867/791/327 | 60,80 % | 410/398/137 | 60,52 % |
| q≥0,80 / hausse 0,10 | 850/475/344 | 67,49 % | 402/170/145 | 71,85 % |
| q≥0,85 / hausse 0,15 | 845/397/349 | 69,38 % | 401/149/146 | 73,11 % |
| q≥0,90 / hausse 0,15 | 844/366/350 | 70,22 % | 401/149/146 | 73,11 % |
| q≥0,95 / hausse 0,15 | 844/360/350 | 70,39 % | 401/149/146 | 73,11 % |

### 17 DEV, mesurés après le verrouillage

| Politique exploratoire | GuitarSet TP/FP/FN | GuitarSet F1 | URMP TP/FP/FN | URMP F1 |
|---|---|---:|---|---:|
| **S614 verrouillé** | 998/585/582 | **63,10 %** | 243/149/256 | **54,55 %** |
| q≥0,70 / hausse 0,10 | 1070/1627/510 | 50,04 % | 262/302/237 | 49,29 % |
| q≥0,80 / hausse 0,10 | 1018/895/562 | 58,29 % | 248/160/251 | 54,69 % |
| q≥0,85 / hausse 0,15 | 1006/699/574 | 61,25 % | 243/149/256 | 54,55 % |
| q≥0,90 / hausse 0,15 | 1000/600/580 | 62,89 % | 243/149/256 | 54,55 % |
| q≥0,95 / hausse 0,15 | 998/585/582 | 63,10 % | 243/149/256 | 54,55 % |

Cette fois la tête d'événement émet des nouveaux NoteOn
(non comme S642 où les portes les bloquaient totalement),
mais **la sélectivité est mauvaise**. Exemple q≥0,85 :
260 NoteOn ont été assistés sur GuitarSet DEV,
mais la comparaison à S614 donne seulement **+8 TP nets**
contre **+114 FP nets**, avec **11 régressions / 12
enregistrements** ; sur URMP DEV aucun changement.
Le risque est plus grave à q≥0,70 : +72 TP mais
+1 042 FP nets GuitarSet DEV.

## S650 — Audit indépendant du rapport figé S648

[GitHub Actions S650 : audit réussi](https://github.com/Andriamarosoa/note/actions/runs/38822366278),
[rapport S650 par source et politique](https://github.com/Andriamarosoa/note/actions/runs/38822366278/artifacts/12013557740).

L'audit est recalculé **sans toucher aux poids, à l'audio
ni aux seuils** ; il vérifie les invariants TP/FN et que
les agrégats source→corpus correspondent exactement
au rapport S648.

Sur GuitarSet DEV :
- q≥0,70 : 0/12 sources améliorées, 12/12 régressées ;
- q≥0,80 : 1/12 améliorée, 11/12 régressées ;
- q≥0,85 : 1/12 améliorée, 11/12 régressées ;
- q≥0,90 : 2/12 améliorées, 5/12 régressées,
  5 inchangées.

La référence verrouillée S614 conserve **100 % de
ses sorties originales** : 998 TP/585 FP/582 FN GuitarSet,
243 TP/149 FP/256 FN URMP.

## Décision technique

**S648 refusée comme référence, tout en conservant ses
enseignements et son code.** S649 vérifie la causalité
logicielle de l'architecture mais pas encore le F1
ni le p95 matériel du checkpoint réel en microphone LIVE.
Le score officiel de S614 reste **69,03 % F1 sur son
historique TEST symétrique** : les chiffres ci-dessus
sont des métriques de CALIBRATION/DEV, pas un nouveau TEST.

Défaut résiduel : les vrais candidats faibles sont
dominés par des faux événements sonores et des confusions
de hauteurs. Baisser q seul dégrade plus la précision
que cela ne récupère de notes.

Prochaine itération méthodologiquement justifiée :
**audit événementiel des candidats assistés**
avec distinction TP nouvellement récupérés /
TP historiques déplacés / nouveaux FP / réattaques
harmoniques et temps depuis la dernière attaque.
Puis développer une nouvelle sélection à coût
FP explicitement appris, sur des erreurs
*hors entraînement* par composition, avant
tout réentraînement majeur ou promotion.
