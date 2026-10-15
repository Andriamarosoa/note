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
