# Expérience 06 — candidats onset/offset sans association

État figé le 31 août 2026. L'expérience est terminée et reste strictement
diagnostique. Le flux de candidats V7 non associés n'est pas activé en live.
La baseline officielle reste V7-e8 avec le décodeur associé et les seuils
onset/offset `0.55`.

## Question isolée

Le modèle contient-il des pics onset/offset utiles que le décodeur actuel
supprime seulement parce qu'un événement occupe encore le même canal interne ?

Le contrôle conserve le `LiveBoundaryScoreDecoder` officiel. Le traitement
reçoit exactement les mêmes objets de scores, mais émet chaque front montant :

```text
type, position
```

Il ne produit ni `eventId`, ni corde, ni slot, ni hauteur, ni association. Un
offset peut être émis sans onset ouvert et un nouvel onset peut être émis avant
l'offset précédent. Deux canaux qui franchissent le seuil au même échantillon
produisent deux candidats afin de conserver la multiplicité.

## Audit des données avant implémentation

Une tête globale binaire obtenue par un simple `OR` des six cordes a été
explicitement écartée avant le code :

- sur les 9 541 onsets de validation, les fenêtres globales de 512 ne
  conserveraient que 6 726 fronts séparables : 2 815, soit 29,50 %, seraient
  fusionnés ;
- 45 onsets supplémentaires arrivent exactement au même échantillon qu'un
  autre onset ; un seul bit ne peut pas représenter leur multiplicité ;
- 6 409/9 541 onsets de validation, soit 67,17 %, arrivent pendant qu'au moins
  une note plus ancienne reste ouverte ;
- dans le split d'entraînement, une cible globale binaire de largeur 512 aurait
  une densité de 5,7989 % pour onset et 6,1859 % pour offset ;
- avec le poids positif 28, les optima BCE constants deviendraient `0.632844`
  et `0.648661`, tous deux au-dessus du seuil live `0.55` ;
- 82 offsets de validation sont exactement à la fin du fichier audio : ils
  restent des références d'évaluation, mais aucune fenêtre d'apprentissage ne
  peut commencer après eux. Ce défaut séparé n'est pas corrigé ici.

Il n'y a donc eu ni fusion globale des cibles, ni entraînement, ni changement
de loss, de poids ou de sampler dans cette expérience.

## Intégrité du calcul

- checkpoint V7-e8 inchangé, SHA-256
  `5634ADD0E112A6889B65D5245AD051AD850A1FFFE66FEB8D9E5E74472BA114BF` ;
- mêmes 60 pistes GuitarSet, joueurs `00` à `04`, graine `1337` ;
- joueur `05` non lu ;
- 9 541 onsets et 9 541 offsets de référence ;
- 1 880,8998 secondes d'audio et 162 030 blocs de 512 ;
- une seule inférence par bloc ; chaque `BoundaryScoreChunk` est remis au
  contrôle puis au traitement ;
- le contrôle reproduit exactement le sweep V7 à `0.55`, globalement et sur
  chacune des 60 pistes ;
- aucune métrique d'intervalle, aucun scheduler et aucune activation live ;
- 133/133 tests réussis avec l'environnement TensorFlow réel.

## Résultats onset/offset

| Mesure | Contrôle associé | Candidats sans association | Évolution |
|---|---:|---:|---:|
| Onsets prédits | 111 169 | 401 104 | +289 935 |
| TP onset | 8 856 | 9 155 | +299 |
| FP onset | 102 313 | 391 949 | +289 636 |
| FN onset | 685 | 386 | -299 |
| Précision onset | 7,9662 % | 2,2825 % | -5,6838 points |
| Rappel onset | 92,8205 % | 95,9543 % | +3,1338 points |
| F1 onset | 14,6732 % | 4,4588 % | **-10,2143 points** |
| Offsets prédits | 110 987 | 658 417 | +547 430 |
| TP offset | 7 980 | 9 029 | +1 049 |
| FP offset | 103 007 | 649 388 | +546 381 |
| FN offset | 1 561 | 512 | -1 049 |
| Précision offset | 7,1900 % | 1,3713 % | -5,8187 points |
| Rappel offset | 83,6390 % | 94,6337 % | +10,9947 points |
| F1 offset | 13,2417 % | 2,7035 % | **-10,5383 points** |

Le nombre de prédictions passe de 11,65 à 42,04 fois le nombre de références
pour onset, et de 11,63 à 69,01 fois pour offset.

Le résultat est uniforme :

- F1 onset inférieur sur 60/60 pistes et 12/12 groupes famille–arrangement ;
- F1 offset inférieur sur 60/60 pistes et 12/12 groupes famille–arrangement ;
- TP onset supérieur sur 49 pistes et identique sur 11, jamais inférieur ;
- TP offset supérieur sur 59 pistes et identique sur une, jamais inférieur ;
- FP onset et offset supérieurs sur 60/60 pistes.

| Régime | F1 onset contrôle | F1 onset candidat | F1 offset contrôle | F1 offset candidat |
|---|---:|---:|---:|---:|
| Global | 14,6732 % | 4,4588 % | 13,2417 % | 2,7035 % |
| Comp | 13,5895 % | 4,1952 % | 12,1445 % | 2,5572 % |
| Solo | 19,1691 % | 5,4485 % | 17,8124 % | 3,2515 % |

## Temps et multiplicité

Les candidats vrais sont mieux localisés :

- médiane signée onset : `-3,016 ms` vers `-0,476 ms` ;
- médiane signée offset : `-1,610 ms` vers `-0,045 ms` ;
- médiane causale onset : `3,968 ms` vers `1,610 ms` ;
- médiane causale offset : `6,927 ms` vers `1,610 ms`.

Le traitement conserve techniquement plusieurs candidats simultanés, mais V7
en produit beaucoup trop : la référence ne contient que 45 onsets
simultanés supplémentaires, contre 4 824 pour le contrôle et 34 177 pour le
traitement. Même après regroupement des doublons au même échantillon, le
traitement possède encore 366 927 positions onset uniques pour seulement 9 496
positions de référence.

## Interprétation de conception

Le blocage par événement ouvert cachait bien des vraies frontières : le
traitement récupère 299 correspondances onset et 1 049 correspondances offset.
Mais il servait aussi de filtre massif. En le supprimant, 289 636 faux onsets
et 546 381 faux offsets supplémentaires apparaissent.

Les données établissent donc simultanément que :

1. un onset ne doit pas dépendre de la fermeture d'un événement précédent ;
2. les scores V7 bruts ne constituent pas encore un détecteur de types autonome
   utilisable à `0.55` ;
3. les six canaux réagissent souvent plusieurs fois ou simultanément au même
   signal ;
4. l'association actuelle masque le bruit au lieu de seulement associer les
   frontières.

Le test contrôlé valide bien la règle demandée :

```text
onset 100
onset 700
offset 820
```

Mais cette règle seule ne suffit pas sur les données réelles. Avant toute
association par `eventId`, il faut un détecteur de types qui conserve les pics
proches et simultanés tout en supprimant les répétitions. La solution ne peut
pas recopier la cible globale binaire de largeur 512 avec le poids 28, car
l'audit prédit alors une sortie constamment supérieure au seuil.

## Décision

- le nouveau décodeur reste un outil diagnostique ;
- il n'est pas branché dans `detect_live.py`, le pipeline ou le scheduler ;
- aucun `eventId` n'est fabriqué dans ce traitement ;
- V7-e8 associé à `0.55/0.55` reste la baseline officielle ;
- l'expérience s'arrête ici avant de choisir ou d'entraîner la nouvelle
  représentation sans corde et avant l'étape d'association.

## Artefacts

- protocole :
  `model/causal-boundaries-weight28-window512-v7-epoch08.unassociated-boundary-candidates-protocol.json`,
  SHA-256
  `F945620AE419381B1FA1EFB078FD1A305CCA0E7BFD5F75ED384D097FBB171688` ;
- résultat :
  `model/causal-boundaries-weight28-window512-v7-epoch08.unassociated-boundary-candidates.json`,
  SHA-256
  `3725ABDB75383AC51C2B800C1232FDAE9A3E5F271F8D3D40743C498AAA07AB81` ;
- évaluateur : `scripts/evaluate_boundary_candidates.py`, SHA-256
  `A9C5A02361495C6EC0C1EBCAA54D068EADEE7DF44585AA816718C00F7F75C698`.
