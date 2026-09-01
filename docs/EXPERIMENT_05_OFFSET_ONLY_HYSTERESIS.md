# Expérience 05 — hystérésis offset uniquement

État figé le 31 août 2026. Le traitement est **rejeté** selon la règle
préenregistrée. La baseline live reste V7-e8 avec entrée et réarmement à
`0.55` pour onset et offset.

## Question isolée

Le modèle, les données et le seuil d'entrée restent identiques. Le contrôle
utilise :

```text
onset : entrée 0.55, réarmement 0.55
offset: entrée 0.55, réarmement 0.55
```

Le traitement ne modifie qu'un seul paramètre :

```text
onset : entrée 0.55, réarmement 0.55
offset: entrée 0.55, réarmement 0.50
```

Le protocole a été préenregistré avant l'évaluation dans
`model/causal-boundaries-weight28-window512-v7-epoch08.offset-only-hysteresis-055-050-protocol.json`.

## Intégrité et audit des données

- checkpoint V7-e8 inchangé, SHA-256
  `5634ADD0E112A6889B65D5245AD051AD850A1FFFE66FEB8D9E5E74472BA114BF` ;
- GuitarSet, mêmes 60 pistes de validation, joueurs `00` à `04`, graine
  `1337`, blocs causaux de 512 échantillons ;
- joueur `05` non lu ;
- 9 541 intervalles de référence et 1 880,90 secondes d'audio inchangés ;
- une seule prédiction du modèle par bloc, remise aux deux décodeurs ;
- contrôle reproduit exactement l'ancien rapport à `0.55`, globalement et
  piste par piste ;
- morphologie traitement : 0 creux onset ponté et 533 338 creux offset
  pontés entre `0.50` et `0.55`.

Il n'y a ni entraînement, ni transformation de cible, ni modification de
densité, de poids, de sampler ou de loss dans cette expérience.

## Résultats live

| Mesure | Contrôle | Offset-only | Évolution |
|---|---:|---:|---:|
| Onsets / identifiants ouverts | 111 169 | 60 822 | -50 347 |
| Événements complets | 110 987 | 60 621 | -50 366 |
| Événements encore ouverts | 182 | 201 | **+19** |
| TP onset | 8 856 | 8 757 | -99 |
| FP onset | 102 313 | 52 065 | -50 248 |
| F1 onset | 14,6732 % | 24,8909 % | +10,2177 points |
| TP offset | 7 980 | 7 764 | -216 |
| FP offset | 103 007 | 52 857 | -50 150 |
| F1 offset | 13,2417 % | 22,1316 % | +8,8899 points |
| TP intervalle associé | 3 488 | 3 620 | +132 |
| FP intervalle associé | 107 499 | 57 001 | -50 498 |
| Précision intervalle | 3,1427 % | 5,9715 % | +2,8288 points |
| Rappel intervalle | 36,5580 % | 37,9415 % | +1,3835 point |
| F1 intervalle | 5,7879 % | 10,3190 % | +4,5311 points |

Le F1 intervalle augmente sur 60/60 pistes et sur les 12/12 groupes
famille–arrangement.

| Régime | F1 contrôle | F1 traitement | Ouverts contrôle | Ouverts traitement |
|---|---:|---:|---:|---:|
| Comp | 3,9981 % | 7,6386 % | 70 | 86 |
| Solo | 13,2436 % | 19,1549 % | 112 | 115 |

## Réponse événementielle exacte

La trace compare l'identité interne
`(piste, slot, position onset exacte)` et reproduit les deux séquences du
`LiveBoundaryScoreDecoder`. Elle valide aussi chaque piste et les comptes
globaux contre le sweep source.

| Sort d'une fermeture du contrôle | Nombre |
|---|---:|
| Même événement fermé au même échantillon | 33 825 |
| Même événement : fermeture supprimée puis récupérée | 26 796 |
| Même événement : fermeture supprimée, toujours ouvert à la fin | **39** |
| Identité déjà divergente : cascade non attribuée | 50 327 |
| Total des fermetures contrôle | 110 987 |

Ainsi :

- 26 835 fermetures du même événement sont supprimées à l'instant attendu ;
- cela représente 24,1785 % de toutes les fermetures contrôle, ou 44,2384 %
  des 60 660 fermetures encore directement comparables par identité ;
- 26 796/26 835, soit 99,8547 %, sont récupérées plus tard ;
- 39/26 835, soit 0,1453 %, laissent réellement l'événement ouvert jusqu'à la
  fin de la piste ;
- parmi ces 39 cas, 13 ne se réarment jamais sous `0.50` et 26 se réarment,
  mais aucun nouveau front offset à `0.55` n'arrive ensuite.

Pour les fermetures récupérées, le retard médian vaut 16,16 ms, le p90
102,91 ms et le maximum 25,36 s. Les 39 scores au front supprimé sont très
proches du seuil : minimum `0.550008`, médiane `0.551473`, maximum `0.557509`.

Les suppressions touchent les 60 pistes ; 21 pistes possèdent au moins un cas
encore ouvert à la fin. Le comp concentre 23 239 suppressions et 33 cas
permanents, contre 3 596 et 6 en solo. La famille SS1 concentre 13 des 39 cas
permanents et `+9` des `+19` événements ouverts nets.

## Interprétation de conception

Le latch onset est strictement inchangé, mais les onsets effectivement émis
passent tout de même de 111 169 à 60 822. Ce n'est pas une action directe de
l'hystérésis onset : un offset retardé garde le slot occupé, ce qui bloque les
onsets suivants dans ce slot et provoque une divergence en cascade. La trace
en compte 50 327 côté contrôle après perte de comparabilité d'identité.

Les données montrent donc deux effets différents :

1. l'hystérésis offset filtre beaucoup de faux fronts et améliore fortement le
   F1 ;
2. appliquée dans le décodeur associatif actuel, elle modifie aussi la suite
   des onsets publics par occupation du slot et augmente les événements non
   fermés.

Le `+19` n'est pas le nombre d'offsets bloqués. C'est seulement le bilan
`201 - 182`. Le nombre causal directement observé est 26 835 fermetures
supprimées, dont 39 restent ouvertes.

## Décision préenregistrée

Dix critères sur treize passent. Les trois échecs sont exactement :

- global : 201 événements ouverts, maximum autorisé 182 ;
- comp : 86, maximum autorisé 70 ;
- solo : 115, maximum autorisé 112.

Le traitement `offset release = 0.50` est donc rejeté. Aucun seuil
supplémentaire n'est testé dans cette expérience et la baseline reste
`onset release = 0.55`, `offset release = 0.55`.

## Artefacts et vérifications

- sweep :
  `model/causal-boundaries-weight28-window512-v7-epoch08.offset-only-hysteresis-entry055-onsetrelease055-offsetrelease055-050.json`,
  SHA-256
  `87EBBB9D2093F7276154062BA00AB1DCED83D45014356D62DA792D698E1B1067` ;
- trace :
  `model/causal-boundaries-weight28-window512-v7-epoch08.offset-only-hysteresis-closure-trace.json`,
  SHA-256
  `C272F7A9C623C8F96E43E9D26F230A079E21243DD2311D58DE4ADC035DFBEA95` ;
- 118/118 tests réussis après l'expérience.
