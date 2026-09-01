# Expérience 07 — réarmement confirmé sans événement ouvert

État figé le 31 août 2026. L'expérience est terminée. Le traitement réussit
son objectif isolé de réduction du bruit, mais reste strictement diagnostique :
il n'est pas activé dans le chemin live et ne remplace pas la baseline officielle
V7-e8 associée.

## Question isolée

Peut-on récupérer une partie du filtrage de l'ancien mécanisme d'ouverture sans
réintroduire la règle incorrecte « un nouvel onset doit attendre l'offset de
l'événement précédent » ?

Le contrôle reproduit l'expérience 06 : après un front, un seul échantillon de
score sous `0.55` suffit à réarmer le canal. Le traitement ne modifie qu'un
paramètre :

```text
contrôle   : réarmement après 1 échantillon bas
traitement : réarmement après 16 échantillons bas consécutifs
```

Seize échantillons à 44,1 kHz représentent `0,3628 ms`. Un retour au-dessus du
seuil avant le seizième échantillon remet le compteur à zéro. Cet état est un
état court du détecteur, pas l'état d'une note ouverte.

Les sorties des deux traitements restent strictement :

```text
type, position
```

Il n'y a ni `eventId`, ni corde, ni case, ni hauteur, ni association. Un nouvel
onset peut toujours être émis avant l'offset précédent dès que ses 16
échantillons bas ont réarmé le détecteur. La multiplicité simultanée reste
conservée.

## Audit avant changement

- 60 pistes de validation, 9 541 notes, joueurs `00` à `04` ;
- joueur `05` non lu ;
- distance minimale entre deux onsets annotés successifs d'un même canal :
  3 421 échantillons, soit `77,57 ms` ;
- une seule paire annotée du même canal se chevauche avant son offset ;
- 45 onsets annotés supplémentaires sont exactement simultanés ;
- l'expérience 06 produisait 401 104 onsets et 658 417 offsets candidats,
  soit 42,04 et 69,01 fois les nombres de références ;
- les audits de score antérieurs montraient déjà des centaines de milliers de
  très courts creux autour du seuil.

Le traitement de 16 échantillons est donc très inférieur à la séparation
annotée minimale d'un même canal. Il teste uniquement la confirmation du
réarmement ; aucun seuil, score, modèle, poids, sampler, cible ou loss ne change.

## Intégrité du calcul

- protocole préenregistré avant le code, commit `8936bdc` ;
- automate isolé, commit `dc5a9e7` ;
- évaluateur contrôlé, commit `b368e4e` ;
- checkpoint V7-e8 inchangé, SHA-256
  `5634ADD0E112A6889B65D5245AD051AD850A1FFFE66FEB8D9E5E74472BA114BF` ;
- mêmes 60 pistes, 1 880,8998 secondes d'audio et 162 030 blocs de 512 ;
- une seule inférence par bloc ; le même objet de scores est remis aux deux
  décodeurs et à l'audit morphologique ;
- le contrôle `N=1` reproduit exactement l'expérience 06 globalement et sur
  chacune des 60 pistes ;
- chaque candidat `N=16` est un élément du multiensemble `N=1` ;
- le nombre de candidats supprimés est vérifié égal au nombre de creux bornés
  plus courts que 16 échantillons ;
- aucune métrique d'intervalle, aucun scheduler et aucune activation live ;
- 146/146 tests réussis dans l'environnement TensorFlow réel, aucun skip.

## Résultats onset/offset

| Mesure | Contrôle N=1 | Traitement N=16 | Évolution |
|---|---:|---:|---:|
| Onsets prédits | 401 104 | 158 032 | -243 072 (-60,60 %) |
| TP onset | 9 155 | 9 146 | -9 |
| FP onset | 391 949 | 148 886 | -243 063 |
| FN onset | 386 | 395 | +9 |
| Précision onset | 2,2825 % | 5,7874 % | +3,5050 points |
| Rappel onset | 95,9543 % | 95,8600 % | -0,0943 point |
| F1 onset | 4,4588 % | **10,9158 %** | **+6,4570 points** |
| Offsets prédits | 658 417 | 259 632 | -398 785 (-60,57 %) |
| TP offset | 9 029 | 8 985 | -44 |
| FP offset | 649 388 | 250 647 | -398 741 |
| FN offset | 512 | 556 | +44 |
| Précision offset | 1,3713 % | 3,4607 % | +2,0893 points |
| Rappel offset | 94,6337 % | 94,1725 % | -0,4612 point |
| F1 offset | 2,7035 % | **6,6760 %** | **+3,9725 points** |

Le F1 est multiplié par `2,45` pour onset et par `2,47` pour offset. Le gain
vient presque entièrement de la suppression des faux fronts : seuls 9 TP onset
et 44 TP offset sont perdus.

Le résultat est uniforme :

- F1 onset et offset supérieurs sur 60/60 pistes ;
- F1 onset et offset supérieurs dans 12/12 groupes famille–arrangement ;
- FP onset et offset inférieurs sur 60/60 pistes ;
- TP onset inchangés sur 54 pistes et inférieurs sur 6 ;
- TP offset inchangés sur 30 pistes et inférieurs sur 30 ;
- aucun groupe ne produit davantage de TP, ce qui confirme que le traitement
  filtre des candidats existants au lieu d'en créer.

| Régime | F1 onset N=1 | F1 onset N=16 | F1 offset N=1 | F1 offset N=16 |
|---|---:|---:|---:|---:|
| Global | 4,4588 % | 10,9158 % | 2,7035 % | 6,6760 % |
| Comp | 4,1952 % | 10,3810 % | 2,5572 % | 6,3319 % |
| Solo | 5,4485 % | 12,8232 % | 3,2515 % | 7,9577 % |

## Ce que les données révèlent sur le bruit

L'égalité causale vérifiée est exacte :

| Type | Creux bornés de 1 à 15 échantillons | Candidats supprimés |
|---|---:|---:|
| Onset | 243 072 | 243 072 |
| Offset | 398 785 | 398 785 |

Pour ces creux courts :

- médiane onset et offset : 4 échantillons, soit `0,0907 ms` ;
- p90 onset et offset : 12 échantillons, soit `0,2721 ms` ;
- 55,17 % des creux onset et 56,30 % des creux offset durent au plus quatre
  échantillons ;
- 79,73 % des creux onset et 80,54 % des creux offset durent au plus huit
  échantillons.

La cause mesurée est donc un papillonnement très rapide du score autour du
seuil : il descend brièvement, réarme trop tôt, puis remonte et fabrique un
nouveau front. Attendre `0,3628 ms` récupère bien une partie importante du
pouvoir filtrant de l'ancien mécanisme sans attendre un offset.

## Latence et multiplicité

Le filtrage dégrade légèrement la position des correspondances conservées :

| Mesure | N=1 | N=16 |
|---|---:|---:|
| Médiane signée onset | -0,476 ms | -1,134 ms |
| Médiane causale onset | 1,610 ms | 1,882 ms |
| Médiane signée offset | -0,045 ms | -0,295 ms |
| Médiane causale offset | 1,610 ms | 2,222 ms |

La multiplicité reste représentable mais demeure fortement surproduite :

- référence : 45 onsets simultanés supplémentaires ;
- contrôle : 34 177 ;
- traitement : 10 521 ;
- positions onset uniques : 366 927 vers 147 511, contre 9 496 en référence ;
- positions à multiplicité exactement correcte : 8 860 vers 8 845.

Le traitement supprime donc beaucoup de doublons et de répétitions, au prix
d'une petite perte de correspondances exactes.

## Limite face à la baseline officielle

Le traitement reste moins bon que le décodeur V7 associé officiel :

| Type | F1 V7 associé | F1 N=16 non associé | Prédictions V7 | Prédictions N=16 |
|---|---:|---:|---:|---:|
| Onset | 14,6732 % | 10,9158 % | 111 169 | 158 032 |
| Offset | 13,2417 % | 6,6760 % | 110 987 | 259 632 |

Le traitement conserve un meilleur rappel, mais produit encore 16,56 fois le
nombre d'onsets de référence et 27,21 fois le nombre d'offsets. Il n'est donc
pas un détecteur de types autonome utilisable en live.

## Décision

L'hypothèse isolée est validée : une partie du filtrage de l'ancien mécanisme
d'ouverture peut être conservée sous forme d'un réarmement court, indépendant
de toute note ouverte. La règle demandée reste satisfaite : deux onsets peuvent
se succéder avant un offset.

Le traitement `N=16` devient le meilleur contrôle **diagnostique non associé**
pour une expérience suivante. Il ne devient pas la baseline live officielle.
V7-e8 associé à `0.55/0.55` reste inchangé.

La quantité de faux candidats restante impose encore une étape de détection de
type ou de consolidation, sans corde/case publique, avant toute association par
`eventId`. Aucun entraînement, aucune association et aucune activation live ne
sont lancés ici.

## Artefacts

- protocole :
  `model/causal-boundaries-weight28-window512-v7-epoch08.unassociated-boundary-rearm-low1-16-protocol.json`,
  SHA-256
  `794D54EC573B355CC77F96EA0DFD7242E82D1A6571F39BD65EFC4418A0EFD7CF` ;
- résultat :
  `model/causal-boundaries-weight28-window512-v7-epoch08.unassociated-boundary-rearm-low1-16.json`,
  SHA-256
  `4E5249C4AB2E6C3B170AA450BE8E27F6DFCF95692DF0917623C268485E20AD88` ;
- évaluateur : `scripts/evaluate_boundary_rearm.py`, SHA-256
  `06FD9C7BF4982F21DFAF9ABF736342A52FDFE29D637A953EF4B67FB047BF1A3A` ;
- automate : `src/causal_note/detector.py`, SHA-256
  `F2359EC1B1357580403B2DBB8BECBD407B5FC07A841909F01BA219C6F2B8FCFD` ;
- durée murale : 3 135,23 s, soit 52,25 min ; facteur temps réel de calcul
  `1,6249`.
