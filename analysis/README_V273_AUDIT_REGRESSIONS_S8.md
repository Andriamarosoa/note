# V27.3 — audit des régressions avec S8

**Le catalogue étendu modifie défavorablement le choix des verdicts, y
compris lorsque le mauvais verdict existait déjà et que S8 seule était
correcte.** Les archives permettent de localiser ces décisions ; elles ne
permettent pas d'attribuer causalement chaque erreur à une entrée du réseau.

Audit sans réentraînement ni changement de seuil, sur les sorties vérifiées
du run [37789646292](https://github.com/Andriamarosoa/note/actions/runs/37789646292).
Source entraînée : `3f83c2b07308c9114f1db0b5a904c0d80a91b91a`.

## Les populations à distinguer

546 régressions signifient : le verdict figé était correct et le système
avec S8 le rend faux. Le contrôle à sept sélections en avait 522.

- 378 persistent entre les deux systèmes ;
- 144 anciennes régressions sont réparées ;
- 168 nouvelles apparaissent : `378 + 168 = 546`.

À part, **163 anciennes corrections sont perdues** : ces fragments étaient
faux dans le verdict figé, corrigés par le contrôle, puis faux avec S8.
Ce ne sont pas des régressions face au verdict figé. Face au contrôle, les
331 pertes de justesse se décomposent donc en `168 + 163`.

| Vrai K | Régressions totales | Nouvelles | Réparées | Corrections totales |
|---|---:|---:|---:|---:|
| K2 | 168 | 43 | 48 | 262 |
| K3 | 252 | 91 | 62 | 228 |
| K4 | 126 | 34 | 34 | 77 |
| Total K2–K4 | 546 | 168 | 144 | 567 |

Les 17 autres corrections concernent K5 ; aucune régression de vrai K5/K6
n'est possible ici, puisque seules les prédictions initiales K2/K3/K4
peuvent être modifiées. K3/K4 représentent 378/546 régressions, soit 69,2 %.

## Les mauvais verdicts ne viennent pas tous de la nouvelle sélection

- **512/546** régressions choisissent un verdict disponible dans l'ancien
  catalogue.
- Parmi les **168 nouvelles**, **143** choisissent un ancien verdict et
  **25** une destination nouvellement disponible.
- **S8 seule est correcte sur 297/546 régressions**, dont **95/168 nouvelles**.

Ces nombres invalident l'explication selon laquelle S8 fournirait simplement
de mauvais nouveaux verdicts suivis aveuglément. Le réseau étendu change
aussi les scores de destinations existantes. Son architecture ajoute 544
paramètres, ses entrées changent et les nouveaux groupes contribuent aux
logits partagés : cet audit n'isole pas lequel de ces facteurs cause l'écart.

Il ne faut pas en déduire un veto automatique de S8 : **274 corrections
réussissent alors que S8 seule conserve le verdict initial faux**. Évaluer
les groupes complets reste nécessaire.

## Transitions choisies

Les flèches désignent le K initial puis le K final, pas la vérité.
Le net compte les corrections moins les régressions face au verdict figé.

| Transition | Corrections | Régressions | Nouvelles régressions | Net |
|---|---:|---:|---:|---:|
| K2 → K3 | 131 | 152 | 36 | −21 |
| K2 → K4 | 6 | 16 | 7 | −10 |
| K3 → K2 | 210 | 184 | 61 | +26 |
| K3 → K4 | 71 | 59 | 23 | +12 |
| K3 → K5 | 9 | 9 | 7 | 0 |
| K4 → K2 | 52 | 25 | 4 | +27 |
| K4 → K3 | 97 | 86 | 25 | +11 |
| K4 → K5 | 8 | 15 | 5 | −7 |

K3 → K2 cause le plus de régressions brutes, mais conserve un net positif.
L'interdire en bloc supprimerait aussi 210 corrections. K2 → K3 est la
transition au net le plus négatif dans cet essai.

## Les probabilités et les historiques discriminent peu

Le réseau annonce **+397,42** sur ses changements, contre **+38 réel** :
872,92 corrections prévues pour 584 observées, et 475,50 régressions
prévues pour 546 observées.

Le gain annoncé sépare les 584 corrections et les 546 régressions avec
une AUC de **0,543** ; une AUC de 0,5 correspond à un classement aléatoire.
Ce calcul exclut les 766 changements entre deux verdicts faux.

Les historiques sont ici résumés par la moyenne sur **tous** les groupes
proposant le K choisi, sans attribuer le résultat à un masque représentatif.

| Historique | Utilisé par ce sélecteur ? | Gain historique négatif sur les corrections | Sur les régressions | AUC |
|---|---|---:|---:|---:|
| Global | Oui | 298 | 289 | 0,530 |
| Local par similarité | Non | 238 | 266 | 0,542 |

Un veto rétrospectif sur le signe de l'historique global retirerait 289
régressions mais aussi 298 corrections, soit −9 net. Pour l'historique
local, ce compte serait +28. **Ce dernier nombre n'est pas le résultat d'un
réseau entraîné avec l'historique local**, ni une règle validée. Les colonnes
locales ont été calculées et exportées, puis neutralisées dans les entrées
du sélecteur pour garder le même bras global que le contrôle précédent.

## Un cas concret de nouvelle régression

Événement **63735**, `04_Funk2-119-G_solo.jams`, à **25,317 s**, fold 1 :

- vrai K : **2** ; verdict figé : **2** ; contrôle : **2** ; S8 seule : **2** ;
- système étendu : **3**, destination déjà disponible auparavant ;
- gain annoncé pour K3 : **+0,4975**, contre **−0,1486** dans le contrôle ;
- probabilité annoncée de correction : **71,54 %** ;
- gain historique moyen global : **−0,1270** ; local inutilisé : **−0,2210**.

La probabilité de justesse du verdict initial augmente même de 18,90 % à
21,80 % entre les modèles, ce qui va vers davantage de prudence. Dans ce
cas, l'augmentation du score de l'alternative K3 l'emporte néanmoins.
Il s'agit d'un exemple extrême de confiance erronée, pas d'un cas représentatif.

## Preuves et suite justifiée

Le script `scripts/audit_v273_s8_regressions.py` contrôle les empreintes des
archives, l'identité des anciens candidats, les populations et les exports.

- [audit.json](evidence/v273-s8-regressions/audit.json) : transitions, calibration, historiques et cas détaillés ;
- [regressions-546.csv](evidence/v273-s8-regressions/regressions-546.csv) : toutes les régressions, marquées nouvelles ou persistantes ;
- [corrections-perdues-163.csv](evidence/v273-s8-regressions/corrections-perdues-163.csv) : pertes séparées des anciennes corrections.

L'expérience suivante pertinente est une comparaison contrôlée de
l'exploitation des historiques locaux avec S8, en conservant la concurrence
des groupes complets. Cet audit donne une raison de la tester, pas une
garantie de gain. Il ne démontre pas de signal acoustique manquant ni de
généralisation : les folds 0/1/2/4 restent exposés au développement.
