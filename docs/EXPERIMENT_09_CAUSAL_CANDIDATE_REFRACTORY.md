# Expérience 09 — période réfractaire causale fixe de 50 ms

État figé le 1er septembre 2026. L'expérience est terminée.

Le filtre réduit fortement les faux candidats et améliore le F1 sur toutes les
pistes, mais il **échoue à la règle d'acceptation préenregistrée** : il supprime
trop de supports offset, dégrade la multiplicité simultanée et laisse encore
plus de cinq fois le nombre de frontières attendu. Il n'est donc pas activé
dans le live.

## Modification unique

Le contrôle reste le diagnostic N=16 de l'expérience 07. Le traitement ajoute
une seule règle avant que le canal interne soit effacé :

```text
clé d'état = type onset/offset + canal interne

premier candidat                 -> émis immédiatement
candidat suivant à 1..2205       -> supprimé
candidat suivant après 2205      -> émis et devient la nouvelle ancre
```

Un candidat supprimé ne prolonge pas la fenêtre. Onset et offset ont des états
séparés, tout comme les six canaux. Le traitement n'utilise aucun futur, ne
décale aucune position et n'ajoute aucune latence algorithmique.

La sortie publique reste strictement :

```text
type, position
```

Il n'y a ni corde publiée, ni case, ni hauteur, ni association onset–offset,
ni `eventId`.

## Protocole et limite d'adaptation

Le protocole a été préenregistré au commit `a68a4a7`. Une revue indépendante a
ensuite verrouillé, avant le code, l'appariement officiel, le support simultané
et les seuils entiers dans l'amendement `e78bc15`.

La durée de 50 ms a été choisie après l'analyse de ces mêmes 60 pistes de
validation. Ce résultat est donc descriptif sur un jeu ayant servi à adapter
la règle ; ce n'est pas une confirmation indépendante et aucune généralisation
au joueur 05 ne peut être revendiquée.

## Sécurité suggérée par les annotations, mais non garantie

Les frontières annotées successives d'un même canal sont séparées d'au moins :

| Type | Minimum annoté | Marge au-delà de 50 ms |
|---|---:|---:|
| Onset | 3 421 échantillons = 77,57 ms | 27,57 ms |
| Offset | 2 844 échantillons = 64,49 ms | 14,49 ms |

Cette marge ne protège pas contre les erreurs du modèle. Un faux candidat peut
devenir l'ancre et supprimer ensuite un candidat correspondant à une vraie
frontière. Le post-audit devait précisément mesurer ce cas.

## Intégrité

- joueurs `00` à `04`, joueur `05` non lu ;
- 60 pistes, 9 541 notes et 1 880,8998 secondes d'audio ;
- 162 030 blocs causaux de 512 échantillons ;
- une seule inférence par bloc ;
- même objet immuable remis au contrôle, au traitement et à l'observateur ;
- contrôle N=16 reproduit exactement Exp08 globalement et sur 60/60 pistes ;
- partition relationnelle Exp08 et support par note reproduits sur chaque piste ;
- chaque sortie du traitement appartient au multiensemble du contrôle ;
- chaque suppression respecte exactement type, canal, ancre et distance ;
- aucun score brut ni aucune liste de candidats n'est écrit ;
- 171/171 tests réussis dans l'environnement TensorFlow réel ;
- durée murale : 1 692,02 s, soit 28,20 minutes ;
- joueur 05 toujours verrouillé.

## Résultat global

### Onset

| Mesure | Contrôle N=16 | Traitement 50 ms | Évolution |
|---|---:|---:|---:|
| Prédictions | 158 032 | **48 346** | -109 686 (-69,41 %) |
| TP | 9 146 | 9 087 | -59 |
| FP | 148 886 | **39 259** | -109 627 (-73,63 %) |
| FN | 395 | 454 | +59 |
| Précision | 5,79 % | **18,80 %** | +13,01 points |
| Rappel | 95,86 % | 95,24 % | -0,62 point |
| F1 | 10,92 % | **31,40 %** | +20,48 points |
| Prédictions / références | 16,56× | **5,07×** | encore trop élevé |

### Offset

| Mesure | Contrôle N=16 | Traitement 50 ms | Évolution |
|---|---:|---:|---:|
| Prédictions | 259 632 | **53 104** | -206 528 (-79,55 %) |
| TP | 8 985 | 8 787 | -198 |
| FP | 250 647 | **44 317** | -206 330 (-82,32 %) |
| FN | 556 | 754 | +198 |
| Précision | 3,46 % | **16,55 %** | +13,09 points |
| Rappel | 94,17 % | 92,10 % | -2,08 points |
| F1 | 6,68 % | **28,05 %** | +21,38 points |
| Prédictions / références | 27,21× | **5,57×** | encore trop élevé |

Le F1 progresse sur 60/60 pistes et dans 12/12 groupes famille–arrangement
pour les deux types. Cette amélioration uniforme ne suffit toutefois pas à
valider la règle, car elle masque des pertes structurelles importantes.

Le rappel offset du régime solo descend notamment à 84,55 %, contre 94,96 %
pour le régime comp après traitement.

## Pourquoi le verdict « utile » échoue

| Garde-fou préenregistré | Résultat | Verdict |
|---|---:|---|
| Réduction FP onset ≥ 15 % | 73,63 % | réussi |
| Réduction FP offset ≥ 15 % | 82,32 % | réussi |
| Conservation TP onset ≥ 99 % | 99,35 % | réussi |
| Conservation TP offset ≥ 99 % | **97,80 %** | échoué |
| F1 supérieur global, comp, solo et ≥10/12 groupes | 12/12 groupes | réussi |
| Support simultané non inférieur | onset 89→88 ; offset 71→60 | échoué |
| Notes avec onset et offset supportés ≥ 7 198 | **7 270→6 552** | échoué |

Le filtre est donc déclaré **non accepté**, malgré le gain de F1.

Le critère séparé « sur-prédiction résolue sur ce diagnostic » exigeait au
plus deux prédictions par référence et au moins 90 % de rappel. Le rappel
global passe, mais les ratios restent à 5,07× et 5,57×. La sur-prédiction
reste donc **non résolue**.

## Ce que l'audit des relations révèle après changement

| Relation parmi les FP restants | Onset | Offset |
|---|---:|---:|
| Isolés | 18 878 = 48,09 % | 26 509 = 59,82 % |
| Répétition forte même canal/référence | 1 020 = 2,60 % | 2 646 = 5,97 % |
| Proches d'une seule référence | 13 517 = 34,43 % | 9 764 = 22,03 % |
| Ambigus | 5 844 = 14,89 % | 5 398 = 12,18 % |

La répétition forte, cible directe de la règle, devient minoritaire. Le filtre
a donc bien retiré une grande partie des rafales du même canal. Les erreurs
restantes sont surtout des activations isolées ou des candidats répartis dans
un voisinage que cette règle ne peut pas résoudre.

Parmi les occurrences supprimées, 65,68 % des onsets et 60,82 % des offsets
étaient des FP isolés dans le contrôle ; 17,76 % et 20,60 % étaient des
répétitions fortes. Le filtre retire donc beaucoup de bruit réel.

Il supprime aussi 3 083 occurrences onset et 5 375 occurrences offset qui
étaient appariées dans le contrôle. Ces nombres ne sont pas les pertes nettes
de TP : après rematching complet, d'autres candidats remplacent la plupart de
ces occurrences, et les pertes nettes sont seulement 59 TP onset et 198 TP
offset.

Parmi ces occurrences auparavant appariées, 2 377/3 083 onsets, soit 77,10 %,
et 4 808/5 375 offsets, soit 89,45 %, sont bloqués par une ancre elle-même
classée faux positif dans le contrôle. La règle garde donc fréquemment le
premier faux signal du canal puis supprime un candidat plus pertinent.

## Défaut de conception mis en évidence par les canaux

L'audit un-à-un sur le canal supervisé donne :

| Support par note sur le même canal | Contrôle | Traitement | Perte |
|---|---:|---:|---:|
| Onset | 8 879 | 8 706 | -173 |
| Offset | 7 681 | 7 082 | -599 |
| Onset et offset | 7 270 | 6 552 | -718 |

La perte de support par note est bien plus grande que la perte de TP public.
Cela signifie que l'appariement public trouve souvent un candidat de
remplacement provenant d'un autre canal. Il conserve alors le score `type,
position`, mais ne prouve pas que le canal ayant porté l'onset porte aussi le
bon offset.

La multiplicité reste elle aussi surproduite après filtrage : 1 854 onsets
simultanés supplémentaires contre 45 en référence, et 702 offsets contre 54.
Les réactions réparties sur plusieurs canaux ne sont pas fusionnées par une
réfractarité qui travaille séparément dans chaque canal.

Le filtre dépend donc d'un canal interne qui n'est pas une identité fiable de
l'événement demandé. Une fausse activation sur ce canal peut occuper les 50 ms
et bloquer la vraie frontière, tandis qu'une activation d'un autre canal peut
masquer cette perte dans la métrique publique. C'est exactement la limite de
conception redoutée : une erreur de canal influence encore la détection du
type.

La séparation minimale des annotations n'était donc pas fausse ; elle était
insuffisante pour garantir le filtre, parce qu'elle décrit les vraies notes et
non les fausses ancres produites par le modèle.

## Effet temporel

| Erreur absolue médiane des correspondances | Contrôle | Traitement |
|---|---:|---:|
| Onset | 2,99 ms | 5,31 ms |
| Offset | 2,70 ms | 8,96 ms |

Le filtre ne décale pourtant aucun candidat et n'ajoute aucun délai. La
dégradation vient du candidat conservé : garder le premier élément de la
fenêtre laisse parfois une correspondance plus éloignée de la frontière que
le candidat supprimé.

## Décision et arrêt

La période réfractaire fixe de 50 ms est une amélioration diagnostique forte,
mais elle n'est ni une solution complète ni une règle admissible pour le live.
Allonger simplement la fenêtre risquerait d'augmenter encore les pertes offset
et le défaut de support par note.

Le résultat indique que la prochaine modification ne doit pas dépendre d'une
corde/canal supposé correct. Elle devra traiter la détection de `type,
position` avant l'association, tout en conservant la multiplicité réelle des
événements simultanés. Conformément au protocole, aucune nouvelle architecture,
aucun entraînement et aucune activation live ne sont lancés sans un nouvel
accord.

## Artefacts

- protocole, SHA-256
  `A8C5B1BF5D9D05BFE59C7F2800B5BC756771485045F043CFB0B0B210781A5576` ;
- amendement, SHA-256
  `9881D3113054F31795CA14EE6B7859194CDDBD20DC105A3476ADDE8C9821FF02` ;
- détecteur expérimental, SHA-256
  `BE4F15D4999D802640074D91F1A772EAC764F91F391F6E35EAA6CF2FAB5460FB` ;
- évaluateur, SHA-256
  `D0AF0924C469086CA1D2456188586E85B7CFFCBB128BF34C1D7D84A4DA819695` ;
- tests de l'évaluateur, SHA-256
  `AD6F5B8EF46DF49E9666E7947F7A3E10FE7CBB753F676BCF63570F1E37DE6E68` ;
- résultat, SHA-256
  `A731E25C87A73D0C5F16B820BA7657E521F10036B1D72B9047AB340266020C6B`.
