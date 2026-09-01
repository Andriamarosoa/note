# Expérience 10 — audit des cibles anonymes `type, position`

État figé le 1er septembre 2026. L'audit est terminé. Aucun modèle n'a été
modifié, aucun entraînement n'a été lancé et le live n'a pas changé.

## Verdict

La cible la plus directe respecte bien le besoin demandé :

```text
onset_count[t]  = nombre exact d'onsets à l'échantillon t
offset_count[t] = nombre exact d'offsets à l'échantillon t
```

Un compte `k` est ensuite développé en `k` événements publics identiques. Par
exemple :

```text
onset_count[100] = 2

-> onset, 100
-> onset, 100
```

Cette représentation ne contient ni corde, ni canal, ni case, ni hauteur, ni
`active`, ni `eventId`. Elle conserve les événements simultanés et autorise un
nouvel onset avant tout offset.

Elle est **structurellement valide sur les positions qui possèdent un
échantillon audio**, mais elle n'est pas encore prête pour un entraînement :

- 408 offsets annotés se trouvent exactement après le dernier échantillon de
  leur fichier ;
- le sampler V7 modifie la densité des nouvelles cibles de plus de deux fois ;
- le diagnostic CE catégorielle réutilisant le poids 28 pousse la cible exacte
  vers « aucun événement » ;
- ce même poids pousse une cible anonyme large de 512 vers « événement partout ».

Le modèle, la loss, le sampler et le décodeur de la prochaine version ne sont
donc pas sélectionnés par cet audit.

## Données exactes

Aucun contenu JAMS ou WAV de Player05 n'a été ouvert ou utilisé. L'index voit
seulement les noms présents dans les répertoires ZIP. L'audit porte sur 240
pistes train et 60 pistes validation, avec exactement la même partition que V7.

| Split | Type | Annotations | Supervisables | Positions positives | Simultanés supplémentaires | Maximum supervisé | Densité positive |
|---|---|---:|---:|---:|---:|---:|---:|
| Train | Onset | 44 220 | 44 220 | 44 059 | 161 | 3 | 0,013762 % |
| Train | Offset | 44 220 | 43 894 | 43 833 | 61 | 3 | 0,013692 % |
| Validation | Onset | 9 541 | 9 541 | 9 496 | 45 | 3 | 0,011448 % |
| Validation | Offset | 9 541 | 9 459 | 9 451 | 8 | 2 | 0,011394 % |

« Positions positives » compte une position une seule fois, même si son compte
vaut 2 ou 3. « Simultanés supplémentaires » est la différence entre le nombre
d'événements et le nombre de positions.

Les annotations offset brutes, y compris la fin exclusive des WAV, contiennent
une multiplicité plus élevée : maximum 6 en train et 5 en validation. Les 326
offsets train et 82 offsets validation placés à la fin du fichier expliquent
la différence avec la colonne supervisable. Ils n'ont pas été supprimés ou
décalés : ils sont conservés comme un problème explicite à résoudre.

## Pourquoi une sortie binaire anonyme ne suffit pas

Faire un simple `OR` des six anciennes sorties supprime la multiplicité exacte.
Avec les fenêtres positives de 512, il fusionne aussi des événements proches
dans un même plateau :

| Split | Type | Événements supervisables | Fronts binaires | Événements non séparables | Perte |
|---|---|---:|---:|---:|---:|
| Train | Onset | 44 220 | 29 695 | 14 525 | 32,85 % |
| Train | Offset | 43 894 | 34 175 | 9 719 | 22,14 % |
| Validation | Onset | 9 541 | 6 726 | 2 815 | 29,50 % |
| Validation | Offset | 9 459 | 7 733 | 1 726 | 18,25 % |

Le nombre `2 815` signifie donc : sur 9 541 onsets validation, seuls 6 726
fronts distincts restent après l'union binaire. Ce ne sont pas 2 815 paires
indépendantes.

Une cible **count** large de 512 conserve davantage d'information qu'un OR.
L'inversion causale

```text
x[t] = y[t] - y[t-1] + x[t-512]
```

reconstruit exactement toutes les frontières supervisables des 300 pistes dans
le contrôle sparse. Un simple décodeur de fronts montants n'y arrive cependant
pas : quand une fenêtre expire exactement au démarrage d'une autre, la variation
peut s'annuler. Il perd 15 onsets et 26 offsets en validation, contre zéro perte
avec l'inversion causale. Un test unitaire dense synthétique séparé vérifie la
même inversion ; il ne prétend pas être un second passage dense sur les 300
pistes.

Cette exactitude mathématique ne garantit pas la robustesse avec des comptes
prédits imparfaits ; la cible large n'est donc pas retenue automatiquement.

## Ce que le sampler et la loss révèlent

Le simulateur reproduit exactement les 400 fenêtres de validation V7 :

- 134 fenêtres ancrées onset, 133 offset et 133 aléatoires ;
- 261 423 éléments slot-onset positifs ;
- 249 851 éléments slot-offset positifs.

Sur ces mêmes fenêtres :

| Cible validation | Type | Densité flux complet | Densité batches | Rapport batches/flux | Optimum constant avec poids 28 |
|---|---|---:|---:|---:|---:|
| Compte exact | Onset | 0,011448 % | 0,032012 % | 2,80× | 0,0089 |
| Compte exact | Offset | 0,011394 % | 0,029756 % | 2,61× | 0,0083 |
| Compte large 512 | Onset | 4,8750 % | 12,5913 % | 2,58× | 0,8013 |
| Compte large 512 | Offset | 5,2720 % | 13,2626 % | 2,52× | 0,8107 |
| V7, six slots | Onset | 0,9815 % | 2,6567 % | 2,71× | 0,4332 |
| V7, six slots | Offset | 0,9731 % | 2,5391 % | 2,61× | 0,4218 |

L'« optimum constant » est la probabilité constante qui minimise la loss si le
modèle n'apprend aucune distinction acoustique. C'est un diagnostic de pression
de la loss, pas le résultat prédit d'un vrai entraînement.

Avec l'ancien seuil 0,55 utilisé uniquement comme repère :

- `0,0089` et `0,0083` donnent toujours zéro pour la cible exacte ;
- `0,8013` et `0,8107` donnent toujours positif pour la cible large ;
- le poids 28, le seuil 0,55 et le sampler V7 ne sont donc pas transférables
  tels quels à une nouvelle tête anonyme. La BCE slot de V7 n'a pas été
  présentée comme une loss de compte : l'audit du compte utilise une CE
  catégorielle.

Le pool d'ancres V7 contient aussi les doublons de positions simultanées. Une
position contenant plusieurs événements a donc une probabilité de tirage plus
élevée alors qu'une projection binaire n'en montrerait qu'un.

## Relations temporelles révélées par les annotations

En validation, parmi les paires onset–offset séparées de moins de 512
échantillons :

- 3 918 ont l'offset avant l'onset ;
- 1 142 ont l'onset avant l'offset ;
- 7 sont exactement au même échantillon.

Pour deux notes successives de la même corde annotée :

- 2 597 ont un offset puis un nouvel onset moins de 512 échantillons après ;
- 7 ont l'offset et le nouvel onset au même échantillon ;
- 1 est un vrai retrigger : le nouvel onset arrive avant l'ancien offset.

La corde est utilisée ici uniquement pour vérifier la succession des
annotations. Elle n'entre pas dans la cible. Ces chiffres confirment que le
futur décodeur doit préserver l'ordre, définir le traitement des égalités et
ne jamais exiger un offset avant d'accepter un nouvel onset.

## Diagnostic par blocs de 512

Un multiensemble causal de frontières par bloc reste une possibilité de
conception, mais n'a pas été choisi :

| Split | Blocs | Blocs vides | Maximum combiné onset+offset | Blocs non vides couverts par capacité 6 | Instances conservées avec capacité 6 |
|---|---:|---:|---:|---:|---:|
| Train | 625 380 | 89,75 % | 8 | 99,9361 % | 99,9410 % |
| Validation | 162 030 | 91,05 % | 8 | 99,9724 % | 99,9737 % |

Une capacité exacte de 8 couvre toutes les observations. Choisir 6 serait
presque suffisant, mais écrêterait réellement des événements ; l'audit refuse
donc de transformer ce « presque » en décision. Une architecture par bloc
devrait en outre empêcher les échantillons futurs du bloc d'influencer une
frontière antérieure.

## Décision et prochaine étape

La base de conception la plus simple est le **compte exact anonyme par
échantillon**, développé en événements répétés `(type, position)`. Mais avant
de coder ou d'entraîner le prochain modèle, deux décisions sont obligatoires :

1. définir le traitement explicite des 408 offsets de fin de flux ;
2. concevoir puis auditer un sampler et une loss adaptés à la très faible
   densité exacte, sans réutiliser le poids 28.

Conformément au protocole, l'expérience s'arrête ici pour approbation. Il n'y a
pas de V8, pas de checkpoint et pas de modification live dans cette étape.

## Intégrité et artefacts

- protocole préenregistré au commit `1fa7186`, SHA-256
  `BDAFD38D09C230841CE4AE42EC341129FFB7263F662683C3BA7F10F6F194D1A9` ;
- amendement préimplémentation au commit `2a4c8ad`, SHA-256
  `8DADC12687037C9AC5DF6408F5C3F88A9AEF06145DBB214B4B53351E0B310BAF` ;
- implémentation et résultat au commit `647e672` ;
- script d'audit, SHA-256
  `7C414A863E941B3595F03F7A420A5FB80810A5F4D2D9C5350C8089491135B356` ;
- tests dédiés, SHA-256
  `A48998D126A594F73DBB7B5C73F24C4934AAAF492C436B1C33593D533DD95898` ;
- résultat JSON, SHA-256
  `D1B311666BAFE8191D6CE3B655EE6CFB8B140F053741B7DF9D98613D0E93CA14` ;
- 182/182 tests réussis dans l'environnement TensorFlow réel ;
- aucun contenu annotation/audio Player05 ouvert ou utilisé.
