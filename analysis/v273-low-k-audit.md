# Audit des surcomptages à vrai K < 4 — fold 3

**Audit terminé. La pondération des classes est une cause démontrée de
l'aggravation du surcomptage dans cette expérience. Elle n'explique pas à
elle seule les erreurs déjà présentes sans pondération.** La fenêtre élargie
n'est pas une cause générale suffisante : sans pondération, le total des
surcomptages K<4 reste à 874, avec 23 comme avec 31 trames.

Le périmètre couvre les **14 744 groupes à vrai K=0, 1, 2 ou 3**, parmi les
15 279 groupes du fold 3. Les prédictions de 4, 5 et 6 sont bien conservées
lorsqu'elles surcomptent ces groupes. Aucun autre fold externe n'a été évalué.
Les quatre modèles restent figés ; aucun entraînement ni correcteur ajouté.

## 1. Cause de l'aggravation : le coût d'apprentissage favorise les classes rares

| Vrai K | Groupes | Surcomptages à 31 trames, uniforme | Surcomptages à 31 trames, pondéré |
|---|---:|---:|---:|
| 0 | 10 285 | 315 — 3,06 % | 525 — 5,10 % |
| 1 | 3 025 | 292 — 9,65 % | 480 — 15,87 % |
| 2 | 806 | 201 — 24,94 % | 235 — 29,16 % |
| 3 | 628 | 66 — 10,51 % | 194 — 30,89 % |
| Total K<4 | 14 744 | **874** | **1 434** |

À 31 trames, changer seulement la pondération de l'entraînement crée
593 nouveaux surcomptages et en supprime 33 : **+560 au total**. Parmi les
nouveaux, **556 étaient auparavant des comptes exacts** et 37 des sous-comptages.
La hausse est présente dans chacune des cinq compositions. Les mêmes données,
poids initiaux, graines, permutations de lots et budgets de huit époques ont
été vérifiés entre les quatre variantes.

Les poids de classes appliqués à l'entropie croisée sont :

| K | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Poids | 0,6331 | 1,1379 | 2,1478 | 2,6348 | 3,6285 | 5,4781 | 5,4781 |

Ils ont été recalculés depuis les fréquences de l'apprentissage et concordent
exactement avec les poids sauvegardés. **Ce n'est pas une erreur de calcul des
poids** : c'est un effet du choix de l'objectif. Une erreur sur un exemple de
classe 4 pèse 5,73 fois plus qu'une erreur sur un exemple de classe 0. Le modèle
apprend à moins manquer les grandes classes, au prix de davantage de petites
classes prédites trop haut. Les poids agissent sur la perte d'entraînement,
pas comme un bonus ajouté aux sorties pendant cette analyse.

Le cas le plus net est **3 → 4** : 180 des 194 surcomptages à K=3 du modèle
pondéré à 31 trames prédisent 4. Sur 116 de ces groupes, le modèle uniforme à
31 trames prédisait correctement 3. Cela éclaire le gain précédent sur K=4 :
la frontière entre classes s'est déplacée et pénalise une partie des K=3.

Avec 23 trames, la pondération augmente déjà les surcomptages K<4 de
874 à 1 283 (+409). Passer de 23 à 31 trames en pondéré les fait ensuite
passer à 1 434 (+151). Ces deux effets ne doivent pas être confondus.

## 2. Les cibles comptent des attaques attribuées, pas toutes les notes présentes

Les **9 266 annotations** des 50 pistes ont été réaffectées aux candidats
entiers conservés : **8 812 attribuées et 454 non attribuées**, avec les mêmes
étiquettes K sur les 15 279 groupes. Aucun décalage de lignes ni écart de
comptage n'a été trouvé dans cette reproduction du protocole.

**K=0 ne signifie pas nécessairement silence**. Cela signifie qu'aucune
nouvelle attaque n'est attribuée à ce groupe. La fenêtre peut contenir une
note antérieure encore active, ou une attaque appartenant à un autre groupe.

Pour le modèle pondéré à 31 trames :

- À K=0, une attaque hors du groupe est associée à **224/1 927 = 11,62 %**
  de surcomptage, contre **301/8 358 = 3,60 %** sans cette attaque.
- À K=1, les taux correspondants sont **130/364 = 35,71 %** et
  **350/2 661 = 13,15 %**.
- À K=2, une attaque hors du groupe dans la partie ajoutée de la fenêtre est
  associée à **48/105 = 45,71 %**, contre **187/701 = 26,68 %** sans elle.

Ces associations indiquent un contexte ambigu pour le comptage, **sans prouver
que chaque attaque voisine cause l'erreur**. La simple présence d'une note
antérieure n'est pas une explication universelle : à K=0, le taux de
surcomptage est de 4,08 % avec une note antérieure active, contre 25,72 % sans.
Compter seulement les 400 faux positifs contenant une note antérieure aurait
donc donné une conclusion trompeuse sans les dénominateurs.

Il existe également **85 surcomptages parmi 347 groupes K=0 sans aucune note
annotée chevauchant la fenêtre** (24,50 %). Cela ne démontre pas que le signal
audio est silencieux : bruits, transitoires, queues acoustiques ou limites des
annotations nécessitent un examen acoustique spécifique. Une confusion
d'harmoniques n'est pas établie par cet audit.

## 3. Les interventions ne justifient pas de couper la fenêtre

Les quatre réseaux ont été rechargés avec leurs poids réels. Les **61 116
décisions** rejouées correspondent exactement aux sorties enregistrées ; le
plus grand écart de probabilité est 0,00000120. Le décodage et l'export des
prédictions n'expliquent donc pas le surcomptage observé.

Sondes sur le modèle pondéré à 31 trames, toujours sur les mêmes 14 744 groupes :

| Entrée ou représentation modifiée, à poids fixes | Comptes exacts | Surcomptages | Sous-comptages |
|---|---:|---:|---:|
| Aucune — modèle original | 12 319 | 1 434 | 991 |
| Huit dernières trames mises à zéro | 12 198 | 286 | 2 260 |
| Huit dernières remplacées par la 23e trame | 12 143 | 1 714 | 887 |
| Huit premières trames mises à zéro — contrôle | 11 424 | 2 558 | 762 |
| Représentation spectrale mise à zéro avant la tête | 12 303 | 817 | 1 624 |
| Représentation des candidats mise à zéro avant la tête | 8 325 | 5 812 | 607 |

Le zéro en fin de fenêtre fait baisser le compte, mais transforme notamment
386 surcomptages en sous-comptages et fait perdre 121 comptes exacts au total.
La répétition d'une trame produit un effet opposé sur le surcomptage. La
représentation candidate joue aussi un rôle protecteur : sa suppression fait
fortement monter le surcomptage. **Aucune de ces sondes n'est une correction.**

Sur 25 surcomptages, les deux modifications de fin de fenêtre donnent un
compte exact alors que le contrôle en début de fenêtre laisse un surcomptage.
Seulement trois de ces cas comportent une attaque hors du groupe dans la
partie ajoutée. Cette concordance localise une sensibilité pour un petit
sous-ensemble ; elle n'explique pas les 1 434 erreurs. Les interventions
peuvent sortir de la distribution d'apprentissage et ne séparent pas les
sources physiques du signal.

Le graphe réellement chargé confirme que le réseau natif combine une moyenne
et un maximum des caractéristiques spectrales avec la représentation des
candidats, puis prédit une classe K. Les têtes de localisation d'attaques,
d'affectation par corde et de sélection d'événements sont absentes de ce
graphe réduit. **Il n'y a pas de vérification explicite qu'une attaque comptée
appartient au groupe courant.** C'est une limite structurelle documentée ;
son rôle causal exact doit être isolé par une expérience d'apprentissage,
et ne peut pas être déduit uniquement d'une suppression de caractéristiques.

## 4. Exemples vérifiables

Les exemples sont choisis pour illustrer des mécanismes distincts ; ils ne
constituent pas un échantillon représentatif des fréquences d'erreur.

| Piste et début du groupe | Vrai K | Prédiction pondérée, 31 trames | Constat |
|---|---:|---:|---|
| `01_SS2-88-F_comp.jams`, 4,98678 s | 0 | 1 | Une attaque hors du groupe dans la fin ajoutée ; les deux sondes de fin donnent 0, le contrôle de début reste à 1. |
| `04_Jazz2-187-F#_comp.jams`, 12,04971 s | 3 | 4 | Les deux modèles uniformes donnent 3, les deux pondérés donnent 4 ; aucune attaque hors du groupe ni note antérieure active. |
| `01_SS2-88-F_solo.jams`, 32,75125 s | 0 | 1 | Une note antérieure, aucune attaque dans la fenêtre ; les quatre modèles donnent 1. |
| `03_Funk3-98-A_solo.jams`, 14,99340 s | 0 | 1 | Aucune note annotée ne chevauche la fenêtre ; les quatre modèles donnent 1. L'origine acoustique reste à examiner. |

Les indices globaux, probabilités et résultats de toutes les sondes pour ces
cas sont conservés dans le diagnostic JSON.

## Verdict et portée

**Cause confirmée de l'aggravation : l'objectif pondéré déplace les décisions
vers les classes plus grandes, particulièrement 3 → 4.** Le défaut de fenêtre
historique ne suffit pas à expliquer le surcomptage résiduel. Les erreurs
restantes sont associées à des indices musicaux ambigus et à des propositions
parfois convaincantes malgré K=0 ; leur origine acoustique exacte n'est pas
entièrement identifiée.

La priorité suggérée par cet audit est un objectif natif qui limite ce
compromis sur les petites classes et une vérification de l'appartenance des
attaques au groupe. Ces pistes nécessitent un test dédié ; elles ne sont ni
implémentées ni présentées comme validées ici. Aucun correcteur de sortie,
nouveau seuil, nouvelle époque ou sélection de modèle sur le fold externe.
**V27.3 reste la référence officielle.** Cet audit concerne son composant
natif de comptage, pas la chaîne complète, et reste limité à une graine et
au fold 3 déjà utilisé pour le développement.

Sources et vérifications :

- [Audit complet](v273-low-k-audit.json), [diagnostic et cas](v273-low-k-diagnosis.json), [provenance](v273-low-k-audit-sources.json).
- [Protocole des sondes](v273-low-k-audit-protocol.md).
- Exécution réussie : https://github.com/Andriamarosoa/note/actions/runs/36358846720
- Probabilités des interventions et contextes de toutes les lignes : https://github.com/Andriamarosoa/note/releases/tag/v273-low-k-audit-36358846720

Les dix tests ciblés ont réussi. Les archives, inventaires, correspondances
de lignes et résultats des sondes ont été vérifiés ; les statistiques des
sondes ont été recalculées localement depuis leurs probabilités sauvegardées.
