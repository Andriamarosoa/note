# NOTE LIVE — S618 à S622, audit des boucles apprises

Date : 15 octobre 2026. **Aucune promotion**. Toutes les branches
expérimentales ci-dessous conservent le checkpoint S614 et le code
historique S80/S397 octet pour octet. Elles n'utilisent ni K oracle ni
annotations lors de l'inférence. Aucune des boucles ci-dessous n'a
ouvert le groupe de test S614 de 22 extraits après la fixation des
sélections.

## Référence à protéger

S614 strict causal, F1 NoteOn hauteur exacte ±80 ms : **69,0277 %** sur
22 extraits historiques déjà publiés, 1 647 TP / 750 FP / 728 FN.
Cette référence mesure la note en émission causale, mais ne compte
pas la latence physique du microphone ou du synthétiseur.

Le set **URMP DEV** réutilisé dans ces expériences vaut **54,5455 %**,
243 TP / 149 FP / 256 FN, sur 5 compositions. Le set **GuitarSet DEV**
vaut **63,1046 %**, 998 TP / 585 FP / 582 FN sur 12 extraits. Ces
deux jeux **ont déjà participé au développement de S614** :
une amélioration sur eux n'est PAS un nouveau test indépendant.

## S618 — tête d'attaque apprise sur les 17 compositions URMP TRAIN

[CI, modèles, prédictions, audit complet](https://github.com/Andriamarosoa/note/actions/runs/38784914444)
[CI causalité et parité streaming](https://github.com/Andriamarosoa/note/actions/runs/38785200514)

Un nouveau classifieur HistGradientBoosting scikit-learn a été
entraîné sur 140 761 exemples, dont 24 064 fenêtres positives, avec
les probabilités des têtes natives NOTE et 65 caractéristiques
acoustiques à bord droit. Aucune source du groupe test S614 n'a été
lue. Checkpoint :
`3a57baff6ab26ba57fe34dc9c521bf2bbdbabaf1688ce5740d3cb1c7a67c63e1`.

- Le contrôle reproduit exactement 243/149/256, F1 54,5455 %.
- `rescue_065` : 385 TP, **1 209 FP**, F1 **36,79 %**.
- `rescue_080` : 346 TP, 627 FP, F1 **47,01 %**.
- `rescue_090` : 285 TP, 269 FP, F1 **54,13 %**.
- L'option gagnante est donc **le S614 inchangé**.

Cause établie : une probabilité d'attaque haute sur plusieurs trames
d'une note tenue ne suffit pas à reconnaître **une nouvelle** attaque.
Le coût supplémentaire de la seule tête, mesuré sur le matériel CI :
p95 **0,859 ms** par groupe de 49 hauteurs (hors calcul S614,
frontend et pilote). Le runtime séparé est strictement causal et le
test de parité bloc/lot passe ; **pas de latence globale certifiée**.

## S619 — apprendre à rejeter un événement NoteOn réellement émis

[CI et audit de 5 plis](https://github.com/Andriamarosoa/note/actions/runs/38785927227)

La sélection native S614 propose un NoteOn, puis une tête événementielle
apprise classe ce NoteOn comme crédible ou non à partir des probabilités
et des caractéristiques acoustiques disponibles **à son émission**.
La suppression d'un NoteOn entraîne aussi la suppression cohérente du
NoteOff correspondant. Cinq modèles, chacun évalué sur une composition
URMP écartée de son propre apprentissage.

Seuil 0,25, fixé par le regroupement hors pli :

| Métrique | S614 | S619 |
|---|---:|---:|
| TP | 243 | 237 |
| FP | 149 | 120 |
| FN | 256 | 262 |
| F1 | 54,5455 % | **55,3738 %** |

[Transfert GuitarSet DEV](https://github.com/Andriamarosoa/note/actions/runs/38786480574),
modèle entraîné sur les 5 URMP DEV, seuil 0,25 fixé :
**998/585/582 → 879/492/701** ; F1 **63,1046 % → 59,5730 %**,
soit **−3,5316 points**. Deux sources progressent, six régressent.
**REJET pour une référence générale : effet de domaine documenté.**

## S620 — exiger une nouvelle montée de probabilité et de fondamentale

[CI et 8 variantes](https://github.com/Andriamarosoa/note/actions/runs/38786241346)

Rejoue la tête S618 avec un détecteur de **transition positive** par
rapport à cinq trames antérieures, une hausse de l'énergie fondamentale
et une hausse de l'attaque native, sans échantillon futur.

- Variante `novelty_090_025_008` : **272 TP, 207 FP, 227 FN**,
  F1 **55,6237 %** (+1,0783 point URMP DEV).
- Trois compositions progressent, deux régressent.
- Les filtres trop permissifs restent négatifs. **Non promu.**

## S622 — synthèse des deux approches NOTE

[CI et audit groupé](https://github.com/Andriamarosoa/note/actions/runs/38786755014)

La nouvelle attaque S620 (probabilité ≥0,90, variation ≥0,25,
hausse physique ≥0,08) est ensuite évaluée par les cinq classifieurs
événementiels OOF de S619. S620 a déjà été sélectionné sur les mêmes
URMP DEV ; **cela reste une nouvelle boucle de développement, et non
une preuve indépendante**.

| Variante URMP DEV | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|
| S614 original | 243 | 149 | 256 | 54,5455 % |
| S620 seul | 272 | 207 | 227 | 55,6237 % |
| S619 seul | 237 | 120 | 262 | 55,3738 % |
| S620 puis S619 seuil 0,15 | 270 | 201 | 229 | 55,6701 % |
| **S620 puis S619 seuil 0,25** | **262** | **176** | **237** | **55,9232 %** |
| S620 puis S619 seuil 0,35 | 236 | 147 | 263 | 53,5147 % |

Quatre compositions progressent, une régresse. Reste insuffisant
pour dépasser 80 % ou remplacer S614. Exemple par source pour la
combinaison 0,25 :

- Jerusalem : 29,85 % → 33,94 % ;
- Nocturne : 56,97 % → 57,46 % ;
- Miserere : 45,68 % → 50,60 % ;
- Hark : 64,19 % → 65,16 % ;
- Jesus : 81,08 % → 78,15 % (régression).

## Défauts persistants et prochaine condition de promotion

1. Les scores URMP des nouvelles sélections ne transfèrent pas encore
   à GuitarSet. L'ancien réseau S614 reste la référence.
2. Un post-filtre peut supprimer des faux positifs mais ne peut pas
   reconstruire une attaque absente : il faut améliorer les **têtes**,
   non seulement leur seuil.
3. La tête `count` strict-LIVE actuelle estime le nombre de notes
   **actives** et non le nombre S80 de **nouvelles notes** d'un
   événement. Le sélecteur S614 a `count_confidence=1.1`, ce qui
   désactive le veto de comptage. Ne jamais présenter son comptage
   comme l'Exact-K historique S80 ou sa promotion.
4. L'entraînement native des cibles S613 marque onset pendant 60 ms
   depuis la référence et sustain seulement à partir de +20 ms,
   alors que la sélection LIVE exige attaque ET sustain.
   Cette discordance temporelle mérite une expérience supervisée
   indépendante avant tout changement d'architecture.
5. Pour promouvoir : entraîner sur des compositions séparées,
   figer les modèles ET les sélections, rejouer une vraie nouvelle
   source indépendante, contrôler par instrument TP/FP/FN,
   NoteOff, délai de disponibilité, calcul p95 et parité causale.
   Aucune statistique sur le test S614 déjà ouvert ne constitue
   une nouvelle validation.

**S623** ajoute un entraînement train-only mixte GuitarSet/URMP pour
mesurer la robustesse inter-domaines sur DEV. Sa promotion est
interdite tant que le transfert et un test nouveau ne sont pas
satisfaisants.


---

## Boucles additionnelles S623–S626 — verdicts confirmés

### S623 : entraînement sur les données TRAIN de GuitarSet et URMP

[CI + poids + rapport des 18 politiques](https://github.com/Andriamarosoa/note/actions/runs/38787299474).
35 enregistrements de TRAIN (18 GuitarSet, 17 URMP),
17 de DEV (12 GuitarSet, 5 URMP), compositions distinctes.

Trois variantes de tête événementielle : URMP seul, guitare seule,
mixte, chacune avec deux poids négatifs et trois seuils.
L'amélioration robuste **n'a pas eu lieu** : seuil bas garde presque
toutes les notes (F1 original GuitarSet **63,1046 %**, URMP
**54,5455 %**), et augmenter le rejet finit par supprimer des notes
correctes. L'entraînement du filtre sur les propositions S614 de
ses propres exemples TRAIN manque probablement de négatifs difficiles
représentatifs des compositions DEV (hypothèse, non preuve causale).

### S624 : huit plis groupés inter-instruments, contrôle sur DEV

[CI groupée](https://github.com/Andriamarosoa/note/actions/runs/38788262813).
Trois familles GuitarSet DEV, cinq compositions URMP DEV ;
chaque pli est exclu de l'apprentissage de sa tête événementielle.
Le plus grand gain sans régression des deux corpus est négligeable :
avec poids négatif 1 / seuil 0,15,
GuitarSet **63,1278 %** (+0,0231 point), URMP **54,5455 %**
(inchangé). Tous les filtres plus agressifs sont régressifs sur
au moins un corpus. **Le filtrage événementiel n'est pas une solution
globale aux défauts d'attaques.**

### S625 : audit direct des vraies notes manquées

[CI + audit individuel des 838 FN de DEV](https://github.com/Andriamarosoa/note/actions/runs/38788514596).
Vérification des probabilités d'attaque `p_onset >= .6` et de maintien
`p_sustain >= .5` sur ±80 ms autour de chaque vraie attaque absente :

| Classe d'échec observée | GuitarSet | URMP | Total |
|---|---:|---:|---:|
| Aucun des deux signaux ne franchit son seuil | 313 | 123 | **436** |
| Maintien franchit seul, attaque insuffisante | 169 | 97 | **266** |
| Attaque franchit seule, maintien insuffisant | 38 | 19 | **57** |
| Les deux franchissent séparément, jamais simultanément | 38 | 8 | **46** |
| Les deux franchissent simultanément, pas de NoteOn apparié | 24 | 9 | **33** |
| **Total** | **582** | **256** | **838** |

**702/838 (83,77 %) n'ont pas de signal d'attaque au seuil dans
la fenêtre considérée.** C'est une observation fondamentale :
changer un seuil, un ordre de sélection ou seulement un NoteOff ne
permettra pas de reconstituer les attaques non proposées par la tête.
La classe de 33 événements avec les deux signaux ne prouve pas à
elle seule une erreur de sélection : états actifs, armement, blocages
temporels et appariement sont encore à auditer individuellement.

### S626 : nouveau réentraînement d'attaque aiguë/rapide et négatifs harmoniques

[CI entraînement / métriques / nouveau checkpoint](https://github.com/Andriamarosoa/note/actions/runs/38789074963).

Entraînement d'une tête supplémentaire HistGradientBoosting sur
**328 275** exemples acoustiques PCM originaux NOTE, dont **30 775**
cibles d'attaque dans les 40 ms après la référence et des négatifs
difficiles autour des attaques d'autres hauteurs. Sources :
17 URMP TRAIN et 18 GuitarSet TRAIN, toutes composition-disjointes
des 5 URMP +12 GuitarSet DEV. Aucun poids S614 ou S80 modifié.
SHA256 tête expérimentale :
`960ede7d58d0a7414655272a29ce80de28a082ab6bed1b340e3c4da5cf141cf1`.

Les huit sélections d'attaque apprises **régressent toutes** par
rapport au contrôle :
- Référence : GuitarSet **63,1046 %**, URMP **54,5455 %** ;
- Variante la moins mauvaise : GuitarSet **61,4495 %**,
  URMP **54,4233 %** (seuil 0,95, montée 0,25, preuve physique 0,08) ;
- Configurations plus permissives : jusqu'à **50,05 %** sur GuitarSet
  et **50,43 %** sur URMP, avec de nombreuses attaques fausses.

**Rejet S626**, même si ses tests et son exécution ont réussi.
Le changement des cibles d'attaque, l'entraînement mixte et les négatifs
difficiles ne compensent pas les faux positifs engendrés lors de l'ajout
de propositions. Aucun score de l'ancien TEST S614 n'a été rouvert
pour sélectionner ces variantes.

## Décision collective fin de S626

- **Référence de fonctionnement inchangée : S614 strict causal,
  69,0277 % F1 NoteOn sur les 22 sources de son test historique.**
- **S80 Exact-K historique conservé inchangé** : il ne faut toujours
  pas confondre son nombre de nouvelles attaques avec le nombre de
  notes actives prédit par la tête `count` autonome de S614.
- Toutes les variantes apprises et sélecteurs ci-dessus restent
  expérimentaux : aucune promotion de S618, S619, S620, S622, S623,
  S624 ou S626.
- Principal défaut prouvé **en développement** : défaut du signal
  d'attaque sur environ 84 % des FN, et non un simple mauvais choix
  d'ordre des sélections. Les boucles montrent aussi une instabilité
  entre URMP et GuitarSet.

### Conception à réviser pour aller plus loin

Plutôt qu'ajouter un nouveau veto ou une sortie binaire entraînée
isolément, il faut désormais expérimenter une **tête multi-tâche
attaques + événement de cardinalité de nouvelles notes + identité
de hauteurs**, entraînée *de bout en bout* avec une fonction de perte
calibrée sur le coût des faux événements harmoniques et l'absence de
vraies attaques. Le contrôleur doit gérer les états NOTE sans futures
trames, et les réattaques simultanées séparément des maintiens.

Toute affirmation de gain devra repartir de compositions inédites,
être gelée avant annotation, mesurer note par note TP/FP/FN, les
offsets et la latence physique, et vérifier que les cas S80 Exact-K
n'ont pas régressé. **Objectif 80 % LIVE : non atteint.**
