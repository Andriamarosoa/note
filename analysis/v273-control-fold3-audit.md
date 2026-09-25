# Audit des erreurs du contrôle V27.3 — fold 3

Suite vérifiée : [correction native des positions et audit des limites restantes](v273-exact-timing-results.md).

**L'audit a identifié un défaut concret de reconstruction temporelle dans les
données, des décisions correctes bloquées par le routage, et des erreurs propres
aux propositions des réseaux. Aucun de ces constats ne suffit à expliquer seul
les 1 133 erreurs polyphoniques.**

Le défaut des données a fait l'objet d'un deuxième audit, puis d'une vérification
des deux divergences qui subsistaient. Les poids, les seuils et les spectrogrammes
du contrôle restent ceux du paquet déjà publié. Aucun gain de modèle n'est annoncé.

## Périmètre et preuves

- [Premier audit réussi](https://github.com/Andriamarosoa/note/actions/runs/36105248716).
- [Audit de suivi réussi, six tests passés](https://github.com/Andriamarosoa/note/actions/runs/36106051196).
- [Archives, métadonnées et listes de cas](https://github.com/Andriamarosoa/note/releases/tag/v273-control-fold3-audit-36106051196).
- [Décisions, métriques et ablations](v273-control-fold3-decision-audit.json).
- [Annotations et entrées](v273-control-fold3-input-audit.json).
- [Reconstruction temporelle](v273-control-fold3-alignment-audit.json).
- [Vérification des deux divergences restantes](v273-control-fold3-alignment-residual-audit.json).
- [Registre des problèmes et limites](v273-control-fold3-audit-status.json).

Les 15 279 décisions du contrôle sont reproduites exactement par un décodeur
scalaire indépendant, par les fonctions de production et depuis les probabilités
du paquet exporté. Les 81 fichiers des deux paquets sont vérifiés. Les huit
archives sources sont contrôlées par empreinte ; seules les entrées et sorties du
fold externe 3 sont analysées. Les codes V91, V92 et V100 inspectés sont identiques
à ceux du producteur de ces données, `8aba7737d4112671beeaedac92329d533fcfc316`.

**Correction de l'explication précédente : K compte ici des débuts de notes
attribués à un groupe de candidats, pas toutes les notes qui résonnent en même
temps.** Le groupe couvre au plus 40 ms ; une annotation est affectée au groupe
ayant le candidat le plus proche, à au plus 20 ms. La classe 6 représente 6+.
Il n'existe aucune cible supérieure à 6 sur ce fold.

## 1. Défaut temporel confirmé et audité

Le cache conserve des positions relatives et quelques positions absolues de
candidats bien classés. `_recover_cluster_start` recherche une origine expliquant
ces positions. Lorsque plusieurs origines les expliquent aussi bien, son dernier
critère choisit celle la plus proche de zéro. Cette égalité ne prouve pas que
l'origine choisie soit la vraie.

Le test minimal contient des candidats aux échantillons 1000 et 1200, avec le
meilleur à 1000. La reconstruction ancienne retourne une origine **800** tout en
déclarant les six positions sauvegardées parfaitement retrouvées. Les 1 765
positions relatives entières possibles survivent exactement à float16 : ce cas
vient du choix de l'origine et non d'un arrondi des positions relatives.

Dans le fold 3 :

- **187 groupes** admettent plusieurs origines compatibles avec les seules
  positions des meilleurs candidats.
- **80 origines retenues** sont incompatibles avec la position du candidat de
  score maximal sauvegardé. Les décalages vont de 5 à 1 265 échantillons, soit
  environ 0,11 à 28,68 ms.
- Une contrainte utilisant ce score, sans annotation, identifie une origine
  unique pour chacun des 15 279 groupes de ce fold. Elle modifie ces 80 origines.
- Les **62 violations d'espacement des groupes non tronqués** disparaissent.
- Les divergences entre comptes d'origine et comptes reconstitués passent
  de **16 à 2**.

Le code de production des spectrogrammes utilise précisément l'origine
reconstituée. Il s'agit donc d'un défaut de préparation des entrées et de certaines
cibles d'occupation des cordes. Les 80 groupes décalés ne contiennent que huit
groupes polyphoniques, dont quatre mal comptés par le contrôle. On ne peut pas
attribuer l'ensemble des erreurs du modèle à ce défaut, ni annoncer quatre
corrections sans recalculer les entrées et tester le modèle.

### Audit des deux divergences restantes

Les lignes `58991` et `58992` concernent le même passage. La première comportait
**57 candidats**, puis en a conservé 48. Son dernier candidat d'origine se situe
à l'échantillon **807820**, tandis que le dernier conservé se situe à **807500**.
Le début de note annoté à **807819** est alors à 319 échantillons du candidat
conservé, mais à 284 du groupe suivant : il change de groupe lors de la
reconstruction des cibles.

La largeur originale du groupe, déjà sauvegardée, retrouve le candidat final à
807820 sans consulter les annotations. La distance redevient **un échantillon**
et la note revient au groupe d'origine. La vérification réattribue les **8 812
débuts de notes déjà affectés** : les divergences passent **16 → 2 → 0**, sans
nouvelle divergence et en conservant le total. Les 454 annotations non attribuées
par le premier audit ne sont pas rejouées dans cette dernière vérification locale.

La correction robuste à préparer est de conserver dès la génération du cache
les débuts de groupes et les positions absolues entières, avec les candidats
complets pour les cibles et les candidats retenus séparément pour les entrées.
La fenêtre spectrale doit employer le début original du groupe. La reconstruction
diagnostique par les scores n'est pas promue comme remplacement général.

## 2. Les 1 133 erreurs de comptage sont toutes localisées

Cette partition est exhaustive et sans chevauchement. Elle utilise la vérité
terrain pour analyser les erreurs ; elle ne constitue pas un sélecteur utilisable
en production.

| Situation dans une erreur polyphonique | Cas |
|---|---:|
| Le spécialiste propose le bon K, mais la base vaut 0 ou 1 | 266 |
| Le spécialiste propose le bon K, mais la transition n'est pas autorisée | 12 |
| Le spécialiste propose le bon K, mais sa marge n'atteint pas le seuil | 99 |
| Seul un réseau de comptage complet propose le bon K | 121 |
| Aucun des trois réseaux de comptage ne propose le bon K | 635 |
| **Total** | **1 133** |

Il existe donc une bonne proposition dans **498 erreurs**, mais la reconnaître
sans annotation reste un problème d'apprentissage. Dans les **635 autres cas**,
changer uniquement le choix entre les trois propositions ne suffit pas. Parmi
les 858 sous-comptages finaux, les trois réseaux sous-comptent ensemble 463 cas.

Sur les 130 groupes de K=5 ou K=6, seuls neuf sont corrects. L'apprentissage
final comporte 355 cas K=5 et 89 cas K=6, contre 39 652 cas K=0. Ce déséquilibre
est mesuré ; son rôle causal précis n'a pas été établi par un nouvel entraînement.

## 3. Les effets secondaires des règles ont été audités

| Étape | Bons K polyphoniques | Sous-comptages polyphoniques | Surcomptages polyphoniques |
|---|---:|---:|---:|
| Ancre V104 | 789 | 956 | 224 |
| Fusion des petits K | 790 | 954 | 225 |
| Sauvetage V27.1 | 829 | 909 | 231 |
| Transitions V27.3, contrôle retenu | **836** | **858** | **275** |

Le sauvetage gagne 39 bons comptes polyphoniques à cette étape, mais dégrade
88 groupes auparavant corrects de K=0 ou K=1. C'est cohérent avec son critère
interne, qui donne priorité au score polyphonique : un effet mesuré de l'objectif
de sélection, et non une erreur de calcul.

Les transitions finales corrigent 69 erreurs et dégradent 62 réponses correctes,
soit un gain net de sept. Elles créent 51 nouveaux surcomptages et 11 nouveaux
sous-comptages, tout en supprimant sept surcomptages et 62 sous-comptages.
Par exemple, la transition 3→4 corrige 49 comptes et en dégrade 42. La supprimer
réduirait le surcomptage mais ferait perdre sept bons comptes sur ce fold.

L'ablation qui remplace tous les K prédits >=2 par le spécialiste donne
**770 bons comptes polyphoniques**, contre 836 : elle corrige 111 erreurs mais
dégrade 177 bonnes réponses. Utiliser ce spécialiste partout est invalide pour
K=0/1, puisqu'il ne peut produire que 2 à 6. Les protections ne peuvent donc pas
être simplement retirées sur la base des 498 propositions favorables observées.

Aucun seuil n'a été choisi à partir de ces résultats externes.

## 4. Hypothèses vérifiées mais non établies comme causes

- Les entrées sont finies, tous les spectrogrammes sont non nuls et aucun
  masque de candidats n'est vide : pas de corruption générale détectée.
- Dans la géométrie des entrées actuelles, 43 groupes polyphoniques comportent
  un début de note attribué hors de la fenêtre spectrale ; 33 sont mal comptés.
  Les autres caractéristiques des candidats restent disponibles. Cette
  association ne prouve pas une absence de toute information acoustique et
  laisse **1 100 erreurs polyphoniques hors de ce groupe**.
- 34 groupes polyphoniques ont moins de candidats conservés que leur K réel ;
  32 sont sous-comptés et deux sont corrects. Le nombre de candidats n'impose
  donc pas à lui seul un plafond dur au K prédit par les réseaux de comptage.
- Aucun double début sur une même corde n'est trouvé dans l'attribution
  reconstituée de ce fold.
- Aucune séparation de sources, écoute contrôlée ou ablation acoustique ne
  permet ici d'imputer les erreurs restantes aux harmoniques ou aux résonances.

## Décision et suite justifiée

Le diagnostic des deux anomalies de données est terminé. Leur correction dans
la génération des caches doit précéder un nouvel entraînement et être auditée
sur les positions exactes, les cibles et les fenêtres produites. Les anciens
artefacts restent immuables pour conserver une comparaison traçable.

Le contrôle reste à **836 / 1 969 = 42,4581 %**, avec 275 surcomptages et
858 sous-comptages. Les causes profondes des mauvaises probabilités des réseaux
ne sont pas entièrement établies. Le fold 3 a déjà servi au diagnostic : toute
correction ultérieure doit être choisie en interne et son évaluation sur ce fold
présentée comme du développement, sans nouveau score officiel revendiqué.
