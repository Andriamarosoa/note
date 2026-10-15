# S644 — Boucle d'audit des portes de candidats d'attaque faible

## Point de départ figé

S642 a réellement entraîné une tête d'attaque, mais le rejeu
S643 a constaté **0** tentative de rescue sur 12 GuitarSet DEV
et 5 URMP DEV, sans aucune amélioration du F1. La tête de latence
S642 a uniquement gagné **+0,57 point F1 causal GuitarSet**,
sans gain URMP ni changement du F1 historique.

Le modèle original NOTE S614, S80, S632 et les checkpoints
appris par S642 ne sont PAS modifiés. Empreintes :
- S614 : `7ccf046e3a1870eaca3a108e6f9565298aeeadf43732a2f83559542c03f5f17f`
- tête d'attaque S642 : `2f7364865803bb0cab8c564911e25574203b129169f6abf5f4ecdf14acc7f256`

## Hypothèse soumise à mesure

La politique S642 exige, pour une candidate inactive :
1. attaque existante 0,13 ≤ onset original < seuil S614 ;
2. probabilité de maintien du S614 ≥ 0,26 ;
3. score q de la nouvelle tête ≥ 0,965 ;
4. seule la meilleure candidate de la trame est élevée au seuil
   d'attaque original avant d'entrer dans le sélecteur causal S614 ;
5. le sélecteur peut encore rejeter (confirmation, réarmement,
   réfractaire, plafond de polyphonie).

Il faut compter chaque étage individuellement. Un score q élevé
n'est **pas** synonyme de NoteOn émis ; une tentative n'est
**pas** une correction.

## Expérience S644

[Workflow train-lock-dev](https://github.com/Andriamarosoa/note/actions/workflows/v273-s644-gate-anatomy-rescue.yml)

Protocole défini **avant ouverture DEV** :
- TRAIN authentique : 17 URMP + 18 GuitarSet ;
- vérifier composition disjointe de DEV et absence de sources TEST ;
- pour chaque source, calculer q du checkpoint S642 original et
  compteurs de filtrage de chacun des seuils ;
- relecture de **six variantes prédéfinies** : S614, q≥0,50,
  q≥0,70, q≥0,85, q≥0,95 et q≥0,965 original, en gardant
  les contraintes acoustiques de S642 identiques ;
- pour chaque variante : nombre d'essais, vrais NoteOn ajoutés,
  vrais positifs, faux positifs, faux négatifs et enregistrements
  corrigés/régressés ;
- verrouiller la meilleure variante **sur TRAIN uniquement**
  si elle améliore les deux corpus sans trop de régressions ;
  sinon verrouiller S614 ;
- seulement après, calculer les mêmes statistiques sur les
  17 DEV sans modifier la règle déjà choisie.

Cette grille n'entraîne **aucun nouveau poids**. Les seuils
testés ne deviennent pas automatiquement une nouvelle référence,
et les statistiques complètes DEV constituent un audit exploratoire,
pas un autre jeu de sélection valide. Toute promotion exige ensuite
une évaluation indépendante sur de nouveaux morceaux et des mesures
de causalité/latence LIVE.

## Résultats vérifiés — S644 terminé

[GitHub Actions S644 réussi](https://github.com/Andriamarosoa/note/actions/runs/38817435779).
[Artifact JSON des 35 sources TRAIN, 17 DEV et six variants](https://github.com/Andriamarosoa/note/actions/runs/38817435779/artifacts/12011601672).

**Défaut de conception prouvé :** la tête q S642 a des valeurs
très élevées sur les notes à faible probabilité S614, mais le
filtrage acoustique original exige simultanément onset>=0,13
et sustain>=0,26. Après ces deux portes, q ne dépasse
**jamais 0,965** : maximum **0,95892 TRAIN**
sur 22 861 états admissibles et **0,96232 DEV**
sur 10 952 états admissibles. L'exigence
q≥0,965 ferme ainsi *mathématiquement* toutes les
nouvelles admissions de la version S642 sur ces audios.

Statistiques q parmi ces cas acoustiquement admissibles :
TRAIN médiane 0,638, p99 0,925 ; DEV médiane 0,610,
p99 0,931. Sur les cas d'attaque faible avant garde,
q dépasse parfois 0,99 ; les portes acoustiques et la
tête q ne sont **pas calibrées conjointement**.

### Expériences sur TRAIN, mêmes poids

| q minimum | GuitarSet TP/FP/FN | F1 | URMP TP/FP/FN | F1 |
|---|---|---:|---|---:|
| S614 original | 2072/898/700 | **72,17 %** | 1622/541/446 | **76,67 %** |
| 0,50 | 2247/1907/525 | 64,89 % | 1797/1492/271 | 67,09 % |
| 0,70 | 2206/1633/566 | 66,74 % | 1754/1194/314 | 69,94 % |
| 0,85 | 2133/1232/639 | 69,51 % | 1670/762/398 | 74,22 % |
| 0,95 | 2073/899/699 | 72,18 % | 1622/541/446 | 76,67 % |
| 0,965 | 2072/898/700 | 72,17 % | 1622/541/446 | 76,67 % |

0,50 ajoute 175 TP nets GuitarSet mais 1009 FP nets :
la précision s'effondre. 0,95 ajoute **1 TP et 1 FP**
sur TRAIN GuitarSet et ne change rien sur TRAIN URMP.
Aucune politique ne respecte simultanément les
conditions d'amélioration sur les deux corpus.
**Le verrouillage TRAIN a choisi S614.**

### DEV réévalué sans changer le verrouillage TRAIN

| q minimum | GuitarSet TP/FP/FN | F1 | URMP TP/FP/FN | F1 |
|---|---|---:|---|---:|
| S614 original | 998/585/582 | **63,10 %** | 243/149/256 | **54,55 %** |
| 0,50 | 1130/1424/450 | 54,67 % | 315/483/184 | 48,57 % |
| 0,70 | 1090/1221/490 | 56,03 % | 301/391/198 | 50,55 % |
| 0,85 | 1038/892/542 | 59,15 % | 257/237/242 | 51,76 % |
| 0,95 | 998/588/582 | 63,04 % | 243/149/256 | 54,55 % |
| 0,965 | 998/585/582 | 63,10 % | 243/149/256 | 54,55 % |

À q≥0,85 DEV, **689 tentatives GuitarSet / 640 NoteOn
réellement assistés** mais seulement +40 TP nets avec
+307 FP. Sur URMP, 177 tentatives / 171 NoteOn assistés,
+14 TP contre +88 FP nets. Aucun avantage robuste.

La boucle S644 montre donc que « baissez le seuil »
ne constitue pas une solution. Le prochain test S645
doit examiner la **nouveauté temporelle** de q et les
garde-fous d'attaque physique, au lieu d'émettre toute
note dont q reste simplement élevé. S80/S614/S632
et les poids S642 restent inchangés, pas de promotion.
