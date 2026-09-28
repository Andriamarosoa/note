# Échelle spectrale fixe : gain polyphonique partiel, régression sur K=1

**Les deux entraînements de 12 époques et leur audit sont terminés.**
Remplacer la normalisation locale par une division fixe par 12 fait passer
l’Exact K polyphonique interne de **26,53 % à 28,75 %**, soit 47 comptes
exacts supplémentaires sur 2 111. Mais K=1 perd 95 comptes exacts et le
score global recule de 11 comptes. **Le changement ne satisfait donc pas
le critère annoncé de non-régression sur K=1, 2 et 3.**

Le résultat est mixte. Il ne justifie ni une promotion du modèle ni
l’affirmation que la normalisation était la cause générale du mauvais
comptage. La référence officielle V27.3 à **42,6019 %** reste inchangée ;
ce score de la chaîne complète n’est pas comparable directement à ce test
du composant natif sur sa validation interne.

## Comparaison principale à 12 époques

Deux modèles frais, sans pondération ni correcteur, avec les mêmes poids
initiaux communs, données, lots, objectif et graines. Le seul traitement
modifié remplace LayerNormalization sur les trois canaux spectraux par
leur division fixe par 12, retirant les six paramètres gamma/beta de cette
normalisation. Le fold externe 3 n’est pas évalué.

| Mesure | Normalisation actuelle | Échelle fixe | Différence |
|---|---:|---:|---:|
| Exact K poly, apprentissage | 2 102/5 274 = 39,86 % | 2 099/5 274 = 39,80 % | −3 comptes |
| Exact K poly, validation | 560/2 111 = 26,53 % | 607/2 111 = 28,75 % | **+47 ; +2,23 points** |
| Exact K global, validation | 12 989/15 952 = 81,43 % | 12 978/15 952 = 81,36 % | −11 ; −0,07 point |
| Exact K pour K=0 à 3, validation | 12 957/15 428 = 83,98 % | 12 922/15 428 = 83,76 % | **−35 comptes** |

En polyphonie, 153 erreurs sont corrigées et 106 bonnes décisions sont
perdues : le solde est +47. Le gain concerne quatre des cinq compositions
internes ; la cinquième perd neuf comptes exacts.

| K vrai | Groupes de validation | Exacts avec normalisation | Exacts avec échelle fixe | Solde |
|---:|---:|---:|---:|---:|
| 0 | 10 763 | 10 369 | 10 406 | +37 |
| 1 | 3 078 | 2 060 | 1 965 | **−95** |
| 2 | 993 | 325 | 347 | +22 |
| 3 | 594 | 203 | 204 | +1 |
| 4 | 374 | 28 | 52 | +24 |
| 5 | 115 | 4 | 4 | 0 |
| 6 | 35 | 0 | 0 | 0 |

K=1 passe de 66,93 % à 63,84 %. K=2 progresse de 32,73 % à 34,94 % ;
K=3 reste presque inchangé, de 34,18 % à 34,34 %. Le gain polyphonique
provient donc principalement de K=2 et K=4, sans résolution du déficit
général des petits K.

## Audit de la régression sur K=1

Les 95 comptes perdus sont un **solde**, pas le nombre de nouveaux échecs.
Sur les 3 078 vrais K=1 :

- **204 anciennes réponses correctes deviennent fausses** : 115 sont
  prédites à 0, 88 à 2 et une à 3.
- **109 anciennes erreurs deviennent correctes** : 62 venaient de 0,
  46 de 2 et une de 3.

Tous changements réunis, les vrais K=1 prédits à 0 passent de 735 à 794
(+59). Ceux prédits à 2 ou plus passent de 283 à 319 (+36).
**La dégradation associe donc sous-comptage et surcomptage.** Elle ne peut
pas être présentée comme une simple tendance à prédire davantage de notes.
K=0 désigne ici l’absence de nouvelle attaque assignée, pas nécessairement
un signal audio silencieux.

Pour l’ensemble K=0 à 3, les surcomptages passent de 813 à 841 (+28) et
les sous-comptages de 1 658 à 1 665 (+7). Le test ne résout donc pas non plus
l’objectif initial de réduction des erreurs pour les petits K.

## Ce que ce test établit sur la cause

La normalisation influence les décisions apprises : son remplacement
produit un gain polyphonique limité, accompagné de régressions mesurées.
Cependant, les scores polyphoniques sur les exemples déjà vus sont presque
identiques (39,86 % et 39,80 %). Préserver les valeurs des canaux à cette
entrée **ne suffit pas à résoudre la difficulté d’apprentissage du compte**
dans ce protocole.

Ce test ne sépare pas l’effet de la préservation des amplitudes de celui
du conditionnement numérique ou du retrait des paramètres gamma/beta.
Les transitions de classes localisent les échecs de décision ; elles ne
prouvent pas leur cause acoustique individuelle. On ne peut notamment pas
attribuer tous les échecs aux harmoniques ou au niveau sonore.

## Les points intermédiaires ne remplacent pas le verdict

| Époque | Normalisation : Exact K poly validation | Échelle fixe : Exact K poly validation | Solde |
|---:|---:|---:|---:|
| 4 | 400/2 111 = 18,95 % | 389/2 111 = 18,43 % | −11 |
| 8 | 411/2 111 = 19,47 % | 428/2 111 = 20,27 % | +17 |
| 12, critère principal | 560/2 111 = 26,53 % | 607/2 111 = 28,75 % | +47 |

Le constat provisoire à quatre époques est ainsi remplacé par le résultat
à douze prévu au protocole. Une seule graine et une seule validation
interne déjà explorée ne permettent pas de revendiquer un gain statistique
généralisable. Aucun nouvel entraînement n’est lancé par cette publication.

## Vérification et preuves

Les quatre archives finales ont été vérifiées par SHA256 et inventaire
complet. Les **355 854 lignes de probabilités** des deux modèles, des deux
partitions et des trois checkpoints ont été rescorrées. Le diagnostic et
le rapport automatique sont reproduits exactement en local. Les identités,
partitions, poids initiaux communs, ordres des lots, compteurs d’Adam,
empreintes des checkpoints et métriques de validation concordent.

- [Exécution réussie](https://github.com/Andriamarosoa/note/actions/runs/36490993251).
- [Archives et rapport automatique](https://github.com/Andriamarosoa/note/releases/tag/v273-normalization-36490993251).
- [Protocole fixé avant résultats](v273-normalization-protocol.md).
- [Diagnostic complet et transitions par K](v273-normalization-final-diagnosis.json).
- [Provenance et empreintes vérifiées](v273-normalization-final-sources.json).

Après extraction vérifiée des archives dans leurs dossiers respectifs :

```sh
PYTHONPATH=.:src python -B scripts/audit_v273_normalization_pair.py \
  --root /path/to/results --preflight /path/to/results/norm-preflight \
  --config analysis/v273-native-paired-config.json \
  --output /path/to/recomputed-verdict
```
