# V28.0-F — entraînement imbriqué et évaluation sur cinq folds

Protocole fixé le 10 septembre 2026, avant tout score externe V28.0-F.

L'expérience entraîne V28-CHEC harmonique depuis zéro, le compare à V27.3
sur les mêmes 76 768 lignes, puis publie une synthèse des cinq folds.
Elle ne modifie pas la référence officielle et ne déclenche aucun export
ou déploiement.

## Portée et résultat de départ

V28.0-E est terminé : 39,5833 % d'exact-K polyphonique interne pour
l'harmonique, contre 35,8108 % pour le contrôle. Voir le
[rapport interne](v280-internal-ablation-results.md). Cette comparaison
sélectionne l'architecture ; son époque 2 et ses poids ne sont pas transférés.

Les compositions ont déjà été utilisées dans les expériences V10–V27,
V28.0-D et la sélection architecturale V28.0-E. Le présent résultat est une
**évaluation OOF de développement après sélection d'architecture**. Le fold
tenu à l'écart est exclu des gradients et du choix d'époque du modèle qui le
prédit, mais il ne faut pas prétendre que toute la recherche architecturale
ignorait ce fold. Même le fold 0 a eu un groupe inspecté pendant D.

## Partitions figées

La fonction canonique `_balanced_group_folds` de V10.4 assigne les
compositions d'après leurs identités et effectifs. Tous les interprètes et
les versions solo/comp d'une composition restent ensemble.

| Fold externe | Lignes externes | Fold de validation interne | Lignes internes d'ajustement | Lignes de réentraînement |
|---:|---:|---:|---:|---:|
| 0 | 15 846 | 1 | 46 921 | 60 922 |
| 1 | 14 001 | 0 | 46 921 | 62 767 |
| 2 | 15 282 | 0 | 45 640 | 61 486 |
| 3 | 15 823 | 0 | 45 099 | 60 945 |
| 4 | 15 816 | 0 | 45 106 | 60 952 |

Pour chaque fold :

1. choisir comme validation interne le plus petit identifiant parmi les quatre
   folds autorisés ; entraîner sur les trois autres pendant 12 époques ;
2. sélectionner l'époque ayant le plus de comptages polyphoniques exacts en
   validation interne ; départager par NLL polyphonique, puis époque la plus
   ancienne ;
3. créer un nouveau modèle et un nouvel optimiseur, puis entraîner sur les
   quatre folds autorisés pendant ce nombre d'époques, sans validation externe ;
4. figer et sauvegarder le choix, puis effectuer une seule passe d'inférence
   externe ;
5. attendre les cinq résultats avant toute conclusion comparative.

L'agrégation exige exactement 76 768 indices globaux uniques, 240 pistes et
9 401 lignes polyphoniques. Les empreintes de chaque partition, les groupes,
les époques, les nombres de mises à jour et les ordres de lignes sont vérifiés.

## Sources immuables

| Source | Run | Commit | Artefact |
|---|---:|---|---|
| Métadonnées des clusters | 34346175521 | `97c158e8c8b110b85eeac80a336a06768a95b936` | `v280-cluster-metadata` |
| GuitarSet | 34287254337 | `d11f73079a9eb12b2ebce3c8d1df8209b0bd6bdf` | `v272-verified-guitarset` |
| CQT causal | 34323015277 | `3d2fe62b4527236ab91e86dad3cc9c5d81e67b08` | `v280-causal-cqt-outer-clean-cache` |
| Sélection architecturale E | 34351021229 | `7473be3c0a6b4dbb5344ef90744f66a872720f69` | `v280-e-internal-comparison` |
| Référence V27.3 | 34297767492 | `dd43b4f92b234dc7c84377e18389fd34350cd1b4` | `v273-selective-transition-summary` |

Les cinq digests ZIP exacts sont fixés dans `SOURCES` du script. Les runs
doivent être réussis et les artefacts non expirés. Les MD5 GuitarSet, le
SHA-256 du payload compact, les fichiers CQT et le rapport de sélection E
sont également vérifiés. Les prédictions V27.3 sont téléchargées uniquement
par l'agrégateur, après les cinq folds.

SHA-256 de `predictions.npz` V27.3 :
`a353887c6ef1b60d5f6a3854627e99a5cf110e0cb7db0ed4ec50643b26bf805b`.
SHA-256 de son `report.json` :
`53a59f53910484bf86858390c33d53359a72eaa9004275e3ec7f0f0814925564`.

## Représentation et apprentissage

Le graphe harmonique à 110 402 paramètres, les cinq sorties, les poids de
pertes, les crops causaux `24 × 238 × 3`, les masques de supervision auxiliaire
et le traitement de fin d'enregistrement sont ceux validés dans E.
Toutes les lignes gardent leur cible et poids principal Exact-K.

Seed : `28035` ; Adam : `0,0002` ; batch : `32`. Chaque ligne de la partition
d'ajustement apparaît exactement une fois par époque. Le mélange de groupes
et les amorces polyphoniques réutilisent le sampler déterministe de E.
Il n'y a ni pondération de classes, augmentation, teacher, poids externe,
fusion avec V27.3, ni ajustement de seuil après lecture des résultats externes.

Le cache préparé contient les cibles des 240 pistes de développement, séparées
par des indices de rôle vérifiés à chaque chargement. La préparation lit leurs
JAMS pour matérialiser les cibles et références événementielles. Les fonctions
d'entraînement utilisent uniquement les indices autorisés et ne désérialisent
pas le fichier réservé à l'évaluation événementielle. Les noms des 60 pistes
historiques sont indexés uniquement pour vérifier leur exclusion ; leurs JAMS
et audio ne sont pas lus. Locked12 n'est pas indexé.

## Exécution et reprise

Un seul entraînement lourd s'exécute à la fois. Les folds 0 à 4 s'enchaînent
dans cet ordre. Chaque fold contient deux jobs de six époques pour la sélection
interne, puis deux jobs couvrant au maximum six époques chacun pour le refit,
et enfin un job d'inférence. Si le budget choisi est inférieur ou égal à six,
le second job de refit vérifie et préserve l'état complet sans ajouter d'époque.

Les reprises entre jobs restaurent les poids du dernier état, le compteur
de mises à jour et tous les moments Adam. Elles ne repartent pas du meilleur
checkpoint intermédiaire. Un test compare la mise à jour suivante après reprise
à celle d'un entraînement ininterrompu. Un autre exécute les deux phases
complètes sur données synthétiques, contrôle l'absence de lecture des features
externes pendant l'apprentissage et vérifie leur unique lecture finale.

La limite par job d'entraînement est de six heures. Découper en blocs ne réduit
pas le budget scientifique. Les historiques et checkpoints partiels sont
conservés en cas d'échec, mais un état incomplet n'autorise pas le job suivant.
L'exécution complète peut durer plusieurs dizaines d'heures sur CPU.

## Évaluation et interprétation

Métrique principale : exact-K agrégé sur les lignes de vrai K au moins égal à 2.
Publier également les métriques globales, par fold et par vrai K : confusion,
sous-comptes, sur-comptes, NLL et Brier du compteur V28.

Le ranking reste exactement celui des `top_samples` V10 utilisés dans V27.3.
L'évaluation réalise le top-K sans inventer de candidat manquant et conserve
les unissons. Publier F1, précision, rappel et TP/FP/FN à 5, 10, 20 et 50 ms
pour V28, V27.3 et l'oracle vrai K avec ce même ranking. Reproduire avant
comparaison les 4 005 comptages polyphoniques corrects, 62 558 comptages globaux
corrects et les TP/FP/FN V27.3 à 50 ms : `34 404 / 7 859 / 9 816`.

Le delta est apparié par ligne. Son intervalle percentile à 95 % utilise
10 000 rééchantillonnages de pistes complètes avec le seed fixé. Ce bootstrap
décrit l'incertitude conditionnelle sur ce développement ; il n'élimine pas
le biais de sélection d'architecture et ne rend pas le résultat indépendant.

Les trois critères de gain matériel recommandés par la recherche sont
calculés : au moins +5 points polyphoniques face à V27.3, au moins quatre folds
strictement positifs, borne basse du bootstrap supérieure à zéro. Ils sont
publiés même en cas d'échec. **Aucune promotion automatique n'est réalisée.**

Les rapports conservent le nombre de paramètres, la mémoire maximale du
processus, le temps de la première inférence avec compilation et les temps
de lots suivants. Le coût par cluster amorti sur un lot n'est pas une mesure
de latence streaming ; le benchmark de déploiement reste une étape distincte.
La provenance et les mesures du cache CQT sont conservées dans son manifeste.

Artefacts GitHub : cache complet 7 jours, états de reprise 14 jours,
résultats par fold et comparaison finale 90 jours. La synthèse contient les
prédictions OOF appariées, les métriques recalculées et tous les échecs éventuels
aux critères de gain matériel. V27.3 reste la référence à l'issue du workflow.
