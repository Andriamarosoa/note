# Décision et correction livrée — fold 3 uniquement

**La branche ln est rejetée par la validation interne. Le paquet retenu
revient aux poids et aux seuils du contrôle V27.3 de cette expérience.**
Les fichiers de poids exportés sont identiques, par SHA-256, aux trois
réseaux du contrôle. Les seuils sont également identiques. Le retour est
donc matérialisé par un paquet de modèles et un chargeur d'inférence.

Cette décision applique le choix de revenir à V27.3 lorsqu'une nouvelle
piste n'apporte aucun gain. Elle supprime la régression introduite par ln ;
elle ne constitue pas une amélioration au-delà du contrôle.

## Travail effectué

Une correction de l'architecture a été implémentée : seuls les poids de
la projection ln peuvent apprendre, tandis que tous les anciens poids
et les seuils restent fixes. L'indice est ajouté avant les couches de
comptage ; aucune nouvelle règle n'est ajoutée après leurs sorties.

Les cinq tests ont réussi sous TensorFlow 2.15.1. Le test d'intégration
effectue de vraies mises à jour sur chacun des trois réseaux, confirme
l'identité de tous les anciens poids, la parité des probabilités lorsque
ln=0, puis sauvegarde et recharge les modèles.

Les trois réseaux internes ont ensuite reçu chacun deux entraînements
de branche : étiquettes seules, ou étiquettes mélangées aux probabilités
du contrôle. Trois forces de projection ont été évaluées pour chaque
objectif, soit six candidats complets. Les réglages et critères figuraient
dans le [protocole publié avant l'exécution](v273-frozen-ln-protocol.md).

## La sélection interne rejette les six candidats

La validation interne comporte 15 952 exemples, dont 2 111 polyphoniques.
Les probabilités restent soumises aux seuils du contrôle.

| Objectif de la branche | Force | Bons K polyphoniques |
|---|---:|---:|
| Étiquettes seules | 0,25 | 853 |
| Étiquettes seules | 0,5 | 852 |
| Étiquettes seules | 1 | 850 |
| Étiquettes + contrôle | 0,25 | 853 |
| Étiquettes + contrôle | 0,5 | 853 |
| Étiquettes + contrôle | 1 | 852 |
| Contrôle recalculé dans cet environnement | 0 | **855** |
| Contrôle archivé | 0 | **858** |

Les écarts numériques d'inférence entre CPU déplacent trois décisions
internes situées aux frontières des seuils archivés : indices `3814`,
`13790`, `57261`. La sélection exigeait de dépasser **à la fois** le
contrôle recalculé et le contrôle archivé, sans augmenter le surcomptage.
Ce détail numérique ne décide pas du rejet : les six candidats restent
inférieurs même au contrôle recalculé.

Le choix `control` a été écrit et haché dans `selection.json` **avant**
le calcul des scores externes. Aucun nouvel entraînement final de branche
ni changement de seuil n'a été engagé après ce rejet.

## Résultat sur le fold externe 3

| Variante | Bons K polyphoniques | Exact K polyphonique | Surcomptages polyphoniques |
|---|---:|---:|---:|
| Ancien réentraînement complet avec ln | 810 / 1 969 | 41,1376 % | 298 |
| Contrôle V27.3 de l'expérience | 836 / 1 969 | 42,4581 % | 275 |
| **Modèle retenu et exporté** | **836 / 1 969** | **42,4581 %** | **275** |

- Les **35 nouveaux surcomptages** précédemment introduits par ln sont
  tous supprimés.
- Le retour au contrôle corrige 65 erreurs polyphoniques du bras ln et
  perd les 39 bonnes réponses que ce bras obtenait à sa place : **gain net
  de 26**, soit **1,3205 point** sur ce fold.
- Les deux exemples suivis donnent de nouveau **2 pour la ligne 50105**
  et **3 pour la ligne 42907**.
- Il n'y a aucun changement par rapport au contrôle sur les 8 120 exemples
  où ln vaut zéro, ni sur les autres exemples puisque le contrôle entier
  a finalement été retenu.
- Le score global revient à **12 705 / 15 279**, contre 12 694 pour le bras ln.

Il reste **275 surcomptages polyphoniques** dans le contrôle. La régression
liée à cette piste est corrigée ; le problème général de comptage n'est
pas résolu à 100 %.

## Paquet et utilisation

- [Entraînements et sélection réussis](https://github.com/Andriamarosoa/note/actions/runs/35159220367).
- [Poids, seuils et résultats sauvegardés](https://github.com/Andriamarosoa/note/releases/tag/v273-frozen-ln-35159220367).
- [Résultat machine](v273-frozen-ln-results.json).
- [Toutes les métriques de sélection](v273-frozen-ln-selection.json).
- [Décision exploitable et empreintes du paquet retenu](v273-frozen-ln-decision.json).

Une [exécution indépendante du chargeur](https://github.com/Andriamarosoa/note/actions/runs/35160036336)
a rechargé les **fichiers exportés**, sans entraînement, et recalculé les
trois réseaux sur les 15 279 exemples du fold 3. Toutes les décisions
correspondent exactement au modèle retenu ; l'écart maximal de probabilité
est de 0,000001014. Les 836 bons K polyphoniques et 275 surcomptages sont
reproduits. La [preuve machine](v273-frozen-ln-runtime-verification.json)
identifie chaque poids, la calibration et les entrées vérifiées.

Le fichier `v273-frozen-ln-fold3.zip` de la release contient les trois
fichiers `final/<composant>/model.weights.h5`, `calibration.json`, la
décision figée et les empreintes de tous les fichiers. Son SHA-256 est :

`ae39ffac9b99450d9517233c3bb5b7845abb27e5d4b5222e80f17d760e232c37`.

Le chargeur `scripts/predict_v273_frozen_ln.py` fournit une fonction sans
étiquette de vérité terrain :

```python
from scripts.predict_v273_frozen_ln import predict_native_counts

# Les quatre entrées natives et les comptes de l'ancre V104 existante.
counts, probabilities, provenance = predict_native_counts(
    native_inputs, anchor_counts, bundle_dir="modele_extrait"
)
```

Ces fichiers constituent les réseaux de comptage et leur calibration ;
les étapes existantes de construction des candidats et de l'ancre V104
fournissent toujours leurs entrées. L'environnement vérifié utilise
Python 3.11, TensorFlow 2.15.1 et NumPy 1.26.4.

## Portée

Seul le fold externe 3 a été examiné. Il avait déjà servi au diagnostic ;
cette vérification est donc un résultat de développement. Aucun meilleur
score officiel, aucune validation sur tous les folds et aucune disparition
de toutes les erreurs ne sont annoncés.

Le score historique V27.3 de 42,6019 % concerne une autre population.
Le contrôle utilisé ici est celui de l'expérience à sources reconstruites :
ces deux nombres ne doivent pas être confondus.
