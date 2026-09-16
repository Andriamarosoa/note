# Audit V27.3 : indice de décroissance dans les entrées natives

Audit du 16 septembre 2026, limité aux **folds externes 3 et 4 terminés**
du [run 35099819151](https://github.com/Andriamarosoa/note/actions/runs/35099819151).
Producteur : `a192bb3137304b5206fdd45a837995c95f80fcbb`.
Les folds 0, 1 et 2 étaient encore en entraînement au contrôle de l'exécution.

**Verdict technique : scores et décisions sauvegardées reproduits exactement.
Aucun défaut bloquant trouvé dans le calcul de l'indice ou son injection.
Verdict expérimental provisoire : 38 bonnes réponses polyphoniques perdues
avec l'indice, sur 3 820 exemples.** V27.3 reste la référence.

## Résultats recalculés

| Fold | Sans indice de décroissance | Avec indice | Écart en points |
|---|---:|---:|---:|
| 3 | 836 / 1 969 = 42,4581 % | 810 / 1 969 = 41,1376 % | −1,3205 |
| 4 | 768 / 1 851 = 41,4911 % | 756 / 1 851 = 40,8428 % | −0,6483 |
| Cumul disponible | 1 604 / 3 820 = 41,9895 % | 1 566 / 3 820 = 40,9948 % | **−0,9948** |

Il s'agit de l'Exact-K pour les vrais K ≥ 2. Le cumul couvre 10 compositions
et 30 103 clusters, sans doublon. L'indice corrige 61 exemples polyphoniques
et en dégrade 99. Les bonnes réponses globales passent de 24 799 à 24 805
(+6), mais le critère principal préétabli est l'Exact-K polyphonique.
Aucun verdict sur les cinq folds n'est encore possible.

## Contrôles réalisés

- Tailles et SHA-256 des six ZIP Actions téléchargés ; inventaires et
  empreintes des **110 fichiers** des deux résultats et de quatre paires
  d'experts internes partagées.
- **240 pistes, 24 compositions et cinq folds** comparés aux cinq archives
  V27.3 historiques, elles-mêmes vérifiées par SHA-256. Les budgets d'époques
  correspondent exactement à leurs rapports internes.
- Reconstruction des 74 588 étiquettes natives et des identifiants de pistes
  depuis les prédictions des experts partagés : couverture exacte une fois,
  dont 9 354 polyphoniques. Les étiquettes des résultats de comptage concordent.
- Empreintes des partitions recalculées. Pour les réseaux de comptage,
  apprentissage et prédiction sont disjoints ; les poids de classes utilisent
  seulement les étiquettes d'apprentissage.
- **24 entraînements de comptage** contrôlés : poids sauvegardés, historiques
  complets, budgets, graines, paramètres initiaux identiques entre bras,
  pondérations et ordre déterministe déclaré.
- La projection témoin reste nulle. Les normes enregistrées des 12 projections
  du bras avec indice sont positives : l'entrée supplémentaire a bien été
  apprise selon les contrôles du producteur.
- Sélection des seuils V27.1/V27.3 **rejouée à l'identique sur les étiquettes
  internes** ; empreinte du fichier de calibration figé conforme.
- Décisions externes rejouées depuis les probabilités et seuils sauvegardés,
  sans étiquette dans l'interface de décodage. Scores globaux, polyphoniques,
  par K et matrices de confusion conformes.
- Preflight : **36 tests réussis, aucun ignoré**, TensorFlow 2.15.1. Cet audit
  a aussi réexécuté localement 25 tests acoustiques et de protocole avec succès.
- Les 25 archives d'experts et les deux résultats terminés sont présents
  dans la [préversion de sauvegarde](https://github.com/Andriamarosoa/note/releases/tag/v273-native-decay-35099819151).

## Ce que le test mesure

Les deux bras utilisent déjà un spectrogramme logarithmique : le canal
historique est `log1p(power / scalar)`. « Avec/sans ln » était un raccourci
imprécis. Le test ajoute un **indice de nouveauté par rapport à une décroissance
exponentielle estimée**, à partir de ce même spectrogramme.

Le calcul inverse `log1p` par `expm1`, ajuste une droite sur le logarithme de
la puissance relative, puis mesure l'excès positif sur la continuation prévue.
Les neuf fenêtres d'ajustement finissent avant le cluster ; l'observation
reste dans le budget historique de +40 ms. Aucun label ni statistique entre
clusters n'entre dans ce calcul. Les tests couvrent notamment une décroissance
pure, une nouvelle attaque, le silence et la saturation.

L'indice est injecté avant les couches cachées des réseaux V26 uniforme,
V26 pondéré et V27.2 uniforme. Aucun correcteur **supplémentaire** n'est ajouté
en sortie. Les règles de fusion, sauvetage et transitions qui définissent
V27.3 sont conservées et calibrées séparément dans chaque bras. Ce n'est
donc pas une expérience supprimant toutes les règles de décision.

Sur les deux folds, l'écart polyphonique entre bras vaut −1 bonne réponse
après la fusion basse cardinalité, −11 après le sauvetage V27.1 et −38 après
les transitions V27.3. La dégradation s'accentue à la dernière étape. Cela
décrit l'interaction des réseaux et des seuils ; cela n'isole pas une cause
unique. Aucun seuil n'a été modifié d'après ces résultats externes.

## Limites importantes

1. **Pas une reproduction du checkpoint historique.** Les sources ont été
   réentraînées après expiration des anciens artefacts. Les 74 588 clusters
   reconstruits diffèrent des 76 768 historiques ; les populations
   polyphoniques sont 9 354 et 9 401. Le score officiel
   4 005 / 9 401 = 42,6019 % reste distinct. Lui soustraire un score actuel
   ne mesure pas l'effet de l'indice.
2. **Pas une validation de toute la chaîne sur des compositions inédites.**
   Les modèles de proposition V8.x sont reconstruits sur le train historique
   commun, sans réentraînement par fold. Leurs poids peuvent donc dépendre
   de données appartenant aux folds externes. L'isolation des nouveaux experts
   et réseaux de comptage ne rend pas toute la chaîne indépendante. L'écart
   apparié reste une comparaison de développement sur ces candidats communs.
3. **La calibration interne reste une étape de sélection.** L'ancre V10.4
   reprend le protocole historique : les mêmes exemples internes servent à
   la sélection d'époques et à la calibration. De plus, des caractéristiques
   d'apprentissage de cette fusion proviennent d'experts entraînés en incluant
   le fold de méta-validation. Ce niveau interne n'est pas un test indépendant
   supplémentaire. Le fold externe reste exclu des experts V10.1/V10.2 et de
   la fusion concernés, sous la réserve des sources V8.x ci-dessus.
4. **Deux folds et une seule configuration de graines.** Le résultat ne mesure
   pas la variabilité entre réentraînements. Il évalue cette représentation
   de la décroissance, pas l'utilité générale des logarithmes.

L'audit rejoue les décisions depuis les probabilités archivées. Il ne
réexécute pas l'inférence TensorFlow ni le F1 depuis l'audio. Les empreintes
des poids sont vérifiées ; les normes de projection proviennent des rapports
du producteur. L'empreinte d'ordre des exemples est calculée par le programme,
sans trace exhaustive des accès réels aux minibatches. Le test de parité Keras
de deux bras à entrée nulle a réussi dans le preflight.

## Reproduction

Les identifiants et empreintes sont dans le
[manifeste des sources](v273-native-paired-audit-sources.json).
Extraire les résultats sous `fold-3/` et `fold-4/`, et les quatre paires
d'experts sous `experts/`. Conserver les ZIP aux chemins du manifeste.
Les cinq ZIP historiques sont ceux de la configuration figée.

```sh
PYTHONPATH=.:src python -B scripts/audit_v273_native_paired_results.py \
  --input-dir /chemin/resultats \
  --historical-dir /chemin/archives-v273 \
  --config analysis/v273-native-paired-config.json \
  --sources analysis/v273-native-paired-audit-sources.json \
  --output /chemin/audit.json
```

Le [rapport machine](v273-native-paired-audit.json) conserve décomptes et
empreintes. L'agrégation des cinq folds reste attendue. Aucun modèle n'est
promu par cet audit.
