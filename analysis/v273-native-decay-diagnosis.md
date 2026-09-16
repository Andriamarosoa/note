# Où se produit la régression avec l'indice de décroissance ?

Diagnostic du 16 septembre 2026 sur les folds **2, 3 et 4** terminés du
[run 35099819151](https://github.com/Andriamarosoa/note/actions/runs/35099819151).
Il complète l'[audit initial de deux folds](v273-native-paired-audit.md).
Le fold 2 a été contrôlé avec le même auditeur : partitions, poids,
calibration interne et décisions finales conformes. Son artefact Actions
est `10467681938`, SHA-256
`1765a218ce0bf5627e90b5c8e1fd57535462ce5e07ca8d404d1c860d7ce4e34b`.

**La régression se concentre dans l'interaction entre les probabilités
apprises avec l'indice et les transitions finales de V27.3. La calibration
interne de ces changements se transfère mal aux folds externes disponibles.**
Ce diagnostic ne démontre pas une cause acoustique particulière.

## Localisation dans la chaîne

Les nombres ci-dessous comptent les bonnes réponses sur les mêmes **5 481
exemples polyphoniques**. Les prédictions intermédiaires utilisent les règles
et seuils déjà figés ; aucun seuil n'a été recherché sur ces exemples.

| Étape | Sans indice | Avec indice | Écart |
|---|---:|---:|---:|
| Fusion basse cardinalité | 2 081 | 2 078 | −3 |
| Sauvetage V27.1, avant les transitions finales | 2 189 | 2 188 | −1 |
| Après les transitions finales V27.3 | 2 210 | 2 170 | **−40** |

Le déficit augmente donc de **39 bonnes réponses à la dernière étape**.
Avec le témoin, ces transitions corrigent 113 cas et en dégradent 92 :
gain net +21. Avec l'indice, elles en corrigent 120 et en dégradent 138 :
perte nette −18. Le supplément de corrections utiles (+7) est dépassé
par le supplément de corrections erronées (+46).

Le défaut de transfert apparaît directement pour le bras avec indice :

| Fold | Gain des transitions sur la calibration interne | Gain sur le fold externe |
|---|---:|---:|
| 2 | +20 | −7 |
| 3 | +36 | −12 |
| 4 | +24 | +1 |

Le score interne a servi à choisir les seuils ; ce n'est pas une estimation
indépendante de leur efficacité. Les trois calibrations emploient le fold
interne 0, avec des réseaux différents.

| Transition | Gain net sans indice | Gain net avec indice | Différence |
|---|---:|---:|---:|
| 2 → 3 | +8 | −8 | −16 |
| 3 → 2 | +1 | −4 | −5 |
| 3 → 4 | +7 | −7 | −14 |
| 4 → 3 | +5 | +1 | −4 |

Les changements vers 3 ou 4 notes expliquent comptablement 30 des 39 bonnes
réponses supplémentaires perdues à cette étape. Les cas réellement K=2
abîmés par 2→3 passent de 18 à 36 ; ceux réellement K=3 abîmés par 3→4
passent de 43 à 71. Ce sont des erreurs de surcomptage observées dans les
prédictions, et non une supposition sur les signaux audio.

## Distinguer réseaux et seuils

Un rejeu croisé conserve les probabilités sauvegardées et échange uniquement
les calibrations internes déjà choisies entre bras. Il inclut le seuil de
sauvetage V27.1 et les quatre seuils de transitions V27.3.

| Probabilités des réseaux | Calibration figée | Bonnes réponses polyphoniques |
|---|---|---:|
| Sans indice | Sans indice | 2 210 |
| Sans indice | Avec indice | 2 197 |
| Avec indice | Sans indice | 2 196 |
| Avec indice | Avec indice | 2 170 |

Sur les probabilités du bras avec indice, remettre les seuils du témoin
remonte de 2 170 à 2 196 : **+26 réponses nettes**, sans réentraînement.
Il reste cependant −14 par rapport au témoin complet. Réciproquement, les
seuils du bras avec indice appliqués aux probabilités témoins perdent 13
réponses. Les effets interagissent : on ne peut pas attribuer une part
causale unique et additive à chaque composant.

La tête V27.2 brute, évaluée uniquement sur les vrais K≥2, passe aussi de
2 893 à 2 868 bonnes réponses. Ce contrôle ne constitue pas un modèle
déployable sur tous les K ; il montre que le problème ne se réduit pas à
changer un seuil tout en déclarant les probabilités améliorées.

## Portée de la conclusion

La cause opérationnelle localisée est une hausse des révisions erronées,
surtout des augmentations de K, quand les décisions de V27.3 reçoivent les
nouvelles probabilités et leurs seuils internes. Les règles existaient déjà
dans V27.3 : « sans correcteur supplémentaire » ne signifiait pas les retirer.

La raison physique pour laquelle cette représentation acoustique conduit à
ces probabilités reste à établir. Les spectres étaient déjà logarithmiques ;
l'indice est une transformation déterministe des mêmes observations. Les
présents résultats ne démontrent ni que la loi de décroissance est fausse,
ni que les résonances sont responsables, ni que tous les indices temporels
sont inutiles. Les limites de sources reconstruites, de validation de la
chaîne complète et de graines restent celles de l'audit initial.

### Complément : un indice positif n'implique pas une nouvelle note

Un contrôle acoustique ultérieur a utilisé le véritable prétraitement V100,
sa quantification float16 et `native_decay_features`, avec des sinusoïdes
continues à fréquences et amplitudes constantes, et phases initiales fixées. Les
fenêtres sont des extraits de ces sons tenus, sans démarrage de note.

| Signal continu, aucune nouvelle attaque | Maximum de l'indice |
|---|---:|
| Une sinusoïde à 220 Hz | environ 0,000001 |
| Deux sinusoïdes à 220 et 233,08188 Hz | 2,8814 |
| Deux sinusoïdes à 110 et 116,54094 Hz | 2,4499 |
| Une fondamentale à 110 Hz et ses sept harmoniques | 0,9930 |

Pour les signaux complexes, 64 configurations de phases initiales
ont été examinées. Ces valeurs sont celles de l'indice, **pas des nombres de
notes prédites**. Les amplitudes 0,1 et 0,5 du rapport machine sont des repères
descriptifs et ne sont pas les seuils de décision du réseau.

Le mécanisme est concret : les composantes présentes dans une même bande
spectrale interfèrent. Sa puissance peut décroître puis remonter alors que
les sons restent continus. Une pente estimée sur les neuf premières fenêtres
peut passer le filtre de fiabilité, puis sous-estimer la puissance suivante.
L'excès positif constitue alors un indice de nouveauté sans nouvelle attaque.
Les caractéristiques fournissent donc au réseau un signal acoustiquement
ambigu pour le comptage des nouveaux événements.

Le [résultat reproductible](v273-native-decay-specificity.json) démontre cette
absence de spécificité de l'indice. Il ne démontre pas encore que les
interférences expliquent chaque erreur de GuitarSet ; le réseau complet n'a
pas été appliqué à ces sons synthétiques.

Un cas réel illustre séparément la décision erronée : fold 3, ligne globale
50105, vrai K=2 et compte avant correction égal à 2 dans les deux bras.
Sans indice, P(2)=0,4574 et P(3)=0,4889 : marge 0,0315, inférieure au seuil
0,1633 ; le compte reste 2. Avec indice, P(2)=0,3831 et P(3)=0,5375 : marge
0,1544, supérieure au seuil 0,1069 ; le compte devient 3 à tort. Cela prouve
le mécanisme de décision sur cette ligne, sans identifier à lui seul sa
cause acoustique.

```sh
PYTHONPATH=.:src python -B scripts/probe_v273_decay_specificity.py \
  --output /chemin/specificite.json
```

Les variantes croisées sont des **diagnostics après observation des résultats**.
Elles ne sont ni sélectionnées pour déploiement ni présentées comme une
nouvelle validation indépendante. Les seuils du run restent inchangés,
les deux autres folds continuent et V27.3 reste la référence.

Reproduction :

```sh
PYTHONPATH=.:src python -B scripts/diagnose_v273_native_decay.py \
  --input-dir /chemin/resultats --folds 2 3 4 \
  --output /chemin/diagnostic.json
```

Les décomptes détaillés et empreintes des sorties sont
conservés dans le [diagnostic machine](v273-native-decay-diagnosis.json).
