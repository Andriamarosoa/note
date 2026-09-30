Audit du 30 septembre 2026 : chevauchement des notes et succession des attaques

**Le signal temporel le plus net est la difficulté à compter de nouvelles attaques quand des notes antérieures sonnent encore. Le simple décalage entre les nouvelles attaques n'explique pas, à lui seul, les échecs polyphoniques.** Ce constat est une association dans les données existantes, pas une preuve de causalité.

Cet audit porte sur le [run terminé 36688979041](https://github.com/Andriamarosoa/note/actions/runs/36688979041), époque 12 fixée dans son protocole : 15 952 groupes de validation interne, 50 pistes, cinq compositions. Le bras `observed_only` est le témoin principal ; `with_residual` sert de contrôle de robustesse sur les mêmes groupes. Aucun entraînement ni changement de modèle n'a été effectué pour cet audit. Ce ne sont pas les résultats de l'expérience high pitch rescue.

La métrique est **K correctement compté** : le nombre de nouvelles attaques assignées au groupe. Elle ne vérifie ni l'identité des hauteurs ni la durée des notes. Une note qui continue de sonner n'ajoute pas une nouvelle attaque à K. Les anciennes étiquettes 0…6 du run sont conservées pour l'analyse ; son K=0 ne signifie pas silence. La convention future K=-1 n'est pas appliquée rétroactivement.

Il faut distinguer deux phénomènes :

- **Départs presque simultanés ou successifs** : écart entre la première et la dernière attaque du groupe. Le seuil descriptif principal est 10 ms, avec contrôle à 5, 20, 30 et 40 ms.
- **Chevauchement de durées** : plusieurs notes sont encore actives au même instant, d'après leurs débuts et fins annotés. Des attaques successives peuvent donc produire un chevauchement. On distingue aussi les notes du groupe des notes antérieures, assignées ailleurs ou non assignées, qui sonnent encore au début de ce groupe.

**Les 2 111 groupes K≥2 présentent tous un chevauchement entre toutes leurs nouvelles notes.** Même le plus court intervalle commun dure 11,61 ms ; sa médiane est 169,21 ms. Il n'existe donc aucun groupe K≥2 réellement sans chevauchement dans ce lot. Il serait incorrect de présenter cette validation comme un test équilibré « polyphonie chevauchée contre notes successives sans chevauchement ». Aucun de ces groupes ne contient plusieurs nouvelles attaques sur une même corde annotée : l'étalement des départs décrit ici des notes sur des cordes différentes.

Le compteur reste faible sur cette population : 622/2 111 groupes corrects (29,46 %) pour le témoin, 643/2 111 (30,46 %) avec résidu. Parmi les 1 489 erreurs du témoin, 1 320 sont des sous-comptages, soit 88,65 %. Avec résidu : 1 339 sous-comptages sur 1 468 erreurs, soit 91,21 %. Le problème dominant n'est donc pas simplement de compter toutes les résonances comme des nouvelles notes supplémentaires.

Quand une ou plusieurs notes **antérieures** sonnent encore à la première nouvelle attaque, le taux de réussite baisse. Les effectifs figurent entre parenthèses ; chaque pourcentage est le taux de K exact du témoin.

| Nouvelles attaques attendues | Aucune note antérieure encore active | Au moins une note antérieure encore active |
|---|---:|---:|
| K=1 | 87,25 % (1 623) | 37,32 % (1 455) |
| K=2 | 43,75 % (256) | 35,14 % (737) |
| K=3 | 51,18 % (170) | 24,29 % (424) |
| K=4 | 19,51 % (164) | 9,52 % (210) |
| K=5 | 4,17 % (24) | 8,79 % (91) |
| K=6 | 0 % (6) | 0 % (29) |
| Ensemble K≥2 | 37,42 % (620) | 26,16 % (1 491) |

Les 1 491 groupes avec des notes antérieures représentent 70,63 % des groupes K≥2 et 1 101 des 1 489 erreurs polyphoniques. Cette concentration ne mesure pas le nombre d'erreurs causées par ces anciennes notes : le groupe de comparaison est différent.

Le très grand écart brut en K=1 est notamment influencé par le type de jeu. En accompagnement (`comp`), les taux sans/avec anciennes notes sont 48,86 % / 25,66 % ; en solo, 91,91 % / 79,18 %. Mélanger ces populations amplifie fortement l'écart. Nous avons donc donné le même poids aux mêmes catégories dans les deux conditions.

| Comparaison standardisée | Groupes retenus | Témoin : avec moins sans anciennes notes | Avec résidu : même différence |
|---|---:|---:|---:|
| K≥2 : même répartition de K | 2 111 / 2 111 | −13,13 points | −13,45 points |
| K≥2 : même K, composition et type de jeu | 1 943 / 2 111 | −9,58 points | −9,28 points |
| K≥2 : même K et piste | 862 / 2 111 | −8,42 points | −9,69 points |
| K≥2 : même K, composition, jeu et classe d'écart avec l'attaque précédente | 1 088 / 2 111 | −9,82 points | −7,03 points |
| K=1 : même composition et type de jeu | 3 078 / 3 078 | −13,54 points | −12,23 points |
| K=1 : même composition, jeu et classe d'écart avec l'attaque précédente | 2 225 / 3 078 | −9,53 points | −6,74 points |

La standardisation utilise les poids de la population regroupée, uniquement dans les catégories ayant au moins cinq cas de chaque condition. Les groupes exclus et les effectifs par catégorie sont conservés dans le JSON. Ces lignes n'ont pas toutes la même population et ne sont pas des estimations causales interchangeables. Le contrôle par piste conserve notamment moins de la moitié des groupes K≥2.

L'association ne dépend pas uniquement des attaques très proches d'une frontière de groupe : en exigeant que l'ancienne note ait démarré au moins 20 ms plus tôt, les taux bruts K≥2 restent 37,33 % sans / 25,90 % avec ; après standardisation K/composition/jeu, l'écart est −9,71 points. Les contrôles à 50 et 100 ms donnent également des taux bruts plus faibles avec anciennes notes. Les résultats par composition, par présence de grave et par étalement des attaques sont disponibles dans le JSON.

À l'inverse, **les attaques successives dans un groupe ne sont pas uniformément plus difficiles que les départs presque simultanés**. Les notes des deux colonnes ci-dessous se chevauchent en durée.

| K | Départs étalés sur ≤10 ms | Départs étalés sur >10 ms |
|---|---:|---:|
| 2 | 32,56 % (562) | 43,62 % (431) |
| 3 | 26,49 % (185) | 34,47 % (409) |
| 4 | 28,57 % (70) | 10,53 % (304) |
| 5 | 25 % (4) | 7,21 % (111) |
| 6 | 0 % (3) | 0 % (32) |
| Tous K≥2 | 30,70 % (824) | 28,67 % (1 287) |

Les attaques étalées contiennent davantage de K élevés. À répartition de K identique sur les catégories suffisamment représentées (K=2,3,4 ; 1 961 groupes), leurs taux deviennent 34,54 %, contre 29,96 % pour les départs rapprochés. En contrôlant aussi composition et jeu : avantage de 5,02 points pour les départs étalés, sur 1 902 groupes. Cependant K=4 présente l'effet inverse, les K=5/6 rapprochés sont trop rares et le signe dépend aussi du seuil choisi. Il ne faut pas transformer cet audit en règle « étaler les notes améliore le modèle ».

| Seuil de séparation proche/étalé | Témoin : étalé moins proche, à répartition de K identique |
|---|---:|
| 5 ms | −2,34 points |
| 10 ms | +4,58 points |
| 20 ms | +8,47 points |
| 30 ms | +3,55 points |
| 40 ms | −0,70 point |

Le seuil de 40 ms ne laisse que 43 groupes étalés. Ces contrôles ne montrent pas de dégradation temporelle simple et monotone. Ils empêchent de désigner la seule succession des attaques comme explication générale.

Deux exemples vérifiables dans `00_BN2-166-Ab_comp.jams` illustrent les erreurs ; les temps se rapportent à l'audio original, et les noms utilisent MIDI avec C4=60.

| Groupe global | Nouvelles notes annotées | Notes précédentes encore actives | K attendu | Témoin / résidu |
|---|---|---|---:|---:|
| 579 | A#3 à 0,2132 s ; E4 à 0,2153 s ; G#3 à 0,2173 s | C#3 jusqu'à 0,3689 s ; G#4 jusqu'à 0,4289 s | 3 | 2 / 2 |
| 598 | F#2 à 2,1798 s | E3 jusqu'à 2,2133 s ; A#3 jusqu'à 2,2088 s ; C#4 jusqu'à 2,2006 s | 1 | 0 / 1 |

Le JSON contient aussi des réussites : par exemple le groupe 680 retrouve bien K=3 avec une ancienne note encore active. Les exemples sont les premières lignes par catégorie et résultat, selon l'index global, et non une sélection de cas extrêmes. Ils sont reconstitués depuis les annotations ; aucune écoute ou séparation acoustique n'est invoquée comme preuve.

L'interprétation de travail est donc : **la transition vers de nouvelles notes sur des notes déjà en cours mérite une attention particulière, et le comptage de plusieurs attaques reste difficile même sans notes antérieures.** Avant de modifier les relations fréquentielles, le prochain test utile serait un jeu contrôlé conservant les mêmes notes, K et niveaux, en faisant varier séparément la fin des anciennes notes et le décalage des nouvelles attaques. Un tel test n'a pas été lancé dans cet audit.

Les limites sont matérielles : cinq compositions seulement, une graine d'entraînement, validation interne déjà examinée et conditionnée par les candidats existants. Les 415 attaques annotées qui ne sont assignées à aucun groupe restent en dehors de ce score Exact K. Les fins annotées servent d'approximation des notes encore audibles ; les queues de résonance réelles peuvent différer. Les groupes d'une piste sont corrélés : aucun test de significativité supposant des observations indépendantes n'est avancé. L'audit ne permet ni de conclure sur la transcription complète ni de prouver quel mécanisme acoustique produit les erreurs.

Les preuves et la reproduction sont fournies avec le dépôt :

- [Résultats et effectifs détaillés](evidence/v273-polyphony-timing/report.json).
- [Script de reproduction](../scripts/audit_v273_polyphony_timing.py).
- [Sept tests des définitions temporelles et de l'assignation](../test/test_polyphony_timing_audit.py).
- [Prédictions et rapports source](https://github.com/Andriamarosoa/note/releases/tag/v273-residual-native-36688979041) : `residual-observed_only.zip` et `residual-with_residual.zip`, avec `report.json` et `epoch-12-validation.npz` dans deux dossiers nommés `observed_only` et `with_residual`.
- [Géométrie source](https://github.com/Andriamarosoa/note/releases/tag/v273-ownership-36519613364) : `ownership-inputs.zip`, contenant notamment `report.json`, `rows.npz` et `full-timing/`.
- [Annotations GuitarSet](https://zenodo.org/records/3371780/files/annotation.zip), archive originale inchangée.

Les empreintes SHA-256 des deux NPZ, du fichier de configuration, du rapport de géométrie, des fichiers de candidats et des annotations sont vérifiées. Les 9 542 attaques sont assignées indépendamment par recherche locale et recherche exhaustive ; 9 127 sont rattachées aux groupes. Les 15 952 valeurs de K obtenues correspondent exactement aux cibles archivées. Les prédictions correspondent à l'argmax archivé. Aucun fold externe n'est évalué.

```bash
python -m unittest test.test_polyphony_timing_audit -v

python scripts/audit_v273_polyphony_timing.py \
  --geometry-dir /chemin/ownership-inputs \
  --prediction-dir /chemin/residual-predictions \
  --annotations /chemin/annotation.zip \
  --config analysis/v273-native-paired-config.json \
  --output-dir /chemin/audit-output \
  --rows-output /chemin/audit-rows.npz
```

Python avec NumPy suffit ; les poids des modèles et TensorFlow ne sont pas nécessaires. Le NPZ optionnel conserve, pour chaque groupe, les indicateurs temporels, les prédictions et les événements annotés. Le JSON est déterministe et comporte l'empreinte du script qui l'a produit.
