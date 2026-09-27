# Résultat principal : fenêtre de 23 contre 31 trames, fold 3

**Mise à jour : les quatre entraînements et l'audit complet sont terminés.**
Voir le [bilan complet](v273-window-pair-results.md). Le texte ci-dessous
conserve l'état intermédiaire publié après la comparaison principale.

La comparaison principale, sans pondération des classes, est terminée et
auditée. Le modèle à 31 trames obtient **34,0274 % d'Exact K polyphonique**,
contre **32,3007 %** à 23 trames : **+1,7268 point**, soit 34 groupes
correctement comptés de plus sur les mêmes 1 969 groupes.

La comparaison secondaire avec pondération reste incomplète à la publication
de ce rapport. Le modèle à 23 trames pondéré a terminé (728/1 969 = 36,9731 %),
celui à 31 trames pondéré est encore en entraînement. Les scores de deux
pondérations différentes ne permettent pas d'isoler l'effet de la fenêtre.

## Résultats vérifiés de la comparaison principale

| Mesure externe, fold 3 | 23 trames | 31 trames |
|---|---:|---:|
| Exact K polyphonique | 636/1 969 = 32,3007 % | 670/1 969 = 34,0274 % |
| Exact K global, silences inclus | 83,2777 % | 83,5657 % |
| Surcomptages polyphoniques | 288 | 288 |
| Sous-comptages polyphoniques | 1 045 | 1 011 |
| Surcomptages sur tous les groupes | 883 | 895 |
| Sous-comptages sur tous les groupes | 1 672 | 1 616 |

Sur les groupes polyphoniques, 122 erreurs deviennent correctes et 88 groupes
auparavant corrects deviennent erronés. Le bilan net est bien +34. Dans
160 autres groupes, une erreur devient une autre erreur : ils ne sont pas
comptés comme des corrections. Il reste 1 299 erreurs polyphoniques.

Le gain net est positif pour chacune des cinq compositions : +2, +4, +5,
+12 et +11 groupes. L'intervalle descriptif obtenu par rééchantillonnage de
ces cinq compositions est de +0,996 à +2,496 points. Une seule graine et cinq
compositions déjà utilisées pour le diagnostic ne prouvent pas la
généralisation de ce résultat.

En validation interne, le gain est plus faible : 399/2 111 = 18,9010 % contre
403/2 111 = 19,0905 %, soit +0,1895 point. Aucune époque ni variante n'a été
sélectionnée à partir du score externe.

## Conséquence pour la recherche de la cause

La couverture temporelle est corrigée et ce premier entraînement présente
un gain mesuré. En revanche, le total des surcomptages polyphoniques reste
inchangé. La correction n'a donc pas résolu le problème de surcomptage.

Parmi les 42 groupes polyphoniques ayant auparavant une attaque hors fenêtre,
le nombre de comptes exacts passe de 8 à 9 (deux corrections, une dégradation).
Le gain principal de 34 groupes ne peut pas être expliqué uniquement par
la récupération directe de ces attaques. Cette expérience réentraîne les
deux réseaux sur des durées différentes ; elle mesure l'effet de ce changement
sur l'ensemble de l'apprentissage, sans identifier à elle seule le mécanisme
de chaque erreur résiduelle.

## Portée et contrôle

Il s'agit du **composant natif de comptage à sept classes**, avec décodage
direct par argmax, sans correcteur ajouté. Ce n'est pas une évaluation de
la chaîne V27.3 complète. Sa référence officielle de 42,6019 % n'est pas
remplacée et ne doit pas être comparée directement à ces scores.

Les deux archives ont été vérifiées par SHA-256, puis chaque fichier par
son inventaire. L'audit contrôle les huit époques internes et les huit finales,
les poids initiaux, les graines, les poids de classes, les permutations de
lots observées, les partitions, les identifiants et l'ordre des prédictions.
Il rejoue l'argmax et les métriques depuis les probabilités sauvegardées.
L'inférence depuis les poids TensorFlow n'a pas été rejouée localement.

Seul le fold 3 est évalué comme test externe. Le proposant commun historique
n'a pas été réentraîné séparément dans chaque partition. L'audit complet
automatique attend toujours les quatre entraînements et inclura la
comparaison secondaire. Aucune promotion de modèle n'est effectuée.

- Exécution : https://github.com/Andriamarosoa/note/actions/runs/36351028493
- Source figée : `cdb11b6cfe50beac86194fc6a2ddca06149e8467`.
- Audit détaillé : [v273-window-pair-primary-audit.json](v273-window-pair-primary-audit.json).
- Protocole : [v273-window-pair-protocol.md](v273-window-pair-protocol.md).
- Artefacts : https://github.com/Andriamarosoa/note/releases/tag/v273-window-pair-36351028493

Ce rapport est un résultat intermédiaire : la comparaison principale est
auditée ; la comparaison secondaire reste en attente.
