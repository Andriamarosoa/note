# V27.3 — ouverture des actions et des fragments K0 à K6

Protocole fixé avant calcul des nouveaux résultats. Base :
`a8fcf8c1c014ad9725bb39fd992bfbaab3b89247`.
Objectif : dépasser le YourMT3+ figé à 51 328/59 309 corrects globaux et
4 028/7 385 corrects polyphoniques, puis confirmer sur des données inédites.
Les données présentes sont du développement déjà exposé. Aucun résultat
sur celles-ci ne sera présenté comme cette confirmation indépendante.

## Série 1 : restriction d'action, poids strictement inchangés

Rejouer les sept probabilités déjà archivées du correcteur S8 et du critique
cohérent. Vérifier les identifiants, folds et exclusions de leurs producteurs.
Comparer la règle actuelle aux argmax K0–K6 et à l'ouverture limitée aux
propositions K0/K1 dont la probabilité atteint 0,50, 0,65, 0,80, 0,90 ou 0,95.
Pour ces cinq gardes, conserver ailleurs la meilleure politique des 211
actuelles. Ces règles sont fixes, sans réglage appris sur ces résultats.
Départager toute égalité en faveur de la référence. Sauvegarder toutes les
variantes, même négatives. Cette série reste limitée aux 7 493 lignes actuelles.
Conserver le bilan séparé K0/K1 et K2–K6 : un gain global peut cacher une
régression polyphonique. Aucune nouvelle preuve de provenance des anciens
producteurs n'est inventée ; leurs limites restent celles documentées.

## Série 2 : compter sur les 59 309 fragments

Extraire les caractéristiques depuis le même audio pickup GuitarSet pour
toutes les lignes natives des folds 0, 1, 2 et 4, sans filtrage par K prédit
ni vrai K. Exclure fold 3 et player05. Les labels sont stockés pour
l'entraînement et l'évaluation mais ne participent pas aux caractéristiques.
L'extraction ne charge aucun poids YourMT3+ et n'utilise pas ses prédictions.

Fenêtre audio inchangée : −80 à +160 ms par rapport au début du groupe.
Conserver les 58 résumés harmoniques précédents et, désormais, les activations
temporelles complètes des 49 gabarits MIDI 40–88 sur les 42 trames. La
normalisation est celle du fragment, sans ajustement global ni annotations.
La géométrie des groupes voisins fournit un masque d'appartenance courant
aux centres de trames, avec la règle native d'affectation à 20 ms et les
égalités au premier groupe. Ce masque est un contexte aux centres des trames,
pas une preuve de localisation exacte des attaques. Les bornes complètes
sont les horodatages entiers du benchmark ; aucune reconstruction float16.
Ne pas revendiquer une équivalence de latence avec YourMT3+ ou la référence.

Quatre bras neuronaux indépendants : résumés58 / trajectoires complètes avec
résumés et géométrie, chacun avec poids polyphonique 1 / 1,5. Dans chaque
bras, réseau dense 128 puis 64 GELU, sortie 7 logits, régularisation L2
0,0001, dropout 0,1, Adam 0,001, batch 256, seed 27405, 12 époques. Conserver
les états 4, 8 et 12 comme candidats préannoncés ; aucune sélection d'époque
ne modifie l'entraînement. Le poids moyen des exemples est ramené à 1
sur les seules données d'ajustement. La perte pondérée ne produit pas des
probabilités supposées calibrées.

Pour chaque fold tenu à l'écart, entraîner de zéro sur les trois autres.
Moyennes et écarts-types sont ajustés uniquement sur ces trois folds,
puis les entrées sont tronquées à ±6 écarts-types. Aucune prédiction de
référence, identité de morceau/interprète, annotation externe, probabilité
YourMT3+ ou résultat d'audit n'entre dans ces réseaux. Les compositions
doivent être disjointes entre les folds. Exporter les identifiants exacts
des partitions et les empreintes de toutes les données et poids.

Chaque candidat est mesuré seul et par mélange fixe de sa distribution
avec le one-hot de `freeze_local_combo`, poids du nouveau réseau 0,6 et 0,8
(poids 1 = réseau seul). Les égalités gardent la référence. Ces mélanges
n'apprennent rien depuis les sorties de référence ; leur usage ne certifie
pas la provenance historique de celle-ci. Soit 36 politiques annoncées.

## Décision, itérations et conservation

Rapporter global, poly, K0–K6, folds, corrections/régressions face à la
référence, au meilleur candidat précédent et à YourMT3+. Mesurer les bonnes
réponses nouvelles hors de l'union des 255 groupes et des 211 politiques,
ainsi que le nouveau plafond oracle, toujours séparé des scores réels.
Auditer les pertes des nouvelles régions d'action avant la série suivante.
Toute série adaptative est annoncée séparément avant son exécution.

Conserver chaque candidat avec ses prédictions, poids, normalisation,
historique, sources et cas gagnés/perdus. Aucun résultat faible ou négatif
n'est supprimé. Une future réutilisation par un sélecteur entraîné exige
des sorties des producteurs régénérées selon ses exclusions ; les seules
sorties OOF finales ne suffisent pas. Aucun modèle n'est promu automatiquement.
