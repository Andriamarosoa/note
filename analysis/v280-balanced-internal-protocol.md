# V28.0-G — comparaison interne de la pondération de cardinalité

Protocole du 12 septembre 2026, avant l'entraînement G. Il réalise le
prochain essai défini dans le [diagnostic de F](v280-undercount-diagnostic.md).
L'hypothèse est qu'un poids relatif accru de la polyphonie dans la perte
principale peut réduire le sous-comptage. Le diagnostic ne prouve pas cette
cause ; une amélioration sur V27.3 n'est pas présumée.

## Deux bras appariés

| Élément | Témoin `control` | Expérimental `balanced` |
|---|---|---|
| Architecture | V28 harmonique, 110 402 paramètres | Identique |
| Perte principale K<2 | Poids 1 | N / (2 × N_nonpoly) |
| Perte principale K≥2 | Poids 1 | N / (2 × N_poly) |
| Pertes et masques auxiliaires | Ceux de F | Identiques |
| Données, ordre des lots, initialisation | Figés | Identiques |

Les poids expérimentaux sont calculés uniquement sur l'ajustement :
`N=46921`, `N_poly=5514`, `N_nonpoly=41407`, soit environ **4,2547** pour
une ligne polyphonique et **0,5666** pour une autre. Chaque groupe reçoit
la moitié du poids total, de moyenne 1 avant l'arrondi float32. Les
fréquences internes à chaque groupe ne sont pas équilibrées séparément.

Les deux bras repartent de zéro, avec les initialisations par nom de couche
de F, seed `28035`, Adam `2e-4`, batch `32`, 12 époques. Chaque ligne apparaît
une seule fois par époque avec le sampler de F. Ni les crops causaux,
l'augmentation, les masques, ni les coefficients des pertes auxiliaires ne
changent. La somme pondérée des pertes d'entraînement n'est pas directement
comparable entre les deux bras ; les métriques de validation sont non pondérées.

## Données et périmètre

Ajustement : folds canoniques **2, 3, 4**, 46 921 lignes. Validation interne :
fold **1**, 14 001 lignes dont 1 776 polyphoniques. Le fold 0 est exclu de
l'apprentissage et de l'inférence. Les vérifications de couverture peuvent
relire ses métadonnées, mais ses cibles ne déterminent aucun poids ni choix.

Le cache F est réutilisé sans recalcul ni nouvelle lecture audio/JAMS :

- run `34462604665`, tentative 1, terminé avec succès ;
- commit `50e615bfb6cbce6ac5bbc57843ef3d1378e1bf79` ;
- artefact `v280-f-prepared-all`, ID `10146298232` ;
- SHA-256 ZIP `0f747055fb3c6acef2104d67a17b036b1252fe58a73bb840754cded5eab00cb1` ;
- SHA-256 manifeste `e43567e3bd0945911915b4b11d88f960e4d95c0519809045dc9c7b056f0e5226`.

Les implémentations de F, E et du modèle sont vérifiées par les trois SHA-256
du diagnostic. Les fichiers du cache et les partitions sont vérifiés dans
chaque job. Le fichier événementiel est seulement vérifié par son empreinte,
pas désérialisé pour cette expérience. Aucune prédiction V27.3 ou externe F
n'est téléchargée pour entraîner ou comparer G.

Cette hypothèse a été choisie après observation de F et ces compositions
ont déjà servi à la recherche. G est une comparaison interne de développement
sur un seul split et un seul seed, pas une validation indépendante.

## Entraînement et contrôles

Un job de préparation exécute les tests, vérifie la source et fige les poids.
Quatre jobs d'entraînement se suivent : témoin époques 1–6, témoin 7–12,
pondéré 1–6, pondéré 7–12. Chaque bras reçoit **17 604 mises à jour**.
Le découpage conserve les derniers poids et toutes les variables Adam ; il
ne redémarre pas du meilleur checkpoint. Aucun ancien modèle F n'est utilisé
comme témoin entraîné : le témoin est réentraîné dans le même run que G.

Les tests vérifient notamment la masse 50/50, l'indépendance vis-à-vis des
cibles hors ajustement, la conservation des masques auxiliaires, l'égalité
entre les gradients pondérés et ceux de la moyenne des deux groupes, la
reprise complète des deux bras, l'égalité de leurs initialisations et
l'absence d'accès aux features externes sur une exécution synthétique.
Les tests TensorFlow doivent réussir dans Actions avant les données réelles.

Les historiques, les meilleurs et derniers poids, les moments Adam et les
prédictions internes du meilleur et du dernier état sont conservés. Ces NPZ
contiennent les identités et quatre sorties : cardinalité finale,
Poisson-binomial, probabilités par corde, logits résiduels. Un contrôle
NumPy reconstruit les deux combinaisons à partir des sorties enregistrées.
Il tolère le calcul float32 ; les comptages et identités restent exacts.

## Sélection, comparaison et suite

Sélection dans chaque bras : maximum de corrects polyphoniques internes,
puis minimum de NLL polyphonique, puis époque la plus ancienne. Le meilleur
état peut différer du dernier ; les deux sont audités et leurs courbes
complètes figurent dans la comparaison.

L'agrégateur exige 12 époques complètes pour chaque bras, les mêmes initialisations
y compris les couches harmoniques, le même nombre de mises à jour et les
mêmes ordres de lignes. Il recalcule Exact-K polyphonique et global, confusion
par K, sous/sur-comptage, NLL et Brier des sorties finale et Poisson-binomial.
Il compare uniquement des prédictions internes alignées.

La préférence interne est `balanced` si son nombre de corrects polyphoniques
est strictement supérieur, `control` s'il est inférieur, `tie` en cas
d'égalité. Les dégradations éventuelles en global, NLL ou sous-comptage
doivent être rapportées ; elles ne sont pas masquées par cette préférence.
Le bootstrap apparié par piste, 10 000 réplications seed `28035`, est
descriptif et conditionnel à la sélection des checkpoints.

Le workflow s'arrête après la comparaison. Une confirmation sur une autre
partition interne et une nouvelle évaluation externe feront l'objet d'une
décision à partir de ce résultat. **V27.3 reste la référence**, sans promotion,
refit, déploiement ni évaluation externe automatique. Les critères de gain
matériel face à V27.3 définis pour F restent inchangés.

Artefacts : `v280-g-preparation-audit`, `v280-g-control-1/2`,
`v280-g-balanced-1/2`, `v280-g-internal-comparison`, conservés 90 jours.
La durée complète dépend des runners CPU et peut atteindre plusieurs heures.
