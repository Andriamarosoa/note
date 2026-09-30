# Comparaison native avec/sans résidu — protocole avant entraînement

Date : 2026-09-30. Autorisation utilisateur : « go » après publication de
l'audit de représentation `2ca6f9900cb61e121e7cd9089a6c770903fb38c1`.

## Question et population

Mesurer si le résidu causal améliore le comptage natif K=0..6 du groupe,
par rapport au même modèle recevant seulement le signal observé.
Un seul couple d'entraînements, sur la partition interne existante du fold 3 :
43 357 groupes d'apprentissage, 15 952 de validation, 190 pistes au total.
Configuration des compositions inchangée, empreinte
`7c27989eb3fe330fdff203e388e76ae4c64d8b8e4838cab4d7c4622b4104451a`.
Aucune inférence/mesure sur le fold externe ni sur Locked12.

Les données de validation ont déjà été examinées dans les audits antérieurs :
il s'agit d'une expérience de développement, pas d'un test externe intact.
Les deux bras sont entraînés de zéro, sans réutiliser des poids précédents.

## Intervention unique entre les deux bras

La carte a la forme 31 × 64 × 4 et les paramètres gelés de
`scripts/v273_residual_map.py`. L'observation O et le résidu R emploient
la même échelle physique fixe, pour les fenêtres courtes et longues :

| Bras | Canaux |
|---|---|
| `observed_only` | O256, O256, O2048, O2048 |
| `with_residual` | O256, R256, O2048, R2048 |

Dupliquer O dans le contrôle donne le même nombre de canaux et de paramètres
sans fournir d'information résiduelle. Les cartes float16 sont converties en
float32 puis divisées par 12, sans écrêtage ni normalisation apprise sur les
données. Cette échelle est commune aux deux bras, même si quelques valeurs
dépassent 12 avant division.

Le réseau reste le composant de comptage catégoriel V25/V24 : contexte des
candidats, trois convolutions denses 32/64/96, moyennes/maxima globaux, couches
192/96 et softmax à sept classes. L'entrée spectrale passe explicitement à
quatre canaux dans les deux bras. Le défaut historique reste trois canaux.

Séquences de candidats, masques candidats et statistiques sont identiques.
Les deux bras reçoivent la fraction courte d'appartenance exacte aux voisins,
déjà employée dans `with_neighbors`. Le masque exact de 4 096 échantillons
reste conservé et auditable dans les caches ; le réseau consomme ici sa
projection historique en 31 fractions, pas les 4 096 bits directement.
Cette projection et le pooling global peuvent encore limiter la localisation.
L'expérience ne prétend pas résoudre tous les problèmes de conception.

Les délais sont ceux de l'archive ownership, jusqu'à environ 80,11 ms. Le
résidu et les cartes ne lisent rien au-delà du support autorisé. K et les
annotations ne sont jamais fournis à leur constructeur.

## Apprentissage fixé

- TensorFlow 2.15.1, NumPy 1.26.4, Python 3.11 ; opérations déterministes.
- Graine 16164, graine du dropout de comptage 46164.
- Architecture, poids initiaux, ordre des lignes par époque et budget communs.
- Adam, learning rate 0,0002 ; batch 128 ; cross-entropy non pondérée.
- **12 époques**, sans early stopping ni sélection du meilleur checkpoint.
- Checkpoints descriptifs 4, 8 et 12 ; le résultat primaire est l'époque 12.
- Décodage `argmax P(K)` ; aucun seuil ou correcteur de sortie.
- Une paire, une graine : un éventuel gain demandera ensuite une réplication.

## Préparation et contrôles avant lancement

Le bundle historique et la géométrie sont vérifiés par leurs empreintes et
inventaires. Seuls les 59 309 groupes internes reçoivent de nouvelles cartes.
Les emplacements externes restent NaN. Les 50 caches de validation déjà
audités sont réutilisés seulement si leurs identités, leurs empreintes et
le constructeur concordent. Les cartes d'apprentissage utilisent le même
constructeur figé, sans annotations. Les pôles et valeurs finies sont vérifiés.

Le preflight vérifie l'initialisation historique inchangée, l'identité complète
des poids entre bras, les entrées connectées à K, des gradients non nuls dans
les canaux résiduels, un apprentissage synthétique fini et les effets exacts
de la duplication dans le contrôle. Un échec arrête le lancement.

## Mesures et décision

À l'époque 12 : Exact K global, Exact K pour K>=2, chaque K séparément,
matrice de confusion, surcomptages K<4 et sous-comptages. Comparaisons appariées
sur les mêmes identifiants et même vrai K. Intervalles descriptifs par bootstrap
de composition, 2 000 tirages, graine 9302026 ; seulement cinq compositions de
validation, ce qui limite la précision.

La piste sera jugée favorable à une réplication si Exact K polyphonique
augmente, Exact K global et les rappels K1/K2/K3 ne diminuent pas, et les
surcomptages K<4 baissent. Publier aussi les résultats qui échouent à ces
conditions. Aucune promotion automatique, aucun remplacement du score
officiel V27.3 : population, modèle et protocole différents.

## Question bass/high rescue

Les fichiers et l'historique récupérés ne définissent pas d'étiquettes
`bass rescue` ou `high pitch rescue` pour ces lignes. La génération V8.6
retient `peak` et `baseline_edge`. Il est donc impossible d'attribuer
honnêtement les Exact K existants à trois origines normal/bass/high.
Une stratification par hauteur MIDI serait une autre analyse ; elle ne
devra pas être présentée comme une évaluation des deux mécanismes de rescue.
