# Test natif du contexte des groupes voisins

## Défaut ciblé et hypothèse

L'audit `v273-design-contract.md` a construit deux configurations de propositions
qui ont les mêmes quatre entrées natives mais des cibles différentes : l'attribution
de l'annotation dépend aussi de candidats d'un groupe voisin. Il démontre une
information manquante dans l'interface, pas la proportion d'erreurs réelles qu'un
modèle entraîné corrigera. Le présent test mesure l'effet d'un contexte supplémentaire.

## Comparaison fixée avant entraînement

| Branche | Canal temporel fourni au modèle |
|---|---|
| `local_only` | Fraction des échantillons de chaque trame à distance ≤ 882 d'un candidat du groupe courant |
| `with_neighbors` | Fraction attribuée au groupe courant après recherche du candidat le plus proche parmi tous les groupes, à distance ≤ 882 ; égalité départagée par l'identifiant de groupe le plus faible |

Les positions complètes avant troncature sont utilisées dans les deux branches.
Le calcul géométrique ne reçoit aucune annotation, aucun K ni aucune prédiction de K.
Les fractions portent exactement sur les 256 échantillons de chacune des 31 trames
acoustiques, avec un pas de 128. Les échantillons avant le début du signal ne sont
jamais admissibles. Cette compression par trame peut confondre certaines frontières :
nous ne prétendons pas fournir au modèle le masque exact sans perte.

Le canal `(31,1)` est répété sur les 64 bandes, ajouté après la normalisation spectrale
historique, avant la première convolution. Les deux réseaux sont identiques, avec
288 paramètres supplémentaires par rapport au modèle historique. **Le témoin local
est donc nouveau** ; cette expérience isole l'effet de la concurrence des voisins,
pas celui du seul ajout d'un canal par rapport à l'ancien modèle à quatre entrées.

Deux entraînements frais de 12 époques, mêmes initialisations (graine 16164), même
séquence de dropout (graine 46164), même architecture, normalisation, Adam 2e-4,
lots de 128, ordre des lots et entropie croisée non pondérée. Aucun correcteur de
sortie. Les poids communs et les prédictions sur entrées identiques sont vérifiés
avant lancement, ainsi que le gradient du canal et une mise à jour synthétique.
Le constructeur historique garde exactement son empreinte de poids par défaut.

Partitions internes déjà fixées pour le fold externe 3 : 43 357 groupes en apprentissage
(5 274 polyphoniques), 15 952 en validation (2 111 polyphoniques), 190 pistes au total.
Les cibles et les quatre entrées historiques restent identiques dans les deux branches.
Le lot source archivé contient également des lignes externes : leur contexte nouveau
reste NaN et elles ne sont ni entraînées ni évaluées. Aucun autre fold externe n'est lancé.

## Sources et disponibilité temporelle

Les dix archives d'apprentissage et le lot natif de la release
`v273-window-pair-36351028493` sont figés par SHA-256 dans le fichier de lancement.
Chaque cache source doit correspondre octet pour octet à celui du lot commun ;
chacun de ses champs historiques est également comparé aux lignes correspondantes.
Les archives contiennent déjà toutes les positions : aucune proposition n'est recalculée.

Pour une dernière position courante `e`, les propositions jusqu'à `e + 2 × 882`
suffisent pour déterminer l'appartenance des échantillons admissibles au groupe.
L'égalité entre le masque complet et celui de ce préfixe est vérifiée pour chaque
échantillon des 4 096 échantillons acoustiques de chaque groupe interne.
Le pic V8.6 a besoin du score suivant et sa fusion d'une marge de 4 échantillons.
Les tests vérifient la stabilité du préfixe après cette marge de 5, y compris sur
une chaîne de pics croissants. Les deux branches utilisent le même délai théorique :
`max(début + 2788, e + 1764 + 5, e + 1024)`, au plus 3 533 échantillons après le début
(80,11 ms à 44,1 kHz). **C'est une borne de support des échantillons du flux de
propositions causal figé, pas une latence de bout en bout mesurée.**

## Verdict et audit prévus

Le point principal est l'époque 12. Les points 4 et 8 sont archivés pour diagnostic,
sans sélection du meilleur checkpoint. Le critère exploratoire exige un gain d'Exact K
polyphonique de validation sans recul de K=1, 2 ou 3. Tous les K, y compris K=0,
le total, les surcomptages, les sous-comptages, les corrections et les régressions
sont publiés. Un gain polyphonique avec une régression d'un petit K est déclaré mixte.

L'audit recalcule tous les scores depuis les probabilités, contrôle les partitions,
les poids initiaux, les mises à jour et l'ordre des lots. Il reconstruit le contexte
depuis les propositions complètes, et distingue les groupes où les deux canaux
diffèrent de ceux sans concurrence. Ces strates localisent l'effet observé ; elles
ne prouvent pas une causalité pour chaque erreur particulière.

Une seule graine et une partition interne déjà inspectée : résultat de développement,
sans affirmation de significativité ni de généralisation. Aucun modèle n'est promu
automatiquement. V27.3 reste la référence officielle à 42,6019 %, non directement
comparable au score de ce composant natif interne. À la publication de ce protocole,
aucun gain du contexte n'est encore démontré par entraînement.
