# V28.0-H — corriger le biais de pondération et vérifier une fusion fixe

Protocole du 12 septembre 2026, figé avant l'entraînement H.

## Ce que G a appris

G a terminé 12 époques pour chacun de ses deux modèles harmoniques. Le
checkpoint sélectionné est l'époque 2 dans les deux cas. Sur la partition
interne 1, l'équilibrage 50/50 augmente Exact-K polyphonique de 38,4009 % à
52,1959 %, mais réduit Exact-K global de 77,8587 % à 68,5165 %. Les cas
véritablement à zéro ou une note classés polyphoniques passent de 839 à 2877.

Une exploration de **neuf décisions fixes** sur les probabilités G sauvegardées
est publiée dans [la découverte](v280-prior-fusion-discovery.md) et son
[JSON complet](v280-prior-fusion-discovery.json). Les meilleurs et derniers
checkpoints, les quatre sorties intermédiaires, les identités, les historiques,
les initialisations et les empreintes de G sont revérifiés. Aucune nouvelle
inférence ou optimisation neuronale n'est réalisée pour cette exploration.

La correction seule ne suffit pas. Le candidat retenu est une moyenne égale
du témoin et du modèle équilibré après correction de ses poids :

\[
q^{corr}_k=\frac{q^{balanced}_k/w_k}{\sum_j q^{balanced}_j/w_j},\qquad
p^H_k=\tfrac12 p^{control}_k+\tfrac12 q^{corr}_k.
\]

Les poids sont ceux de l'ajustement : un poids pour K<2, un pour K≥2. Pour
une entropie croisée pondérée isolée, le minimum théorique satisfait
q_k proportionnel à w_k p_k. Notre modèle comporte aussi des pertes
auxiliaires et une optimisation approchée : cette correction est donc une
hypothèse empirique, pas une garantie de calibration.

Sur G, la fusion donne 39,5833 % en polyphonie, 78,0659 % au global et
799 faux cas polyphoniques. Elle gagne **21 corrects polyphoniques** et
**29 corrects au total**, avec 40 faux cas polyphoniques de moins que le témoin.
Son intervalle bootstrap descriptif à 95 % pour le gain polyphonique est
**[−0,5510 ; +2,9889] points**. Il inclut zéro. Ces résultats utilisent les
checkpoints déjà sélectionnés sur G et un choix parmi neuf candidats ; ils
ne constituent pas une confirmation indépendante, et l'intervalle n'est
pas corrigé pour cette sélection.

## Expérience H figée

| Élément | Décision |
|---|---|
| Ajustement | Folds canoniques 1, 3, 4 : 45 640 lignes, dont 5 594 polyphoniques |
| Validation interne | Fold 2 : 15 282 lignes, dont 1 696 polyphoniques |
| Fold 0 | Exclu de l'apprentissage et de toute inférence |
| Modèles | Deux modèles harmoniques neufs, 110 402 paramètres chacun |
| Objectifs | Témoin uniforme ; équilibrage 50/50 de la seule perte principale |
| Durée | **2 époques fixes chacun**, d'après les deux checkpoints retenus sur G |
| Optimisation | Adam 2e-4, seed 28035, batch 32, 2 854 mises à jour par modèle |
| Données et lots | Même cache F, mêmes identités, mêmes ordres, aucune augmentation |
| Fusion | Correction complète, exposant 1 ; moyenne égale, coefficient 0,5 |
| Sélection sur H | Aucun checkpoint, seuil, coefficient ou nouvel essai choisi sur H |

Les deux modèles sont entraînés séquentiellement, puis leurs états sont
audités et figés ensemble **avant l'inférence de validation**. La validation
n'est pas évaluée après chaque époque. Chaque modèle parcourt une seule
fois les features de validation dans le job final ; ses probabilités sont
réutilisées pour les trois décisions : témoin, équilibré brut, fusion figée.
Les pertes auxiliaires et leurs masques sont ceux de G. Les poids 50/50 et
leur correction sont recalculés exclusivement sur les lignes d'ajustement H.

Le budget de deux époques teste le transfert du checkpoint et du décodeur
retenus sur G. Ce protocole ne recherche pas l'époque optimale sur la
nouvelle partition. Les anciens poids G et F ne sont pas réutilisés.

La fusion coûte **deux réseaux et deux passages**, soit 220 804 paramètres.
Le témoin n'est donc pas un contrôle de capacité ou de coût équivalent ; H
évalue un système combiné. Aucun gain de latence ou d'efficacité n'est revendiqué.

## Source, contrôles et verdict

Découverte : run G `34684806334`, tentative 1, commit
`bff41aec998a798055687c94705629eb98626aee`.
Les IDs et SHA-256 des trois artefacts G figurent dans `G_SOURCE` du script.
Le JSON de comparaison est figé par SHA-256
`680e54d783bae6894d42058fbe0f3f9845a70b2c64914c4124910ae78edbe2a9`.
Les sources F et G doivent être terminées avec succès et leurs artefacts
doivent correspondre exactement aux IDs et empreintes attendus.

Cache : run F `34462604665`, artefact `v280-f-prepared-all`, ID
`10146298232`. Le manifeste brut, les fichiers préparés et les implémentations
F/E/modèle/G sont vérifiés dans chaque processus. Les métadonnées des
partitions peuvent être relues pour vérifier leur couverture ; les données
événementielles externes ne sont pas désérialisées.

Les tests vérifient la correction sur une probabilité connue, le calcul des
poids hors validation, les contreparties du critère de décision, l'entraînement
effectif des deux modèles sur une petite fixture, l'exclusion des features
externes, le gel préalable, les deux seuls parcours internes et la
reproduction de la fusion sauvegardée. TensorFlow est obligatoire dans le
job de préparation, avant les vrais entraînements GuitarSet.

Le critère de développement exige simultanément : plus de corrects
polyphoniques, aucun correct global perdu, et aucune augmentation du nombre
de faux cas polyphoniques, tous comparés au témoin sur les mêmes lignes H.
Tous les scores, matrices de confusion, NLL, Brier et sous-comptages sont
publiés, y compris en cas d'échec. Le bootstrap apparié par piste utilise
10 000 réplications et seed 28035 ; il reste descriptif.

Cette seconde partition est une vérification de développement. Le choix
architectural et la recherche ont utilisé GuitarSet auparavant : ce n'est
pas une validation externe indépendante. **V27.3 reste la référence**.
Le workflow s'arrête après le verdict H, sans évaluation externe, promotion,
déploiement, recherche de seuil ou nouvel entraînement automatique.

Artefacts conservés 90 jours : `v280-h-preparation-audit`, `v280-h-control`,
`v280-h-balanced`, `v280-h-internal-comparison`. Les poids, moments Adam,
historiques, états gelés et quatre sorties par modèle sont sauvegardés.
