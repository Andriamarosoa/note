# Préserver les valeurs spectrales : test natif contrôlé

## Hypothèse et intervention

Le comptage natif applique `LayerNormalization(axis=-1)` aux trois canaux
spectraux à chaque point temps/fréquence. Ces canaux décrivent la puissance
logarithmique relative au pré-signal, son excès positif par rapport au
pré-signal, et le flux temporel positif. La normalisation locale peut
atténuer des différences d’intensité entre des triplets proportionnels.
Cette propriété mathématique ne prouve pas qu’elle cause les erreurs de K.

Comparer deux modèles frais :

- `channel_norm` : normalisation actuelle, avec ses gamma/beta entraînables ;
- `fixed_scale` : division fixe des trois canaux par 12, sans soustraction
  de moyenne, sans variance estimée et sans nouvelle coupure des valeurs.

Les canaux du cache sont déjà bornés entre 0 et 12. Le changement porte
uniquement sur `v240_dense_channel_norm`, dans la branche spectrale du
comptage. Il retire ses six paramètres gamma/beta ; les autres couches,
leurs dimensions et leurs poids initiaux doivent être identiques.
Le test ne sépare pas tous les effets du conditionnement numérique de
ceux de la préservation des amplitudes.

## Protocole fixé avant résultats

Partition interne de l’expérience du fold externe 3 : 43 357 groupes vus
et 15 952 groupes de validation interne, dont 5 274 et 2 111 vrais K≥2.
**Aucune ligne du fold externe 3 ne passe dans le modèle pour ce test.**
Le bundle et le découpage existants sont vérifiés par empreintes SHA256.

Entraîner les deux modèles depuis zéro pendant **12 époques**, sans reprise
d’anciens poids. Objectif identique : entropie croisée catégorielle, poids
unitaires, Adam à 0,0002, lots de 128, fenêtre de 31 trames.
La graine d’initialisation et d’ordre des exemples est 16 164 ; le dropout
de comptage de taux 0,08 reçoit la même graine explicite 46 164 dans les
deux variantes. Les autres défauts du constructeur restent inchangés.

Ce contrôle frais est la référence de cette comparaison. Les checkpoints
à 12 de l’expérience précédente provenaient d’une reprise avec un autre
flux de dropout : leurs chiffres ne sont pas le témoin de ce test.

**Comparaison principale : `fixed_scale` contre `channel_norm`, à l’époque
12.** Les checkpoints à 4 et 8 sont des points intermédiaires annoncés à
l’avance, sans substitution automatique au point principal. Le budget
12 s’appuie sur l’exploration précédente ; cette validation interne est un
ensemble de développement déjà inspecté, sans revendication indépendante.

## Contrôles obligatoires

Avant tout entraînement réel : vérifier que l’initialisation historique
par défaut est conservée, que tous les poids initiaux hors normalisation
sont identiques, que seuls six paramètres disparaissent et que le premier
masque de dropout est identique. Un lot synthétique doit traverser chaque
modèle avec une mise à jour finie et des métriques par K correctes.

Pendant l’entraînement : contrôler les ordres des lots et les compteurs
d’Adam, sauvegarder chaque époque, suivre Exact K global, polyphonique et
par K sur la validation. Les scores sur les exemples vus en mode inférence
seront mesurés aux checkpoints 4, 8 et 12 : les métriques de boucle avec
dropout ne leur sont pas assimilées.

Après entraînement : recalculer les scores depuis les probabilités brutes,
vérifier les populations, les empreintes, les pertes et les métriques de
validation, puis compter les corrections et régressions groupe par groupe
et par composition. Les sauvegardes à 4 et 8 sont publiées pendant le calcul.

## Interprétation annoncée

Critère principal : gain d’Exact K polyphonique interne à 12 époques.
Un gain accompagné d’un recul pour K=1, 2 ou 3 sera explicitement qualifié
de résultat mixte. Les résultats de K=0 et K=4/5/6 restent intégralement
rapportés. Une absence de gain ne valide pas cette correction.

Une seule graine et une seule partition interne : même un résultat positif
reste un candidat de développement à confirmer. Aucun correcteur de sortie,
aucun nouveau score externe, aucune promotion automatique. La référence
officielle V27.3 reste inchangée.

La [documentation Keras de LayerNormalization](https://keras.io/2/api/layers/normalization_layers/layer_normalization/)
décrit l’opération testée ; elle ne constitue pas une preuve d’amélioration
sur GuitarSet.
