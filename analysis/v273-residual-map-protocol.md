# Résidu temps–fréquence : protocole avant calcul réel, 2026-09-30

Cette étape construit et audite une entrée expérimentale. Aucun entraînement,
compte corrigé, seuil choisi ou gain Exact K ne sera annoncé. Les paramètres
ci-dessous sont fixés avant d'examiner les cartes GuitarSet.

## Représentation

- Flux PCM mono normalisé comme auparavant. Prévision Burg sur les 8 820
  échantillons passés, ordre 32, retard 64 et contraintes de stabilité inchangés.
- Blocs successifs de **128** échantillons, ancrés à l'échantillon 0 du morceau.
  Chaque bloc est prédit avant d'en observer un seul échantillon. Son audio
  observé entre ensuite dans l'historique pour prédire le bloc suivant.
- Le résidu est conservé en float32. Initialisation par zéro avant le début
  du fichier ; compter les fenêtres concernées par ce démarrage.
- Carte **31 × 64 × 4** : puissance du signal observé et du résidu, chacune
  pour des fenêtres Hann de 256 et 2 048 échantillons. FFT 2 048, 64 bandes
  géométriques de 55 à 6 000 Hz. Fenêtres alignées à droite aux instants
  `origine - 1052 + 128*j`, j=0..30. Dernier endpoint `origine + 2788` exclus.
- Diviser la FFT par la somme de Hann ; appliquer `log1p(puissance/1e-6)`.
  Échelle fixe commune, aucun écrêtage, aucune normalisation par des trames
  futures. Stockage de la carte en float16 avec erreur d'arrondi mesurée.
- Conserver séparément le masque exact d'appartenance de 4 096 échantillons
  (512 octets), ainsi que les fractions par trame. Les fractions ne remplacent
  pas le masque exact. L'audio n'est pas multiplié par ce masque.
- Toutes les propositions sont complètes avant la troncature historique à 48.
  Utiliser uniquement celles antérieures au watermark certifié ; refuser un
  délai de décision inférieur au maximum du support audio et du watermark+5.
  Ce délai est celui des entrées d'appartenance déjà auditées, pas une preuve
  de latence de production. Les annotations n'entrent dans aucune entrée.

La fenêtre longue commence jusqu'à 3 100 échantillons avant l'origine. Les
résidus correspondants utilisent encore 200 ms de passé. Cela ajoute de
l'historique, pas du futur au-delà du délai. Les 64 bandes et le zero-padding
ne promettent pas une résolution de notes voisines. Une carte n'est pas un
séparateur de sources ; des harmoniques et interférences peuvent y persister.

Ce protocole change le rythme de réestimation, la durée de prévision, la
couverture temporelle et la représentation. Il ne sera pas présenté comme
un test isolant la seule suppression de l'agrégation RMS.

## Population et contrôles

Conserver les 15 952 groupes et 50 pistes de la partition interne (composition
fold 0 du fold externe 3). Vérifier les sources, empreintes, identités des
lignes, origines, K et masques contre l'archive ownership-inputs et l'audit
réel précédent. Ne décoder aucun morceau du fold externe ou Locked12.

Avant les données réelles : silence, équivalence de chunks arbitraires,
préfixes arrêtés au milieu d'un bloc, invariance aux suffixes après la décision,
extrêmes temporels admissibles (-882 et +2646), refus d'un watermark trop court,
égalité avec la règle d'appartenance et contenu distinct à énergie globale
égale. Ces derniers tests établissent une conservation d'information, pas K.

Sur chaque piste : générer le flux causal et les cartes, conserver les sorties
par groupe, compter les échecs/non-finis et auditer les modules des pôles de
tous les blocs. Vérifier exactement les fractions d'appartenance historiques.
Recalculer K depuis les annotations sans les passer à la représentation.
Compter les attaques attribuées avant, dans et après l'ancienne fenêtre qui
disposent désormais d'au moins une trame Hann non nulle et d'un masque vrai.
Signaler les lignes avec padding ou initialisation incomplète du prédicteur.

Sur les cinq pistes `00_*_comp.jams` (une par composition), regénérer séparément
le préfixe terminé au dernier endpoint du groupe médian et exiger la même
carte. Toutes les lignes doivent respecter leur délai de décision sauvegardé.

Rapporter formes, tailles, extrema, écart float16, variance spatiale et part du
résidu conservée par rapport au signal observé, par piste et par vrai K.
Ces statistiques ne constituent pas un classement de K ni une validation
de prédiction. Ne pas résumer la carte en un nouveau scalaire choisi pour son
AUC sur ces mêmes labels. Aucun classifieur diagnostique n'est ajusté ici.

## Décision

La représentation est techniquement utilisable seulement si les comptes,
identités et masques concordent, toutes les attaques attribuées ont un support
temporel positif, aucun bloc/cartes ne présente d'échec ou de pôle instable,
les tests de préfixe/chunks passent et aucune donnée post-décision n'est lue.
Un succès autorise la préparation d'une comparaison native avec/sans ces
canaux, à architecture, cible, autres entrées et budget identiques. Il ne
garantit pas qu'elle améliorera Exact K. L'entraînement reste une étape séparée.
