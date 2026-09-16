# Reconstruction des sources V27.3

**Reconstruction terminée avec succès :**
[run 35086829802](https://github.com/Andriamarosoa/note/actions/runs/35086829802),
huit lots et audit final réussis. Les données reconstruites comportent
74 588 clusters, dont 9 354 polyphoniques, sur 240 pistes. Elles diffèrent de
l'archive historique. La suite est définie dans
[le protocole natif apparié](v273-native-paired-protocol.md).

Cette publication conserve les éléments nécessaires à la reprise des
expériences. **Elle ne remplace pas la référence officielle V27.3 et ne
contient aucun verdict sur l'ajout de décroissance dans ses entrées.**

Relance effective : [run 35084258528](https://github.com/Andriamarosoa/note/actions/runs/35084258528),
commit `4a7985df684187c668b36be83dbf7ca6d52938ab`.
La première étape a réussi : les sources encore accessibles sont sauvegardées
dans la [release de recherche](https://github.com/Andriamarosoa/note/releases/tag/v273-source-recovery-35084258528).
L'archive `v273-preserved-original-sources.zip` a pour SHA-256
`fd3cf8c57d671cc2e9b7af026c6cb361a16712dfb2841cef60c07a65b23ba7e0`.
Les prochaines relances peuvent vérifier et utiliser cette sauvegarde même
après expiration des artefacts originaux. Cela ne restaure pas les caches
et poids qui avaient déjà expiré avant cette sauvegarde.

### Reprise après l'erreur d'audit

Le run ci-dessus a terminé les trois époques V8.1, puis s'est arrêté :
l'audit avait atteint sa cible de 4 000 faux positifs sur 24 pistes, mais
V8.6/V8.7/V8.8 en exigent 30. Le seuil minimal de l'audit passe donc de
24 à 30 pistes d'apprentissage. Il s'agit d'une correction explicite de
la reconstruction ; les nouveaux résultats ne sont pas déclarés identiques
à ceux de la chaîne historique.

La reprise réutilise `v81/stream.epoch-03.keras`, SHA-256
`16db642e76add6ea5e8ec02ff7b30c2491e22020378e6d372d002478dc92978b`,
depuis `v273-rebuilt-proposal-sources.zip` dans la même release. Cette archive
a pour SHA-256 `e4a602295b0456a94e78b8ebee1cb2255e74f3f1847766c504a8f286fe942b24`.
L'archive, le producteur, le jeu de données, l'état terminé de V8.1 et le
checkpoint sont contrôlés avant toute réutilisation. L'ancien état et l'audit
échoué sont conservés sous `history/run-35084258528/`. L'audit à 30 pistes
est recalculé ; V8.1 n'est pas réentraîné.

GuitarSet est également mis en cache immédiatement après vérification des
empreintes, pour conserver ce téléchargement même si une étape suivante échoue.

Les caches spectraux historiques et les partitions internes V10.4 ont expiré.
L'utilisateur n'en a pas de sauvegarde. Les anciens poids V8.1/V8.4/V8.6/V8.7/
V8.8 nécessaires pour les recalculer ont également expiré. Il faut réentraîner
la chaîne qui prépare les candidats et reconstruire un témoin explicite.

## Éléments originaux conservés

Les modèles complets V10.1 et V10.2, leurs rapports, le rapport agrégé V10.4
et les métadonnées de clusters préservées pour V28 restent accessibles au
démarrage. Ils sont sauvegardés après contrôle des producteurs et des
empreintes d'archives. Les réglages historiques V10.1/V10.2 ont été lus dans
leurs rapports originaux et sont consignés dans
`analysis/v273-rebuild-historical-config.json` : 13 et 17 époques,
respectivement `ordinal_cumulative_050` et `structured_pb_argmax`.

Les poids complets V10.1/V10.2 servent de sauvegarde. Ils **ne doivent pas
remplacer les experts internes stricts** : les utiliser sur des partitions
ayant participé à leur entraînement contaminerait la comparaison.

## Reconstruction lancée par ce workflow

1. Sauvegarder les éléments originaux encore disponibles.
2. Réutiliser V8.1 terminé : trois époques, 800 exemples d'entraînement et
   200 de validation par époque, avec contrôle du checkpoint sauvegardé.
3. Refaire l'audit des faux positifs sur au moins 30 pistes d'apprentissage,
   puis V8.4 jusqu'au
   checkpoint `control.epoch-01.keras` effectivement utilisé par la chaîne.
   Les époques V8.4 2 et 3 n'étaient pas les sources de V27.3.
4. Réentraîner V8.6, V8.7 et V8.8 avec les programmes historiques, 30 pistes
   source et un maximum de 20 époques par modèle.
5. Recalculer les candidats et toutes les statistiques originales V9.1,
   puis les spectrogrammes natifs 23 × 64 × 3, en huit lots.
6. Comparer les nouveaux tableaux aux métadonnées historiques conservées.

Les programmes d'entraînement en amont sont inchangés par rapport au commit
de référence V27.3. Chaque étape enregistre son état, la commande exacte et
les empreintes des sorties. Une étape incomplète ou un poids modifié bloque
l'utilisation de ses résultats. Aucune approximation des statistiques à
partir de candidats déjà tronqués n'est utilisée.

Les scripts V8.6/V8.7/V8.8 conservent leur évaluation historique finale sur
`locked12`. Ces scores ne servent pas à choisir les paramètres de cette
reconstruction et ne sont pas des résultats du test natif avec/sans
décroissance. Aucun balayage de nouveaux réglages n'est prévu ici.

## Interprétation des sorties

Les fichiers `v273-preserved-original-sources.zip` sont des copies des sources
encore accessibles. Les fichiers `v273-rebuilt-*` sont issus de **nouveaux
entraînements**, pas d'une restauration des anciens poids. Leur identité
numérique avec l'ancienne chaîne n'est pas supposée.

La présence des mêmes pistes, du même nombre de clusters ou de métadonnées
identiques ne suffit pas à rétablir l'ancien score. Les anciens tableaux
`stats` et `spectral` restent indisponibles pour une comparaison directe.
Il faudra reconstruire les experts internes, réentraîner un témoin V27.3,
mesurer sa dérive et effectuer la comparaison appariée avec/sans décroissance
sur les mêmes partitions.

**Ce workflow s'arrête après la reconstruction et l'audit des entrées. Il
n'entraîne pas encore les deux bras du test de décroissance.** La référence
archivée reste 4 005 / 9 401, soit **42,6019 %** d'Exact-K polyphonique.

## Conservation

Les archives sont conservées 90 jours dans Actions et copiées, avec leurs
empreintes, dans une préversion GitHub `v273-source-recovery-<run_id>`.
Ces pièces jointes de release ne dépendent pas du délai d'expiration des
artefacts Actions. La préversion est une sauvegarde de recherche ; elle
n'annonce pas une nouvelle version validée du détecteur. Une archive
partielle de la chaîne porte les états d'étapes dans `rebuild-state.json` et
ne doit pas être utilisée comme chaîne complète.

GuitarSet est retéléchargé depuis son dépôt officiel Zenodo si nécessaire,
avec vérification des empreintes historiques d'`annotation.zip` et
`audio_mono-pickup_mix.zip` avant les calculs.
