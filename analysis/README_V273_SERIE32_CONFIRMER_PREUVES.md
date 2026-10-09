# Série32 — confirmation de preuves acoustiques S30/S31 sans faux gain apparent

Protocole préfixé avant mesure. S30 : les têtes A/B confirment presque toujours le même K, même erroné. S31 : le bloc de 335 caractéristiques et une logistique pièce-exclue donne 16 corrections et 3 régressions sur les 26 changements S29 (17 corrections, 4 régressions, 5 neutres). Question : imposer **deux estimations de fiabilité différentes** pour accepter une nouvelle décision peut-il éviter des régressions sans supprimer toutes les corrections ?

Sources archivées inchangées :
- Run S29 37865173636 `predictions.npz`, parent S18 ;
- Run S30 corrigé 37865974760 `gate_scores.npz` de la tête logistique/boosting sur les 90 premiers votes/audio et probabilités A/B ;
- Run S31 37866124959 `acoustic_scores.npz` des estimations de preuve flux 245, audio+flux 303 et signal complet 335.
Les prédictions de confiance sont **hors morceau** pour la tête concernée ; les producteurs ont néanmoins été développés sur ce même corpus, et le présent ensemble est une nouvelle recherche sur une cohorte connue.

Trois preuves prédéfinies : `S30 logistic90`, `S31 logistic_flow245`, `S31 logistic_all335`, plus un contrôle `S31 logistic_audio_flow303`. Les combinatoires sont **à choix fixe, sans étiquette pour décider** :
1. `all335_and_flow245`
2. `all335_and_audio_flow303`
3. `all335_and_prior90`
4. `flow245_and_prior90`
5. `at_least_two_of_three` parmi all335, flow245, prior90
6. `all_three` all335, flow245, prior90
7. `all335_and_prior90_and_audio303`

Chaque famille est évaluée aux **quatre seuils prédéclarés 0,25 / 0,40 / 0,55 / 0,70**, soit 28 combinaisons, plus S18, S29 et S31 meilleur acoustique de contrôle. Aucun nouveau fit, aucune règle utilisant `true_K` ou l'ID natif du cas. Conserver tous les 26 cas (et en particulier les cinq neutres) et produire un tableau exact des 17 gains conservés et quatre régressions bloquées.

Critères à afficher séparément :
- `aucune ancienne bonne décision S18 perdue`, `au moins une correction`, `poly >= S18` sur l'échantillon connu.
- **Même si une porte satisfait ces critères, elle reste exploratoire** : elle a été choisie dans une grille consultée sur un corpus déjà employé pour les séries précédentes. Tout résultat doit être contrôlé sur un jeu de compositions inédites gelé avant de déclarer un gain validé.
