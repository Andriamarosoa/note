# Contrôle direct de la persistance — protocole additionnel

Décidé le 6 octobre 2026, après le résultat du critère conjoint et **avant**
l'évaluation de cette nouvelle feature. Le protocole temporel initial reste
inchangé dans son fichier et conserve son empreinte.

Le critère moyen des deux fenêtres donne 135 corrections et 167 régressions,
net −32, et est rejeté sur les quatre FIT. Il ne transforme donc pas la variation
temporelle en correction utile. Sur les 272 K3, 313/405 composantes attendues
du premier choix persistent contre 165/411 composantes non appariées. Cette
association ne prouve pas une causalité et ne décrit pas directement la
séparation K2/K3.

Tester une seule information supplémentaire, sans recalcul d'audio : le nombre
de fréquences du **triplet** choisi dans la première fenêtre qui sont appariées
univoquement à celles du triplet choisi dans la seconde, à 55 cents, divisé par 3.
Employer un triplet pour **tous** les vrais K ; jamais de couple conditionné par
la cible. Le pool de la première fenêtre et les choix indépendants déjà archivés
restent figés. Il s'agit de persistance des fréquences sélectionnées, qui ne
garantit ni une fondamentale ni un coefficient actif dans les deux fenêtres.

Contrôle : les deux résidus de la première fenêtre. Nouveau bras : ces deux
résidus + cette unique feature de persistance. Même LR, C=1, pondération
balanced, seuil 0,5 ; aucune optimisation de seuil, durée ou poids. Même
séparation FIT/VAL, mêmes populations, tous vrais K comptés, sélection sur la
rotation FIT et abstention si net <= 0. Rejouer aussi les modèles des folds
internes et comparer les probabilités du contrôle bit pour bit à l'archive.

Le diagnostic annoté décrira les distributions mais ne choisira aucun paramètre.
Un échec de ce test est publié sans nouvelle recherche de seuil. Aucun entraînement
neuronal, aucune annotation comme entrée, aucun fold 3, aucune promotion à partir
de VAL et aucune conclusion pour le chemin compressé. Le surcoût temporel reste
46,44 ms. Les folds internes ont déjà été inspectés.
