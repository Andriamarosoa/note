# S200–S203 — Pourquoi le véritable constructeur S5 owned 527D reste introuvable

**Branche:** `codex/v273-scope-rescue`. **Référence officielle:** `freeze_local_combo`. **Promotion de S80:** non effectuée.

Les reconstructions historiques du catalogue de **255 propositions** et des **neuf audits locaux** sont acquises sur 7 493 événements historiques via [S198](https://github.com/Andriamarosoa/note/actions/runs/38212710300). Cela **ne fournit pas** automatiquement les 416 colonnes supplémentaires de la représentation acoustique `S4 owned` réutilisée par S5.

## S200 — Mesure directe des caractéristiques réellement exploitées

[Action S200 réussie](https://github.com/Andriamarosoa/note/actions/runs/38213481348).

Un audit sans annotation a restauré les quatre véritables checkpoints HGB S5 `fold0/1/2/4`, ainsi que les **six checkpoints S4 owned réellement conservés** aux folds 0, 1 et 2, chacune avec pondérations poly1 et poly2. Les **deux modèles S4 owned du fold4 sont absents** du commit original conservé `854ada718ac9bf1556f00c96b8e1aa90402d6ab3`, des arbres de la branche `codex/v273-open-k0-k6` et de la release correspondante. Le code vérifie explicitement leur absence plutôt que de prétendre que huit checkpoints existent.

- S5: **159 146 séparations d'arbres**; **397 des 416 dimensions owned** utilisées au moins une fois.
- S4 owned: **239 913 séparations**; **397 des 416 dimensions owned** utilisées au moins une fois.
- Union des deux familles: **398 dimensions owned** utilisées, seulement **18/416** inutilisées.
- Sur les arbres S5, les quatre blocs de 104 dimensions représentent **70,48 % des séparations** ; le préfixe acoustique 58D représente 25,76 %, la géométrie 46D 2,56 %, et le one-hot du K figé 1,20 %.

**Conséquence:** il n'existe pas de simplification justifiée qui n'exigerait de reconstituer que quelques dimensions du suffixe. Ces fréquences de split n'identifient pas la formule des données, ni leur ordre.

## S201 — 24 permutations de quatre zones de 104 colonnes

[Action S201 réussie](https://github.com/Andriamarosoa/note/actions/runs/38213783912).

Le protocole a été figé : sélectionner 512 événements du fold historique 0 par SHA256 de l'identifiant, jamais en fonction de leur vraie classe. Former la meilleure hypothèse précédente S118 `log_activation_first` et, avec le **checkpoint S5 original non modifié**, essayer les `4!=24` permutations des quatre zones. Comparer chacune au **vecteur complet des probabilités S5 historiques authentiques** `predictions.npz:raw` vérifié par SHA256. Aucun label `k` du fichier audio ni MIDI n'a été lu. L'identité reproduit l'ancien rejeu S123, écart maximal **2,98×10^-8**.

L'ordre identique historique supposé `[0,1,2,3]` reste le meilleur des 24 :
- **63/512** verdicts K prédits différents des verdicts sauvegardés.
- **0/512** vecteurs complets de probabilités égaux à (10^{-6}) près.
- Erreur moyenne absolue de probabilité **0,0312644**.

Les 23 autres ordres ne résolvent pas le défaut. Changer uniquement l'ordre des quatre zones ne constitue pas le constructeur original.

## S202 — 120 permutations internes dans chaque zone

[Action S202 réussie](https://github.com/Andriamarosoa/note/actions/runs/38214002268).

Dans la représentation S118 hypothétique, chaque bloc 104D est composé de `activation20 + lag1(21) + lag2(21) + lag4(21) + lag8(21)`. Les **120 permutations 5!** ont été testées sur les mêmes 512 événements, modèles et probabilités épinglés, avec l'ordre des quatre zones inchangé.

- Meilleur accord sur la classe S5 sauvegardée : `[0,1,3,2,4]`, **60/512** verdicts différents (contre 63/512 pour l'identité).
- **0/512** vecteurs originaux complets reproduits à (10^{-6}) près pour **toutes** les 120 variantes.
- Son erreur moyenne de probabilité **0,0315205** est **moins bonne** que l'identité (**0,0312644**), malgré trois divergences de classes K en moins.
- Il s'agit uniquement d'une comparaison de fidélité aux anciennes sorties, **pas** de performances face à la vérité terrain. Choisir cet ordre comme nouveau modèle constituerait une adaptation aux sorties historiques, non une restauration prouvée.

## S203 — Recherche dans les publications GitHub hors arborescence Git

[Action S203 réussie](https://github.com/Andriamarosoa/note/actions/runs/38214244350).

Recherche complémentaire aux scans de branches, commits, archives NPZ/ZIP de S114/S132/S195/S199 : téléchargement des **5 vraies archives tar.gz de release** `series41`, `series43`, `series44`, `series45`, `series46`. Membres inspectés respectivement : 24, 16, 5, 310, 5. Aucun fichier Python/notebook source et aucune matrice NPZ 416D/527D dans ces cinq bundles. Résultat négatif **dans ce périmètre seulement**, ne prouvant pas qu'un ancien workspace local non archivé n'existe pas.

## Conclusion et voies techniques honnêtes

1. **La restitution originale exacte de S5 owned 527D n'est pas accomplie.** Les poids S5 enregistrent des modèles HGB et le hash des entrées de développement, pas les formules numériques ni l'ordre de construction du suffixe 416D. Une matrice synthétique ne peut pas être déclarée identique du simple fait de produire 527 valeurs ou quelques verdicts concordants.
2. La plus courte voie pour obtenir le vrai constructeur est de retrouver **le fichier source non versionné** qui a entraîné S4 owned/S5, **ou** un tableau original de designs 527D avec identités événements/folds et provenance attestable ; il servirait à fixer et tester les formules numériques. Les sources Git publiques/release consultées ne le contiennent pas.
3. **Voie alternative possible, non équivalente** : produire un **nouveau** constructeur 527D documenté et entraîner un nouveau S5, sans le présenter comme le S5 original. Cela permettrait un remplacement explicitement expérimental dans une nouvelle chaîne à comparer contre `freeze_local_combo` sur WAV inédits, mais **ne prouverait pas** une reconstruction S80 historique.
4. Toute promotion du **S80 original** demeure refusée tant que S5 527D historique, les autres producteurs S45 manquants et les 193 politiques S46 originales ne peuvent être rejoués sur le même nouveau jeu audio, avec une validation indépendante pairée.

Les travaux S200/S201/S202/S203 ne remplacent ni les métriques historiques exact K, ni un score sur de nouveaux WAV. **`freeze_local_combo` reste la référence officielle.**
