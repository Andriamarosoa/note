# S217–S220 — Résolution technique des désaccords entre folds S18 et 19 branches historiques

**Date :** 12 octobre 2026. **Branche :** `codex/v273-scope-rescue`. **Référence officielle conservée :** `freeze_local_combo`.

## Verdict

Le problème d'ingénierie « quel fold/pièce historique faut-il choisir ? » possède désormais une **règle déterministe, indépendante de l'ordre et des identités des branches**, avec abstention explicite. Deux implémentations indépendantes ont validé chacune ses sorties sur les **19 branches S45/S46 d'origine, les 4 folds, 57 événements audio et 193 colonnes de politique S46**.

**Attention :** décider une sortie unique n'est pas prouver son exactitude. Ce mécanisme est un **nouveau déploiement expérimental**, pas la reproduction du routeur historique S80 ni son remplacement officiel.

## S217 : choix S18 unique

[CI S217 — réussi](https://github.com/Andriamarosoa/note/actions/runs/38294189867) ; [CI S218 — contrôle indépendant réussi](https://github.com/Andriamarosoa/note/actions/runs/38294412914).

**Règle pré-étiquettes :**
1. Chaque fold 0, 1, 2 ou 4 contribue **un vote**, indépendamment de son nombre historique de modèles (5/4/5/5).
2. Une classe Exact-K est retenue avec au moins **3 votes identiques sur 4**.
3. Si la meilleure classe n'obtient que 2 votes ou moins, repli vers la sortie `freeze_local_combo` du même événement. Le cas est marqué comme incertain et **n'est pas masqué**.

Sur les six événements dont les quatre folds S18 étaient initialement discordants, **5 obtiennent au moins 3 votes ; 1 fait l'objet d'une abstention à la référence**. Sur l'ensemble des 57 événements, **4 sorties S18** diffèrent de la référence. Le nombre de prédictions S18 par événement est 1, contre jusqu'à 2 valeurs dans les six cas d'origine.

L'audit indépendant S218 utilise uniquement des boucles `collections.Counter` sans importer le code de S217. Ses **huit contrôles** sur les sorties, les masques et les comptes ont tous réussi.

## S217 révèle un piège : les corrections S46 s'annulent

Dans les 19 branches S46 originales, on observe **602 cellules de proposition actives** par rapport au parent S18 propre à leur fold. Un vote naïf exigeant qu'au moins trois folds se mettent d'accord sur un changement (tout en comptant chaque fold silencieux comme un refus) en conserve **0**. Ce vote ferait de S46 une copie de S18 : il est validé comme règle de consensus, mais **inapproprié pour préserver les spécialistes conditionnels S46**.

## S219 : seuls les spécialistes S46 effectivement actifs votent

[CI S219 — réussi](https://github.com/Andriamarosoa/note/actions/runs/38294699823) ; [CI S220 — validation indépendante réussie](https://github.com/Andriamarosoa/note/actions/runs/38294882827).

1. **Dans le fold**, seuls les producteurs historiques proposant un K différent de leur propre parent S18 comptent comme corrections ; exiger une **majorité stricte de ses 4 ou 5 modèles**, soit 3 voix concordantes.
2. **Entre folds**, les folds ne proposant rien s'abstiennent ; les folds proposant effectivement un K doivent être **unanimes parmi les actifs**. En cas de contradiction, repli sur S18 agrégé.
3. Une proposition n'est admissible que si le S18 de son fold d'origine est **égal au S18 agrégé**, pour éviter de transférer une porte de correction hors de son contexte de parent.
4. La première politique S46 est exactement la valeur S18 agrégée. Le choix ne dépend ni du numéro du fold, ni de la position, ni du nom des 19 morceaux historiques.

**Résultat GOAT sans annotation :** **2 cellules de politique S46** préservées sur **un événement distinct**. Les deux sont `series46__all16__polyLambda2__cut0__votes0__open` et `series46__all16__polyLambda2__cut0__votes1__open`, à l'instant identifiant `example2_di.wav:232745`, candidate K2→K1. **100 cellules** de proposition de spécialistes sont mises de côté parce qu'elles proviennent d'un S18 de fold incompatible avec le parent agrégé. Les sorties restent K0…K6.

S220 compare chaque cellule à une **autre** implémentation `Counter`, avec **sept assertions indépendantes réussies** : 193 prédictions, parent S18 identique, propositions des quatre folds, forces de vote, masques de parent incompatible, masques de conflit et nombres de folds actifs.

Ces **2 changements ne sont pas des corrections démontrées** : aucun fichier MIDI ou vérité terrain n'a été ouvert et aucun score Exact-K n'a été calculé dans ces workflows.

## Politique figée pour la prochaine étape

La règle est désormais explicitement versionnée dans [`v273_s217_s220_experimental_deployment_policy.json`](v273_s217_s220_experimental_deployment_policy.json) et ses producteurs sont : [S217](../scripts/s217_label_blind_hierarchical_branch_consensus.py), [S218](../scripts/s218_independent_verify_S18_fold_consensus.py), [S219](../scripts/s219_sparse_s46_specialist_branch_fusion.py) et [S220](../scripts/s220_independent_sparse_fold_proposal_audit.py).

**Ce que l'étape 1 a réellement résolu :** construire un S18 unique et un ensemble cohérent de propositions S46 sans choisir arbitrairement une branche ou utiliser les annotations. **Ce qu'elle n'a pas résolu :** retrouver la sélection historique S80 d'origine, prouver que les décisions sont exactes, exécuter S69→S80 sur nouvel audio et promouvoir l'officiel.

## Étapes suivantes

Raccorder le S69 par fold (qui n'est pas directement applicable à des morceaux nouveaux), les véritables parents S70/S71/S72, la route S79, puis S80. Ensuite figer des prédictions sur de nouveaux morceaux réellement song-disjoint avant de voir leurs annotations et mesurer K0…K6 / poly / global, corrections et régressions appariées vs `freeze_local_combo`. **Ne pas promouvoir S80 tant que ces vérifications n'ont pas réussi.**
