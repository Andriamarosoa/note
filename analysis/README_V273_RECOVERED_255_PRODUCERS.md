# Huit producteurs exécutables : 255 combinaisons sur de vrais WAV

12 octobre 2026. Référence officielle : `freeze_local_combo`. **S80 non promu.**

Le blocage du catalogue se réduit : les huit votes peuvent maintenant être
calculés sur de nouvelles entrées acoustiques. Les 255 propositions et leurs
neuf audits ont été produits sur les cinq événements GOAT éligibles, dans les
quatre branches d'origine. Ce résultat ne constitue pas encore les sorties
finales des cinq familles S45 manquantes.

## Preuve historique

- Les **70 régressions logistiques** (cinq spécialistes × quatorze ensembles
  de folds) ont été reconstruites avec le code et les hyperparamètres originaux.
  Chaque réentraînement utilise uniquement les identifiants d'entraînement de
  son manifeste original. Tous les postérieurs float32 enregistrés concordent
  **exactement**. Les poids reconstruits sont désormais conservés.
- Les **14 réseaux correcteurs** utilisent les poids H5 d'origine, sans
  réentraînement. Le rejeu NumPy GELU avec accumulation float64 présente un
  écart maximal de **1,0132789611816406 × 10⁻⁶** avec leurs probabilités TensorFlow archivées.
  Les probabilités de ces réseaux ne sont donc pas déclarées bit-identiques.
- Les huit votes sont : action gelée, cinq spécialistes, moyenne des cinq
  spécialistes, correcteur projeté sur K2–K6. Les fonctions originales
  `seven_candidates`, `project_actions` et `joint_groups` restent utilisées.
- **13 375 005 décisions de groupe comparées, zéro différence**, couvrant
  toutes les sorties hors entraînement des quatorze configurations.

Les labels historiques servent à refaire les entraînements originaux, ainsi
qu'aux statistiques des références d'audit. Chaque modèle exclut les labels
de ses événements destinataires. Aucune annotation GOAT n'est ouverte.

Une tentative exploratoire d'inversion linéaire des seules probabilités
quantifiées a été écartée : des erreurs de probabilité atteignaient 0,2715
sur des valeurs acoustiques extrêmes et 15 décisions de groupe divergeaient.
Le runtime sauvegardé utilise exclusivement la recette d'entraînement
originale, validée sur tous les caches, sans ces coefficients approximatifs.

## Passage sur GOAT

Le parent de ces familles historiques est **`freeze_local_combo`**, comme dans
`prepared/inputs.npz`. Remplacer ce parent par S18 changerait l'éligibilité et
la définition des audits. Les sorties S18 sont conservées pour les étages qui
les utilisent réellement.

| Résultat | Valeur |
|---|---:|
| Événements acoustiques GOAT | 57 |
| Événements dont la référence vaut K2, K3 ou K4 | 5 |
| Branches conservées séparément | 0, 1, 2, 4 |
| Propositions par événement et par branche | 255 |
| Tensor des audits | 4 × 5 × 255 × 9 |
| Vrais labels GOAT consultés | 0 |

Les neuf audits utilisent les références historiques hors fold et leurs
propositions doublement croisées, avec les fonctions originales `local_audits`
et `group_context`. Les sorties ne sont pas fusionnées entre folds.

Les caractéristiques de fondamentales 58D absentes de S97 ont été recalculées
sur les WAV épinglés, avec `FUNDAMENTAL_BANK`. Les 58 caractéristiques principales
sont recalculées en float64, comme celles des spécialistes historiques.
Quatorze valeurs sur 3 306 diffèrent après conversion float32 des archives S97,
sur quatre événements, dont un éligible ; l'écart maximal normalisé par
`1 + abs(S97)` vaut environ 9,97 × 10⁻⁶. Cette variation numérique est exposée
dans la provenance et n'est pas déclarée bit-identique à S97. Le rejeu CI utilise
la matrice acoustique conservée, vérifiée par SHA, pour isoler le test des producteurs.

GOAT a déjà été exposé : il sert de test d'exécution, pas de preuve indépendante
d'amélioration d'Exact-K. Aucune nouvelle précision n'est revendiquée ici.

## Reproduction et fichiers conservés

- [Runtime](../scripts/recover_v273_255_producers.py) : étapes `prepare`,
  `extract`, `recover`, `infer`.
- [Workflow](../.github/workflows/v273-recovered-255-producers.yml) : versions
  numériques épinglées, contrôle des SHA, rejeu des poids conservés, puis comparaison
  indépendante des sorties locales et CI. Un groupe de concurrence évite les doublons.
- [Rapport historique](evidence/v273-recovered-255/recovery_report.json),
  [rejeu en précision stabilisée](evidence/v273-recovered-255/precision_replay_report.json),
  [rapport GOAT](evidence/v273-recovered-255/goat_report.json),
  [sorties 255×9](evidence/v273-recovered-255/external_255x9.npz),
  [poids spécialistes](evidence/v273-recovered-255/specialists.joblib).
- [Entrées et provenance](evidence/v273-recovered-255/inputs) : matrices
  acoustiques historiques et GOAT, identifiants, noms de colonnes et empreintes
  des archives originales. Elles restent disponibles après expiration des artefacts CI.

Le premier run CI `38230762655` a échoué au contrôle strict des postérieurs
lors d'un nouveau réentraînement sur le runner GitHub. Le run de dépendances
`38230762422` a réussi. Le workflow de déploiement vérifie maintenant les poids
déjà reconstruits, conservés et SHA-épinglés, sans les réentraîner à chaque
machine. Le contrôle d'égalité des postérieurs et des décisions reste inchangé.
L'étape locale `recover` conserve la recette originale de reconstruction ;
`--saved-specialists` et `--saved-recovery-report` déclenchent son rejeu gelé.

Le deuxième run `38231100456` a retrouvé les **13 375 005 décisions** et
tous les audits GOAT sans écart, mais sa comparaison des descripteurs a détecté
une valeur d'écart-type différant de **6,78 × 10⁻⁵**. Les votes différaient
d'au plus 2,98 × 10⁻⁷ ; la racine de la variance presque nulle amplifiait cet
arrondi. La [comparaison complète](evidence/v273-recovered-255/first_frozen_CI_comparison.json)
est conservée. Le correcteur garde désormais les entrées arrondies float32 et
les poids originaux, accumule les opérations denses et GELU en float64, puis
rend des probabilités float32. **Les contrôles n'ont pas été assouplis** :
tolérance historique du correcteur 2 × 10⁻⁶, égalité exacte des décisions,
tolérance intermachines des descripteurs 2 × 10⁻⁵. Le rejeu local complet
conserve toutes les décisions et tous les audits après ce changement numérique.

## Raccordements encore nécessaires

Il reste à calculer les probabilités/logits des critiques originaux alimentant
S1–S4, exécuter le sélecteur `repair_coherent`, puis raccorder les seize familles
S45 avec leurs parents originaux. S35 utilise encore un adaptateur S20 à la
place de son parent S29 : son passage expérimental ne prouve pas une chaîne
historique intégrale. Les étages S46–S80 et la comparaison indépendante globale
et polyphonique restent requis avant promotion.
