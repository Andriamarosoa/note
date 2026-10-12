# S5 : constructeur 527D reconstruit et probabilités originales reproduites

Date : 12 octobre 2026. S80 reste expérimental ; `freeze_local_combo` reste la référence officielle.

## Résultat

Le constructeur acoustique manquant de S4-owned/S5 est maintenant reconstruit numériquement. Les **quatre modèles S5 originaux**, inchangés et vérifiés par SHA256, retrouvent **exactement les probabilités brutes et corrigées de prior des 59 309 événements historiques**. Aucun modèle n'est réentraîné ; aucun tableau de vrai K n'est chargé.

| Fold évalué | Événements | Écart maximal brut | Écart maximal après prior | Colonnes de discrétisation différentes |
|---|---:|---:|---:|---:|
| 0 | 15 952 | 0 | 0 | 0 / 527 |
| 1 | 13 868 | 0 | 0 | 0 / 527 |
| 2 | 14 665 | 0 | 0 | 0 / 527 |
| 4 | 14 824 | 0 | 0 | 0 / 527 |

Les **412 350 seuils de discrétisation** des 2 108 colonnes de modèles contrôlées sont également identiques. Les identifiants des lignes d'entraînement et d'évaluation concordent avec chaque checkpoint. Le rapport brut est dans [report.json](evidence/v273-s5-owned527-recovery/report.json).

La vérification locale a réussi. Le workflow GitHub dédié rejoue la même vérification en téléchargeant les sources immuables ; son résultat sera consigné après exécution.

## Ce qui a permis la reconstruction

Les recherches précédentes examinaient le protocole, les arbres et les sources Git accessibles. Les modèles HGB originaux conservent aussi leur **`_bin_mapper.bin_thresholds_`**, une signature numérique de chacune des colonnes d'entraînement. Cette trace permet de vérifier une formule de caractéristique sans utiliser les annotations.

Le protocole S4 fixe la structure : 111 colonnes de contrôle et 416 colonnes propres à quatre zones temporelles. La comparaison aux signatures originales a permis de retrouver les conventions manquantes :

- Inversion de `log1p` avec `expm1`, après conversion de la séquence en **float64**.
- Quatre masques : propriétaire natif, dilation de deux frames, avant la première frame native et après la dernière. Les frames hors masque restent à zéro dans la grille de 42 frames.
- Pour chaque zone : 20 descripteurs d'activation, puis les flux positifs et la masse négative aux décalages 1, 2, 4 et 8. La différence temporelle est **divisée par son décalage**.
- Les 20 descripteurs comprennent les huit maxima de pitch les plus forts, leur masse, leur concentration quadratique, leur nombre effectif, quatre comptes relatifs et cinq statistiques temporelles.
- Calculs intermédiaires en float64 ; conversion finale des caractéristiques en float32.

La signature a été mise au point à partir du fold 0, puis les probabilités archivées et les signatures des quatre folds ont été vérifiées. Cette procédure est une reconstruction sur données historiques exposées, pas un test de généralisation indépendant.

## Sources et code reproductible

- Branche d'origine indiquée par l'utilisateur : [`codex/v273-open-k0-k6`](https://github.com/Andriamarosoa/note/tree/codex/v273-open-k0-k6).
- [Modèles et probabilités S5 originaux](https://github.com/Andriamarosoa/note/tree/7165e1ef54510aa9effc9f597b2783c832ca907a/analysis/evidence/v273-open-k0-k6/series5/models), commit immuable `7165e1ef54510aa9effc9f597b2783c832ca907a`.
- [Archive acoustique originale](https://github.com/Andriamarosoa/note/releases/tag/v273-open-k0-k6-37837235723), folds 0/1/2/4.
- Constructeur : [`scripts/v273_s4_owned527.py`](../scripts/v273_s4_owned527.py), fonction `build_owned527(summary58, geometry46, sequence42x49, freeze_k)`.
- Vérificateur : [`scripts/verify_v273_s5_owned527_recovery.py`](../scripts/verify_v273_s5_owned527_recovery.py).
- Workflow dédié : [`.github/workflows/v273-s5-owned527-recovery.yml`](../.github/workflows/v273-s5-owned527-recovery.yml).

Le constructeur est une fonction des données acoustiques et du K prédit par la référence gelée. Il ne lit pas les probabilités archivées. Le vérificateur consulte celles-ci uniquement pour comparer les sorties calculées. Les archives d'entrée, modèles et prédictions sont épinglés par SHA256 ; les extractions locales sont vérifiées contre les archives.

En Python 3.12 :

```bash
python -m pip install numpy==2.3.5 scipy==1.17.0 scikit-learn==1.8.0 joblib==1.5.3
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python -m scripts.verify_v273_s5_owned527_recovery --fetch --cache-root model/s5-recovery --output model/s5-recovery-proof
```

Le dossier de sortie doit être absent. Tout écart de probabilité ou de seuil fait échouer la commande. La CI conserve le rapport JSON et son résumé dans l'artefact `original-s5-owned527-reconstruction-proof`.

## Portée exacte et blocages restants

**Le fichier source historique n'a pas été retrouvé.** Le présent code est une reconstruction dont l'équivalence de sortie est exacte sur l'ensemble historique disponible. Les six checkpoints S4-owned disponibles n'ont pas été rejoués par ce vérificateur S5 ; leur validation propre reste à effectuer.

La sérialisation du champ historique `designs_sha256` est inconnue. L'empreinte des octets de la matrice float32 reconstruite est publiée séparément et ne correspond pas à ce champ. Aucune identité octet par octet avec un fichier de design original absent n'est revendiquée. Les preuves établies sont les signatures numériques et toutes les probabilités historiques.

Cette récupération lève le blocage de reconstruction S5 que les audits de sources et les approximations S118–S135 n'avaient pas résolu. Leurs constats sur l'absence du texte source restent valables ; leurs échecs de fidélité ne décrivent pas ce nouveau constructeur.

Elle ne suffit pas à promouvoir S80. Il reste à intégrer le constructeur aux caractéristiques de nouveaux WAV, exécuter les modèles S5 gelés, compléter les seize propositions originales S43/S44 et les risques S45, appliquer les étages S46–S80 avec une politique de déploiement prédéfinie, puis comparer la chaîne entière à `freeze_local_combo` sur des compositions indépendantes. Aucun score S80 sur nouvel audio n'est établi par le présent rejeu.
