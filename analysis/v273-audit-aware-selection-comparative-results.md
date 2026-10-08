# V27.3 — audit des caractéristiques et intégration train-only des sélections

**8 octobre 2026 — expériences terminées, contrôles réussis.**

Run CI comparatif :
https://github.com/Andriamarosoa/note/actions/runs/37758789956

## Ce que le réseau reçoit désormais

La variante d'identité fournit un **vecteur binaire explicite 7 bits**
H0, H1, H2, H3, H4, H5 et tête corrective Cxy si autorisée. Les 32 ou
64 sous-ensembles légaux sont maintenus ; H0 est obligatoire. Ce vecteur
distingue les combinaisons de même taille, notamment H3+H4 et H2+H5.

La variante des audits acoustiques réalise, dans chaque partition
d'entraînement externe, un KMeans non supervisé sur 26 des
caractéristiques spectrales/naissance/persistance/amortissement
normalisées. KMeans est ajusté exclusivement sur le train externe ;
sa projection sur la validation n'utilise aucun label. Pour chaque K
initial, K proposé, sous-ensemble, et régime acoustique, les audits
calculent le nombre de propositions et les taux correction,
régression et neutralité. Ils sont **lissés par les statistiques
globales de l'entraînement**. Lorsqu'une ligne du train reçoit
son audit, celui-ci est calculé à partir des *autres* folds
internes. Pour les événements du fold évalué, tous les audits
viennent exclusivement du train externe.

Ces caractéristiques sont entrées dans le **réseau de sélection** et
ses risques KEEP/correction. Aucun poids fixe des audits n'est
déployé et aucun audit historique provenant des folds examinés
n'est injecté dans les prédictions.

## Contrôle de reproduction

Le mode « baseline » reproduit **exactement** les 59 309
prédictions du premier sélecteur par combinaison
(run 37755994598) : **zéro divergence**.

Tous les jobs de la matrice CI ont terminé avec succès.

## Résultats 4 variantes, mêmes 59 309 événements, 7 493 révisables

| Système | Exact-K global | Exact-K poly | Fixes | Régressions | Net vs freeze |
|---|---:|---:|---:|---:|---:|
| Référence `freeze_local_combo` | **81,6976 %** | **34,2586 %** | — | — | — |
| Routeur précédent (baseline) | 81,6672 % | 34,0149 % | 426 | 444 | **−18** |
| + identité des têtes | 81,6487 % | 33,8659 % | 480 | 509 | **−29** |
| + audits par contexte acoustique | 81,6402 % | 33,7982 % | 416 | 450 | **−34** |
| + les deux | 81,6453 % | 33,8389 % | 479 | 510 | **−31** |

### Vrai K2, K3, K4 — bilans correction moins régression

| Variante | K2 | K3 | K4 |
|---|---:|---:|---:|
| Baseline | +24 | **−133** | +2 |
| Identité seule | +34 | **−142** | −5 |
| Audits contextuels seuls | +16 | **−141** | +16 |
| Identité + audits | +17 | **−130** | −8 |

La configuration « les deux » améliore K3 de trois prédictions nettes
par rapport au modèle précédent, mais perd au total 13 bonnes
prédictions supplémentaires sur les K0–K6. Aucune configuration
ne dépasse `freeze_local_combo` sur poly Exact-K.

## Diagnostic de conception

- **Les entrées et les gradients fonctionnent.** La composition
  explicite et les statistiques contextuelles sont réellement
  transmises au sélecteur neuronal, et les tests vérifient que les
  gradients peuvent remonter depuis la décision jusqu'aux nouveaux
  canaux. Ce n'est pas un défaut d'interface muette.
- **Le signal de sélection n'est pas assez discriminant** pour
  séparer les corrections bénéfiques des changements destructeurs,
  particulièrement pour un K3 déjà correctement prédit.
- **Les anciens audits ne sont pas tous connectés.** Les résultats
  de clustering A/B, morphologie dédiée, pitch-shift et signatures
  harmoniques doivent être rejoués/convertis en observables
  événementiels au lieu d'importer des scores agrégés issus
  des folds utilisés en évaluation. Les nouveaux régimes KMeans
  ne sont pas automatiquement équivalents au cluster A/B historique.
- Le protocole antérieur d'OOF peut utiliser des prédictions
  spécialistes dont les modèles ont entraîné certaines lignes
  du fold de réception, malgré exclusion directe des labels
  de ce fold dans l'audit. L'indépendance complète de la
  représentation interne reste à établir par audit d'origine.
- Les folds externes 0/1/2/4 sont **déjà exposés à notre recherche**,
  et le player05/fold3 ne figure pas dans la cohorte examinée ;
  les gains/pertes ne sont pas une validation indépendante.
- Les 26 caractéristiques audio restent un sous-ensemble du
  contexte acoustique, sans signature événementielle validée
  de tous les audits historiques.

## Décision

Les variantes sont **rejetées comme modèles de remplacement**.
`freeze_local_combo` n'a pas changé. Ne pas « corriger » ces pertes
par des seuils sélectionnés sur les mêmes folds exposés.

La prochaine recherche justifiable n'est **pas simplement d'ajouter
de nouvelles variables**, mais d'auditer si les observations
acoustiques associées aux anciens échecs (harmoniques, morphologie,
persistances, compression) sont effectivement calculables sur
chaque nouvel événement et si les votes contradictoires
contiennent suffisamment d'information pour distinguer vrai
K3 d'un faux K3. Les nouvelles données non utilisées sont
nécessaires avant validation/promotion.

## Artifacts

- [Contrôle baseline, reproduction exacte](https://github.com/Andriamarosoa/note/actions/runs/37758789956/artifacts/11540659117)
- [Identité binaire des têtes](https://github.com/Andriamarosoa/note/actions/runs/37758789956/artifacts/11541825637)
- [Audit acoustique par régime](https://github.com/Andriamarosoa/note/actions/runs/37758789956/artifacts/11540924363)
- [Identité + audit acoustique](https://github.com/Andriamarosoa/note/actions/runs/37758789956/artifacts/11540734200)

Sources :
- `scripts/learn_v273_audit_aware_selection.py`
- `scripts/evaluate_v273_audit_aware_selection.py`
- `test/test_v273_audit_aware_selection.py`
- `.github/workflows/v273-audit-aware-selection.yml`
