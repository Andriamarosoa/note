# S1–S4 et repair_coherent : producteurs raccordés aux entrées reconstruites

12 octobre 2026. **S80 non promu.** Référence officielle : `freeze_local_combo`.

**CI indépendante réussie** : [run 38234363815](https://github.com/Andriamarosoa/note/actions/runs/38234363815),
commit `8301b6195cb9546c2e731e025e42922d3f08bbc6`.
Les cinq empreintes de champions historiques et les décisions GOAT sont
vérifiées sur le runner GitHub. [Preuve conservée](evidence/v273-recovered-five-families/github_ci.json).

Les cinq familles auparavant absentes sur nouvel audio sont désormais
exécutables à partir des vrais producteurs restaurés, avec leurs poids et leurs
règles historiques. Le rejeu local retrouve **37 465 décisions de champions**
(cinq familles × 7 493 événements éligibles), sans différence. Les cinq vecteurs
natifs complets de **59 309 événements** ont les mêmes SHA-256 que les champions
S44 attendus par S45. Les **1 101 471 décisions** des 147 politiques S1/S3/S4
rejouées concordent également.

## Ce qui a été raccordé

- Le risque de référence du `CatalogueCritic` local original et les logits
  conditionnels du `ContextualRiskCritic` original. Les deux facteurs étaient
  explicitement croisés dans la source historique ; ils ne sont pas remplacés
  par un seul réseau. Quatre checkpoints de chaque réseau sont utilisés.
- Les 70 spécialistes reconstruits et les 14 correcteurs gelés alimentent les
  255 propositions et les neuf audits, recalculés pour chaque destinataire.
- S1 utilise ses calibrations enregistrées ; S2 ses choix par classe initiale ;
  S3 ses arbres et ses choix gelés ; S4 ses gardes de régression enregistrées.
- `repair_coherent` reçoit ses vrais tenseurs **5 × 64 × 21**, reconstruits avec
  les six votes historiques, les audits et les régimes calculés sur les seules
  références permises. Ses quatre réseaux Keras sont rejoués en NumPy.

La précision d'accumulation float64 évite les variations de calcul entre
machines ; les postérieurs neuraux sont rendus en float32. Sur le rejeu local,
l'écart maximal de probabilité de risque est **3,58 × 10⁻⁷**, celui des logits
**5,73 × 10⁻⁶**. Toutes les décisions finales restent identiques. Aucun modèle
prédictif n'est réentraîné. Les labels historiques servent aux audits des
références hors fold ; les labels des destinataires ne produisent pas leurs
caractéristiques.

## Exécution GOAT et limites

Les cinq familles s'exécutent sur les cinq événements éligibles parmi 57,
dans les **19 branches historiques de calibration**, avec le fold amont
correspondant. Les sorties hors du domaine K2/K3/K4 conservent la référence,
conformément à leur définition historique. Aucun label GOAT n'est lu.

Les 19 branches sont conservées séparément. Aucune identité de morceau
historique n'est attribuée arbitrairement aux nouveaux WAV et aucune règle
finale d'agrégation entre branches n'est choisie ici. GOAT est déjà exposé :
ce résultat est une preuve d'exécution, pas une amélioration indépendante.

Il reste à réunir les **seize familles S45 avec des parents cohérents**, dont
le vrai S29 utilisé par S35, puis à exécuter S46 et la suite S80. Les onze
anciennes familles avaient été exécutées avec des adaptations : leur
raccordement doit utiliser les parents S5–S18 désormais reconstruits.
La promotion exige ensuite un gain indépendant global et polyphonique.

## Preuves et reproduction

- [Runtime](../scripts/recover_v273_missing_s45.py).
- [Workflow unique](../.github/workflows/v273-recovered-five-families.yml),
  avec contrôle des cinq SHA historiques et comparaison locale/CI.
- [Rejeu historique](evidence/v273-recovered-five-families/historical_report.json).
- [Exécution GOAT](evidence/v273-recovered-five-families/external_report.json).
- [Vecteurs historiques](evidence/v273-recovered-five-families/historical_five_families.npz)
  et [branches GOAT](evidence/v273-recovered-five-families/external_five_families.npz).

Le script exige une preuve historique complète produite par exactement la
même version du code avant d'autoriser l'inférence sur de nouvelles entrées.
