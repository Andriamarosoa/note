# S83–S85 — Pipeline audio externe natif jusqu'aux 36 politiques PR16

**11 octobre 2026. Succès des deux jobs [GitHub Actions 38176406465](https://github.com/Andriamarosoa/note/actions/runs/38176406465).** Ce test est une inférence réelle sur **audio externe**, pas encore une **validation indépendante S80**.

## Exécution et preuves

- GOAT commit figé `55c6e6668ef4546e4635c5e7eaad3062cf7f036b` : `example1_di.wav` et `example2_di.wav`. Seules les **six premières secondes** sont interprétées dans cette prévalidation. Aucune annotation MIDI ouverte.
- V8.4 gelé restauré depuis `v273-source-recovery-35086829802`, poids SHA256 `499ff06dc3e341b2d73fbe80574d10d6a8111915c0b23bfedf4d2f584a9584c5`.
- Détecteur V8.6 avec seuil gelé **0,15** et regroupement V9.0 de **40 ms** : 168 candidats et 4 groupes natifs pour l'exemple 1 (3 exploitables), 477 candidats et 58 groupes pour l'exemple 2 (54 exploitables). Total **57 événements audio exploitables**. SHA256 `native_groups.npz` : `4eb230fcbbee7d68482c64d35f4ce87647553d01f04085a338b10657d54572fa`.
- PR16 fold0 : **12 réseaux originaux**, 57×12×7 probabilités calculées sur ces événements natifs (et non les candidats spectral-flux provisoires S82). Preuve `v273-s83-native-v84-pr16-external`.
- Chaîne **V8.4 → V8.6/87/88 → freeze_local_combo** : vrai modèle de comptage gelé, sans transplantation, poids SHA256 `b69e64ac2c1253364013bf750cd1fe7f2b623e4ecc4abbb5792c16c22a3010e9`. **62 groupes** natifs au total, dont **57 parfaitement alignés par horodatage** avec PR16 ; sorties K selon la distribution **48/9/4/1/0/0/0** sur ces 62 groupes. Preuve `v273-s84-native-external-frozen-reference`.
- **S85** : reconstitution des **36 politiques PR16** depuis les 12 réseaux et le K de la référence. Leur définition exacte (12×{0,6;0,8;1,0}) a été validée **sur toutes les 59 309 lignes historiques et les 36 colonnes**, égalité stricte. La même formule a produit 57×36 votes sur les nouveaux audios. La politique n°22 (réseau 7, coefficient 0,8) propose **4 changements** par rapport au parent `freeze_local_combo`. Aucune correction ne peut être qualifiée sans vérité terrain. Artefact `v273-s85-original-pr16-external-policies`, SHA256 `fe6434e697f912551aecbdc608a7abfdc505d1b8ce59a858bbb505e81f0a95c7`.

## Où la vraie validation de S80 bloque encore

**S80 ≠ vote PR16**. S80 exige simultanément :
1. les 193 sorties **S46**, construites depuis 16 familles de propositions S43/S44 et 304 modèles de risque S45 spécialisés sur des compositions tenues à l'écart, **sans modèle S45 unique défini pour un nouveau morceau** ;
2. les sélecteurs parent **S69→S70→S71→S72**, avec leurs paramètres et décisions non déployables directement depuis les matrices OOF historiques ;
3. les quatre modèles K1 S80 entraînés sur folds historiques, avec quatre seuils **0,50/0,70/0,65/0,65** ; une règle de déploiement pour nouvelle composition doit être fixée avant labels.

Une moyenne ou remplacement par des prédictions PR16/S45 simulées **changerait le modèle**. Ne jamais prétendre que les 57 événements ont été évalués par S80. Aucun Exact-K indépendant ni test de promotion ne peut être déclaré jusqu'à la chaîne complète et les labels gelés.

## Livrables GitHub

- `scripts/s83_external_native_v84_groups.py`
- `scripts/s84_external_freeze_native_count.py`
- `scripts/s85_external_pr16_policies.py`
- `.github/workflows/v273-s74-promotion-gate.yml`
- [Run vérifié](https://github.com/Andriamarosoa/note/actions/runs/38176406465)
- Identifiants artefact du run : S83, S84, S85, et audit original gelé (consultables sur le run).

**Verdict : première moitié du pipeline S80 externe démontrée, score S80 indépendant NON MESURÉ. Référence officielle `freeze_local_combo` préservée.**

### Travail indispensable suivant

Construire un artefact d'inférence complet et invariant par morceau des **16 familles S45/S46**, puis de S69/S70/S71/S72 et fixer la sélection du modèle K1 S80. Valider l'égalité sur les anciennes décisions sans se servir de ces labels pour de nouveaux choix. Sceller les prédictions sur un corpus réellement nouveau (benchmark étendu IDMT-SMT-Guitar), puis seulement ouvrir les annotations, calculer Exact-K0..6 et comparer au témoin.
