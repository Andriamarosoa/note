# S837 — Autopsie de la première tentative (sans suppression de résultats)

**Run réel :** https://github.com/Andriamarosoa/note/actions/runs/39997037972
**Artifact GitHub :** S837_REAL5_SIX_AXIS_CV_POLICY_GRID_AND_ERROR_AUDIT, ID 12520891182.
**Status :** échec technique après deux politiques mesurées ; aucune promotion.

## Pourquoi les deux premières solutions ont régressé

Données : 5 WAV originaux signés GuitarSet, un `_comp` par composition BN1/Funk2/Jazz1/Rock3/SS1, exactement 20 062 décisions réelles en pas PCM256. Source S708/SHADOW S834 et modèle d'origine préservés. Les politiques sont sélectionnées sur les **quatre compositions autres que l'OUTER**, et appliquées sans modifier aucune autre tête.

### Exact-K (accuracy tous les blocs)
Témoin **56,599541 %**, politique sélectionnée par composition **56,270561 %**, soit **−0,328980 point** et **−66 blocs correctement comptés** au total. Détail :
- BN1 et Funk2 : politique originale retenue, aucun changement.
- Jazz1 : `conf0.50_s0.40`, **−22** blocs corrects.
- Rock3 : `conf0.50_s0.55`, **−44** blocs corrects.
- SS1 : `conf0.50_s0.40`, aucun changement.

Conclusion prouvée : l'option consistant à remplacer une faible confiance du compteur K réseau par un nombre de sustains à seuil fixe ne transfère pas de manière uniformément favorable aux compositions exclues. La *cause acoustique détaillée* de ces 66 blocs reste à démontrer ; une table future par vrai K0–K6, surcomptage/souscomptage et changements d'identité est obligatoire.

### Active Exact-K (accuracy des blocs actifs)
Témoin **46,772514 %**, variante sélectionnée **45,450286 %**, soit **−1,322228 point**, soit **−254 blocs corrects** net :
- BN1 : `on0.65_off0.30` **−112** corrects (−3,3058 points composition).
- Funk2 : `reference`, inchangé.
- Jazz1 : `on0.65_off0.30` **+31** corrects (+0,8329 point).
- Rock3 : `on0.65_off0.15` **+21** corrects (+0,5136 point).
- SS1 : `on0.65_off0.15` **−194** corrects (−4,7456 points).

Conclusion prouvée : **les gains de Jazz1 et Rock3 ne compensent pas les pertes BN1 et SS1**. La sélection sur les autres compositions surestime la transférabilité de `on=.65`. Si une sélection apporte un petit gain mais régresse sur un autre OUTER, cette famille ne peut pas être promue telle quelle. L'analyse des erreurs par vrai K0–K6 et des transitions d'état sur les échantillons est ajoutée à la boucle suivante pour distinguer sous-comptage/surcomptage.

## Interruption technique Onset : aucun score Onset disponible sur cette tentative

Le run s'arrête après la calibration Exact-K et Active-K, par :
```
AssertionError: private S837 evaluation changed S834 baseline onset
00_BN1-129-Eb_comp ({'tp':73,'fp':36,'fn':60},
                     {'TP':73,'FP':36,'FN':60})
```
Les **nombres sont strictement identiques**. L'échec vient exclusivement de la différence de noms des clés (`tp/fp/fn` vs `TP/FP/FN`). Il ne prouve aucune divergence Onset, et ne doit pas être enregistré comme régression musicale.

## Corrections apportées après audit

1. Normaliser les clés de compteurs pour comparer les *nombres exacts*, pas l'orthographe, dans `original_for_axis`.
2. Ajouter `audit_count_change` : identité bloc par bloc des cas récupérés et perdus, matrice par vrai K0–K6, candidats surcomptant/souscomptant, cas corrects conservés et exemples perdus, **après** la sélection cross-composition gelée (pas de fuite OUTER).
3. Conserver `audit_event_change` pour les gains/pertes de TP Onset/Offset, nouvelles/supprimées FP, registre, association harmonique non causale.
4. Rejouer mêmes cinq WAV avec les mêmes poids et mêmes politiques et laisser le veto de promotion systématiquement actif.
5. **Ne pas confondre une première grille exploratoire avec la résolution de la lacune fondamentale des têtes**. Ces expériences sont des diagnostics, pas un +80 % revendiqué.

**Interdit** : supprimer le run échoué, déclarer le +80 %, changer plusieurs têtes simultanément, ou sélectionner des hyperparamètres directement sur la composition OUTER.
