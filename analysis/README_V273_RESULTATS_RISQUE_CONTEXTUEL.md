# V27.3 — Apport de correction isolé, estimateur de risque défaillant

**Un apport utile a été isolé et conservé.** Garder l'estimation de risque du
parent local et utiliser la nouvelle estimation conditionnelle de correction
donne **566 corrections / 518 régressions, soit +48 face au freeze**, contre
**+32 pour le parent**. Le nouveau réseau complet donne pourtant −38. Son
bilan global masquait donc un apport de sa partie correction.

Les cinq nouvelles sorties — réseau complet, deux interventions et deux
croisements — restent candidates, y compris celles dont le bilan est négatif.
La mémoire contient maintenant **46 sorties**, représentant **35 vecteurs de
prédiction distincts**. Cela ne signifie pas 46 nouvelles têtes actives : les
catalogues évalués contiennent toujours huit sélections et leurs 255 groupes.

## 1. Modification et exécution

La branche de risque du parent reçoit la moyenne brute des audits. Elle est
donc insensible à certaines réaffectations des audits entre groupes conservant
cette moyenne. La nouvelle branche reçoit aussi la moyenne des 32 activations
non linéaires des groupes : identité, membres, destination, contexte et audits
peuvent ainsi interagir avant la réduction.

Le test de permutation confirme cette différence de capacité. Il ne suffit
pas à prouver que l'information sera bien exploitée sur d'autres événements.

[Protocole fixé avant entraînement](README_V273_PROTOCOLE_RISQUE_CONTEXTUEL.md).
[Run 37815209004 terminé avec succès](https://github.com/Andriamarosoa/note/actions/runs/37815209004),
code exécuté `07e8641072a6d50a2cae753860659fbb10f219c4`.
Les **16 tests CI passent**. Le premier lancement avait échoué sur un jeu de
test dont l'inversion laissait la séquence inchangée ; la permutation du test
a été corrigée sans modifier le modèle ni ajuster ses paramètres sur les résultats.

Même seed 27402, 30 époques, perte, producteurs, propositions et audits locaux
que le parent. Les décisions du parent sont reproduites à poids figés dans les
quatre folds. Les exclusions des producteurs et des références d'audit sont
vérifiées. La variante ajoute 1 024 paramètres, de 13 538 à 14 562 : il ne
s'agit pas d'une comparaison à capacité constante.

## 2. Le réseau complet produit des apports et davantage de régressions

| Mesure | Parent local | Nouveau réseau complet |
|---|---:|---:|
| Corrections face au freeze | 527 | 705 |
| Régressions face au freeze | 495 | 743 |
| Gain net | +32 | −38 |
| Changements de verdict | 1 720 | 2 493 |
| Gain annoncé par les probabilités | +354,45 | +676,94 |

Face au parent, le nouveau réseau conserve 387 corrections, en perd 140 et
en ajoute **318**. Il évite **118** anciennes régressions, en conserve 377 et
en ajoute 366. Cela représente **436 améliorations et 506 dégradations**, net −70.
Ces 436 améliorations restent enregistrées malgré le mauvais bilan global.

Les gains nets face au parent sont −11 / −32 / −7 / −20 sur les folds
0 / 1 / 2 / 4. Par vrai K, la différence vaut K2 −2, K3 −16, K4 **−62**,
K5 +9, K6 +1 ; K0/K1 n'ont aucun gain net. La dégradation concerne donc
particulièrement les vrais K4.

## 3. Ce que révèle l'audit de surestimation

Sur les **mêmes 2 493 décisions changées par le nouveau réseau** :

| Justesse du verdict initial | Taux |
|---|---:|
| Observée | **29,80 %** |
| Estimée par le parent | **28,57 %** |
| Estimée par le nouveau risque | **15,58 %** |

Le nouveau risque sous-estime fortement la proportion de bonnes réponses
qu'il va détruire. La comparaison porte sur un ensemble fixe de décisions,
et non sur deux sous-ensembles différents sélectionnés par les modèles.

Sur tous les événements éligibles, sa moyenne paraît pourtant proche de la
réalité : 33,10 % prédits contre 33,56 % observés. **Une bonne moyenne globale
ne garantit donc pas une bonne estimation du risque sur les décisions choisies.**

Les pertes probabilistes confirment que les deux parties évoluent différemment :

| Perte sur les mêmes événements exclus | Parent | Nouveau |
|---|---:|---:|
| Justesse du verdict initial, BCE | 0,63435 | **0,72080** |
| Bonne alternative conditionnellement à une erreur initiale, CE | 0,67982 | **0,62748** |
| Perte totale par événement | 1,08599 | 1,13767 |

La partie correction s'améliore sur cette mesure, tandis que le risque se
dégrade. Pour le nouveau risque, la BCE après entraînement est comprise entre
0,444 et 0,458 sur les événements d'apprentissage de chaque fold, contre
0,691 à 0,749 sur les événements exclus. Cet écart établit une mauvaise
généralisation dans cette expérience ; il ne désigne pas, à lui seul, une
unique cause physique ou une nouvelle catégorie d'erreurs.

## 4. Les deux croisements de facteurs

Après ce constat, un [protocole de diagnostic](README_V273_PROTOCOLE_CROISEMENT_RISQUE.md)
a été publié au commit `3a488b0d674d957b3e4580b9f30eabc256074a7c`, **avant
le calcul des croisements**. C'est un diagnostic a posteriori sur les mêmes
folds exposés, distinct de l'expérience initialement annoncée.

Les deux croisements sont calculés, sans apprentissage supplémentaire ni
choix par fragment. Chaque facteur provient d'un modèle ayant exclu le fold
du fragment concerné. Les groupes disponibles restent strictement identiques.

Avec `r` la probabilité que le verdict initial soit correct et `q` la
distribution de la bonne alternative lorsque ce verdict est faux :
`P(initial)=r`, `P(alternative)=(1-r)*q`, avec une masse OTHER explicite.
Les probabilités restent normalisées ; aucun veto individuel n'est introduit.

| Risque | Correction conditionnelle | Corrections | Régressions | Net face au freeze |
|---|---|---:|---:|---:|
| Ancien | Ancienne | 527 | 495 | +32 |
| Ancien | **Nouvelle** | **566** | **518** | **+48** |
| Nouveau | Ancienne | 671 | 701 | −30 |
| Nouveau | Nouvelle | 705 | 743 | −38 |

À correction nouvelle fixée, reprendre l'ancien risque améliore le net de
**86**. À risque ancien fixé, la correction nouvelle apporte **+16**. Avec
le nouveau risque, son effet est au contraire −8. Cela confirme la nécessité
d'auditer **la combinaison exécutée**, et de conserver ses apports particuliers.

Le croisement positif améliore 219 décisions du parent et en dégrade 203 :

- 438 anciennes corrections conservées ; **128 nouvelles corrections** ;
  89 anciennes corrections perdues.
- **91 anciennes régressions évitées** ; 404 persistantes ; 114 nouvelles.
- Gains nets par fold 0/1/2/4 : **−2 / +5 / +13 / 0**.
- Gains nets par vrai K face au parent : K2 +11, K3 +6, K4 −6, K5 +5, K6 0.

Son Exact-K global est **81,7785 %**, et son Exact-K poly **34,9086 %**,
sur le même périmètre natif. Ce +48 ne dépasse pas le +60 du catalogue 7 global
ni le +57 de la calibration précédente ; son apport reste néanmoins conservé.
La comparaison n'autorise pas une promotion sur des données inédites.

Sa perte probabiliste totale descend à **1,05122**. La surestimation demeure :
**+354,77 annoncé pour +48 observé**, soit un écart de 306,77. Ce résultat
isole un apport de correction et un estimateur de risque plus utilisable sur
ces cas ; il ne résout pas toute la calibration après sélection.

## 5. Les 126 régressions et 92 corrections du profil défavorable

Le profil du parent contient 341 cas : 126 régressions, 92 corrections et
123 changements neutres. Il est défini par les audits des groupes complets
proposant le verdict choisi, sans utiliser le label pour attribuer le profil.

| Résultat sur ce profil fixe | Nouveau réseau complet | Ancien risque + nouvelle correction |
|---|---:|---:|
| Anciennes régressions évitées | 38 | **39** |
| Anciennes régressions persistantes | 88 | **87** |
| Anciennes corrections conservées | 63 | **68** |
| Anciennes corrections perdues | 29 | **24** |
| Nouvelles corrections | 6 | **5** |
| Gain net face au parent sur ce profil | +15 | **+20** |

Il existe donc une protection partielle mesurée, avec un coût explicite sur
les réussites. Il n'existe pas encore de séparation parfaite entre les deux.
Les 87 régressions restantes ne deviennent pas automatiquement une autre
catégorie : elles conservent ce profil observable et nécessitent un nouvel
audit de leurs distinctions et de l'utilisation des caractéristiques.

Les combinaisons restent présentes. Parmi les 17 erreurs pour lesquelles un
groupe propose le vrai K alors qu'aucune sélection seule ne le propose, le
nouveau réseau complet en corrige 5, contre 4 pour le parent. Ces cas restent
archivés, y compris si une autre variante les manque.

## 6. Interventions à poids figés et conservation

Pour le nouveau réseau complet : neutraliser le vecteur de risque ajouté
donne 316 corrections / 336 régressions, net −20. Neutraliser les cinq canaux
locaux donne 932 / 992, net −60. Leurs sorties sont conservées séparément.
Il s'agit d'interventions sur les poids entraînés du nouveau modèle, pas de
réentraînements ni du parent historique.

La mémoire précédente de **41 sorties** reste inchangée. Cinq sorties s'y
ajoutent, donnant **46 candidates et 35 vecteurs distincts**. Elles corrigent
au moins une fois **98 erreurs initiales supplémentaires** au-delà de l'union
des 41 anciennes sorties : l'union descriptive passe de 1 573 à **1 671**.
Cette union utilise les labels pour diagnostiquer la complémentarité ; elle
ne constitue pas la performance d'un système sachant choisir la bonne sortie.

Le [registre ajouté](evidence/v273-contextual-risk/candidate-registry.json)
conserve les identités, filiations, empreintes, bilans face au freeze et au
parent, y compris pour les variantes négatives. Les verdicts des 59 309
événements, probabilités compactes, coordonnées des 7 493 fragments éligibles
et poids sont enregistrés. Les 58 caractéristiques primaires restent
accessibles par jointure avec la mémoire précédente. Les outils refusent
d'écraser une mémoire existante ou d'ajouter deux fois les mêmes candidates.

Cette conservation n'est pas l'intégration de cinq nouvelles têtes dans un
sélecteur : un tel consommateur devra disposer de prédictions imbriquées
excluant ses propres événements de tous les producteurs. Les poids disponibles
et les règles de recomposition rendent les propositions concrètes et auditables.

## Preuves et reproduction

- [Vérification du réseau et des exclusions](evidence/v273-contextual-risk/verification.json).
  Cette première étape conservait 44 sorties au total ; le croisement suivant
  ajoute les deux dernières.
- [Vérification indépendante des deux croisements et des 46 sorties](evidence/v273-contextual-risk/factor-swaps-verification.json).
- [Résultats par fold et transitions de régression](evidence/v273-contextual-risk/factor-diagnostics/diagnostics.json).
- [Cas des deux croisements avec enregistrement et position](evidence/v273-contextual-risk/factor-diagnostics/paired-factor-cases.csv).
- [Cas du nouveau réseau complet](evidence/v273-contextual-risk/paired-cases.csv).
- [Manifestes de l'artefact](evidence/v273-contextual-risk/artifacts.json).

Les deux tests locaux de recomposition passent, en complément des 16 tests CI.
Les probabilités ont été recomposées indépendamment en NumPy ; les décisions,
bilans et exclusions sont vérifiés. Les fichiers conservés sont contrôlés par
empreinte SHA256. Les sorties initiales/calibrées/locales des étapes précédentes
ne sont ni remplacées ni filtrées selon leur gain global.

```bash
python -m unittest test.test_v273_risk_factor_swaps
python -m scripts.verify_v273_contextual_risk --results ../contextual-risk-results --parent ../audit-memory-results/corrector8 --corrector-cache ../catalogue-results/corrector-cache --features ../selector-review-artifacts/features --memory analysis/evidence/v273-audit-memory/variants --output /tmp/v273-risk-proof-new
python -m scripts.audit_v273_risk_factor_swaps --parent ../audit-memory-results/corrector8/predictions.npz --candidate ../contextual-risk-results/model/predictions.npz --output /tmp/v273-factor-audit-new
python -m scripts.verify_v273_risk_factor_swaps --results /tmp/v273-factor-audit-new --parent ../audit-memory-results/corrector8/predictions.npz --candidate ../contextual-risk-results/model/predictions.npz --memory-output /tmp/v273-risk-proof-new --previous-memory analysis/evidence/v273-audit-memory/variants --protocol-commit 3a488b0d674d957b3e4580b9f30eabc256074a7c
python -m scripts.audit_v273_contextual_risk_outcomes --evidence /tmp/v273-risk-proof-new --output /tmp/v273-factor-diagnostics-new
```

Tous les résultats portent sur les folds déjà exposés 0/1/2/4. Fold3/player05
restent exclus. Aucune validation finale sur des données inédites ni garantie
de blocage de toutes les régressions n'est revendiquée.
