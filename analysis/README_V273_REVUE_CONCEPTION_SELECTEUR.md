# V27.3 — Revue de conception du sélecteur : constats vérifiés

**8 octobre 2026. Revue terminée ; aucun remplacement de `freeze_local_combo`.**

La priorité justifiée par cette revue est de reprendre le contrat d'entrée,
la signification des risques et la décision du sélecteur. L'examen des K3
sert à vérifier ces mécanismes. Les éléments ci-dessous établissent des
défauts ou limites précis ; ils ne prouvent pas que leur correction suffira
à obtenir un gain Exact-K poly.

## 1. Périmètre et vérifications

- Code examiné : commit `aa4ca4b7729a00e19dcfd1a071d6afac09933dc3` de
  `codex/v273-failure-clustering`. Les scripts concernés sont identiques à
  ceux du run `37758789956`, commit `2507836c4ee590fb944fb8d2df0ac401ae0c7704`.
- Deux variantes archivées relues : contrôle `baseline` et version complète
  `both` (identité des têtes + audits par régime).
- Deux rejeux : **zéro divergence sur les 59 309 prédictions de chaque variante**.
- Les caractéristiques des **7 493 événements admissibles** ont été relues
  dans les quatre artifacts du run `37740493178`, avec vérification des
  identifiants, folds, classes et schémas de caractéristiques.
- Les SHA-256 des six ZIP téléchargés ont été vérifiés. Les neuf fichiers
  sources déterminants sont identifiés par leur SHA-256 dans le rapport JSON.
- **Six tests** contrôlent le contre-exemple probabiliste, l'utilité attendue,
  les égalités KEEP, les identifiants, les décomptes et la dépendance OOF.
- Aucun entraînement ni nouveau test indépendant n'a été réalisé dans cette
  revue. Les résultats du rejeu alternatif sont diagnostiques, pas une variante
  entraînée ni une proposition de déploiement.

## 2. Les 14 interfaces ne sont pas 14 experts acoustiques entraînés

| Interface | Producteur réel | Conséquence pour la revue |
|---|---|---|
| H0 | Classe du modèle gelé convertie en probabilités fixes : 0,84 pour cette classe, 0,04 pour chacune des quatre autres classes poly | La confiance événementielle originale du modèle n'est pas transmise ici. |
| H1 | StandardScaler + régression logistique sur 14 caractéristiques `spectral__` | Un spécialiste entraîné. |
| H2 | Même type de modèle sur naissance, persistance et amortissement | Un spécialiste entraîné, avec 29 caractéristiques. |
| H3 | Même type de modèle sur les 58 caractéristiques de reconstruction harmonique | Réutilise notamment les familles vues par H1, H2 et H5. |
| H4 | Même type de modèle sur les 58 caractéristiques calculées avec un gabarit limité aux fondamentales | Autre vue du même extracteur ; ce n'est pas un réseau indépendant identifiant chaque fondamentale. |
| H5 | Même type de modèle sur sources et cohérence | Un spécialiste entraîné, avec 15 caractéristiques. |
| C23, C32, C34, C43 | Moyenne des probabilités de destination de H1–H5, multipliée par 0,95 ; activée selon K initial | Adaptateurs déterministes, sans apprentissage propre de leur transition. |
| F_keep2, F_keep3, F_keep4 | Valeur KEEP fixe à 0,98 lorsque la classe initiale correspond | Aucun diagnostic acoustique propre. |
| F_keep_any | Valeur KEEP fixe à 1 | Aucun diagnostic acoustique propre. |

Le sélecteur et l'arbitre sont bien neuronaux et entraînés. Ce registre
précise la nature de leurs entrées. Les familles acoustiques sont des
résumés d'activations d'un dictionnaire de gabarits harmoniques fixes ;
plusieurs spécialistes partagent donc le même traitement acoustique initial.
Cette dépendance de conception est vérifiée, mais son coût prédictif n'est
pas isolé par cette revue.

Sources : [producteurs H0–H5](../scripts/evaluate_v273_dynamic_heads.py),
[adaptateurs](../scripts/evaluate_v273_neural_history_mix.py),
[extracteur](../scripts/extract_v273_harmonic_trajectory.py).

## 3. Défaut confirmé : aucune caractéristique spectrale dans le contexte direct

`load_features` trie les noms alphabétiquement. Le sélecteur conserve ensuite
les 26 premiers noms des familles naissance/amortissement/persistance/spectre.
La coupure intervient avant d'atteindre `spectral__`.

| Famille | Disponible dans le schéma | Transmise directement au sélecteur et au KMeans |
|---|---:|---:|
| Naissance | 11 | 11 |
| Amortissement | 11 | 11 |
| Persistance | 7 | 4 |
| Spectre | 14 | **0** |
| Sources | 8 | 0, hors filtre prévu |
| Cohérence | 7 | 0, hors filtre prévu |

Le calcul a été exécuté avec l'expression de sélection du code et le schéma
des artifacts réels. La description « 26 caractéristiques spectrales,
naissance, persistance, amortissement » est donc inexacte pour le contexte
direct. **Il reste une information spectrale indirecte via H1/H3/H4 et leurs
votes** : le système complet n'est pas dépourvu de spectre.

Correction de contrat à tester : remplacer cette troncature par une liste
explicite et contrôlée. Un essai préenregistré peut comparer les 26 entrées
actuelles aux 43 entrées des quatre familles, avec un contrôle de capacité.
Le gain de cette modification n'est pas mesuré ici.

Sources : [tri](../scripts/summarize_v273_harmonic_global.py),
[filtre et KMeans](../scripts/evaluate_v273_audit_aware_selection.py).

## 4. Défaut de signification du score correction/KEEP

Le score actuel est :

```text
score(cible) = log(P_corriger(cible)) - log(P_régresser(cible))
score(KEEP)  = log(P_KEEP) - log(1 - P_KEEP)
```

Ces deux expressions ne correspondent pas à une comparaison commune
d'espérance Exact-K. Même avec des probabilités parfaites, elles peuvent
préférer une réponse moins probable.

Contre-exemple construit, explicitement distinct des événements audio :

| Vérité possible | K2 | K3 | K4 | K5 | K6 |
|---|---:|---:|---:|---:|---:|
| Probabilité | 0,30 | **0,40** | 0,20 | 0,06 | 0,04 |

Avec une prédiction initiale K3, chaque changement a un risque de régression
de 0,40. Le score K2 vaut `log(0,30/0,40) = −0,2877`, tandis que KEEP vaut
`log(0,40/0,60) = −0,4055`. Le décodeur choisit K2, bien que K3 soit plus
probable. Une comparaison d'espérance Exact-K conserverait K3.

Sur la version complète réelle, **540 des 1 687 changements** ont une
probabilité de correction estimée inférieure à leur probabilité de
régression estimée. Ces 540 changements comprennent **158 corrections,
183 régressions et 199 changements neutres**. Ce décompte montre la portée
de la discordance avec l'interprétation « gain attendu » ; il ne prouve
pas que les probabilités sont calibrées.

Deux autres ambiguïtés sont présentes :

- La cible `keep_correct` vaut aussi 1 pour les vrais K0/K1 alors que la
  prédiction initiale est K2/K3/K4 : elle signifie donc « KEEP préféré faute
  de sortie autorisée », et pas toujours « classe initiale correcte ».
  Cela concerne **2 041 lignes d'entraînement admissibles** sur les 7 493.
- Le label de régression est identique pour toutes les destinations d'un
  événement (`vérité == classe initiale`), mais les cinq scores sont appris
  séparément. Dans la version complète, l'écart maximal entre les risques
  des destinations légales dépasse 0,10 pour **4 418 événements**. C'est une
  incohérence entre estimations d'un même événement, pas une preuve à elle
  seule de la cause des erreurs.

Enfin, la classification pondérée et les pertes auxiliaires entraînent
conjointement ces probabilités. Elles ne peuvent pas être traitées comme
des probabilités calibrées sans mesure supplémentaire.

Source : [cibles, objectif et décodeur](../scripts/learn_v273_transition_combo_risk.py).

### Rejeu contrôlé : changer uniquement la formule ne résout pas le problème

Une seule alternative a été examinée : `P_corriger − P_régresser`, avec
KEEP de valeur zéro et KEEP en cas d'égalité. Aucun seuil n'a été ajusté
aux labels, et aucun modèle n'a été réentraîné.

| Variante / décodeur | Exact-K global | Exact-K poly | Corrections | Régressions | Net vs gelé |
|---|---:|---:|---:|---:|---:|
| Référence gelée | 81,6976 % | 34,2586 % | — | — | — |
| Contrôle, original | 81,6672 % | 34,0149 % | 426 | 444 | −18 |
| Contrôle, rejeu gain attendu | 81,6335 % | 33,7441 % | 453 | 491 | **−38** |
| Identité + audits, original | 81,6453 % | 33,8389 % | 479 | 510 | −31 |
| Identité + audits, rejeu gain attendu | 81,5677 % | 33,2160 % | 548 | 625 | **−77** |

Pour la version complète, le bilan K3 passe de −130 à +81, mais K2 passe
de +17 à −195 et K4 de −8 à −36. **Le déplacement des erreurs entre classes
est mesuré ; ce rejeu est rejeté comme remplacement.** Les têtes ayant été
entraînées avec l'ancienne formule, cette intervention ne mesure pas ce que
donnerait un apprentissage cohérent depuis le départ.

## 5. Défaut confirmé du descripteur de composition

Le canal 5 est décrit comme la présence de Cxy dans une combinaison.
`compute_direct_options` le remplit en réalité avec la disponibilité de Cxy
pour toute la transition. Les 32 sous-ensembles qui excluent Cxy portent
donc eux aussi la valeur 1 sur chacune des quatre transitions concernées.

Le contrôle des cas structurels K2/K3/K4 dénombre **128 positions erronées
sur 512 positions légales**. Ce sont des positions de descripteurs sur trois
cas représentatifs, pas 128 erreurs audio. Les sept bits d'identité de la
variante complète indiquent correctement l'appartenance, mais ne corrigent
pas le canal ancien. La correction locale est d'utiliser le bit
`COMBO_BINARY[:, 6]` lorsque Cxy est disponible.

Source : [construction des sous-ensembles](../scripts/learn_v273_transition_combo_risk.py).

## 6. Provenance OOF : dépendance interne confirmée par les partitions

Exemple avec le fold externe 0 : pour donner un audit à une ligne du fold 1,
on utilise les sorties OOF des folds 2 et 4. Mais les spécialistes qui
produisent les sorties du fold 2 ont été entraînés sur 1 et 4 ; ceux qui
produisent les sorties du fold 4 ont été entraînés sur 1 et 2.

Le retrait des labels du fold 1 au moment de compter l'audit ne retire donc
pas leur contribution aux modèles producteurs des votes de référence.

Les **24 chemins** `(fold externe, fold récepteur, fold de référence)` ont
été reconstruits avec les identifiants réels et le masque d'entraînement
`vrai K >= 2`. Chacun contient des identifiants du fold récepteur dans
l'entraînement du producteur de référence : selon le fold, **1 191, 1 287,
1 481 ou 1 493 identifiants**. Le JSON conserve les comptes et empreintes
des ensembles concernés.

**Aucun identifiant du fold externe n'apparaît dans ces ensembles.** Cette
revue confirme une dépendance interne des descripteurs d'audit ; elle ne
démontre ni une fuite de labels du test externe ni son effet numérique sur
le score final. La séparation des seules statistiques d'audit, testée dans
les tests antérieurs, ne suffit pas à vérifier toute la chaîne de producteurs.

Correction à prévoir : construction des audits avec exclusion du fold
récepteur dès l'entraînement des producteurs, et manifeste de provenance
vérifié à chaque étage. Il faut également mesurer la réduction des données
disponibles à chaque étage du double cross-fitting.

## 7. Limite structurelle du périmètre de correction

| Mesure sur la cohorte réelle | Événements |
|---|---:|
| Vrais poly K2–K6 | 7 385 |
| Vrais poly dans le périmètre corrigible | 5 452 |
| Erreurs poly hors de ce périmètre | **1 918** |
| Dont vrais poly initialement prédits K0/K1 | **1 889** |
| Vrais K0/K1 dans les 7 493 lignes admissibles | 2 041 |

Même un correcteur parfait sur son périmètre actuel ne dépasserait pas
**5 467 / 7 385 = 74,0284 % d'Exact-K poly** sur cette cohorte : 5 452 vrais
poly admissibles, plus 15 vrais poly déjà corrects hors périmètre.
Cette borne utilise les labels uniquement pour le diagnostic ; ce n'est
ni une prévision de performance ni un score atteignable démontré.

L'ouverture des événements initialement prédits K0/K1 et l'autorisation
de destinations K0/K1 sont **deux changements différents**. Le premier
permettrait de rechercher des accords sous-comptés ; le second de corriger
des faux accords. Les deux nécessitent des signaux entraînés et une mesure
du risque, notamment sur la très grande population réellement K0/K1.

## 8. Ordre de travail proposé à partir de ces preuves

1. **Rendre les entrées et les contrats exacts.** Liste explicite des
   caractéristiques, appartenance réelle de Cxy, registre des producteurs,
   distinction entre confiance mesurée et constante. Reproduction de
   l'ancien système conservée comme contrôle.
2. **Rendre l'entraînement des audits indépendant du fold récepteur.**
   Vérification des identifiants de tous les producteurs avant comparaison
   de nouveaux mécanismes de sélection.
3. **Réentraîner une décision avec une signification probabiliste commune.**
   Séparer « classe initiale correcte » de « abstention faute de classe
   autorisée ». Comparer les choix sur une même utilité, sans double
   comparaison ratio/odds. Le rejeu négatif ci-dessus interdit de présenter
   un changement de formule à poids fixes comme solution démontrée.
4. **Utiliser les matrices pour départager ces changements.** Mesurer les
   corrections, régressions, cas neutres et effets par vrai K, y compris K2
   et K4. Les 278 K3 détruits ne doivent pas devenir le seul objectif.
5. **Étendre la couverture seulement après mesure.** Distinguer les vrais
   poly actuellement bloqués par la classe initiale des sorties basses
   encore sans spécialiste.

Un jeu final réellement neuf doit être réservé dès maintenant, par
enregistrement et musicien, sans être utilisé pour choisir ces modifications.
Les artifacts de cette revue ne contiennent aucun tel jeu. Une amélioration
sur les folds 0/1/2/4 resterait exploratoire et ne justifierait pas seule une
promotion. Aucun gain n'est annoncé pour player05/fold3.

## 9. Livrables et reproduction

- [Rapport chiffré et empreintes](evidence/v273-selector-design-review/report.json).
- [278 régressions K3 de la version complète](evidence/v273-selector-design-review/both-k3-regressions-summary.csv).
- [249 régressions K3 du contrôle](evidence/v273-selector-design-review/baseline-k3-regressions-summary.csv).
- [Script reproductible](../scripts/audit_v273_selector_design.py).
- [Tests ciblés](../test/test_v273_selector_design_audit.py).
- [Workflow de reproduction et export des matrices complètes](../.github/workflows/v273-selector-design-review.yml).

Les matrices complètes contiennent 7 493 lignes par variante, les
caractéristiques observables, les risques archivés et les identifiants
enregistrement/échantillon. Les exports K3 complets sont également produits.
Ils sont déposés dans l'artifact CI afin de ne pas alourdir l'historique Git.
Les fichiers résumés versionnés permettent de retrouver chaque événement.
Les artifacts d'entrée ne contiennent pas les extraits audio, la liste des
fondamentales candidates ni les postérieurs individuels des spécialistes ;
ces éléments ne sont donc pas présentés comme audités ici.

```bash
python -m unittest -v test.test_v273_selector_design_audit
python -m scripts.audit_v273_selector_design \
  --artifacts model/selector-review \
  --features model/selector-review/features \
  --output model/selector-design-review
```

Répertoire d'entrée attendu : `baseline/` et `both/` avec leurs NPZ du run
37758789956, puis `features/` avec les quatre `rows.jsonl` du run 37740493178.
Le workflow récupère ces artifacts, exécute le rejeu et compare les résultats
aux preuves versionnées. Il ne lance aucun entraînement et ne modifie pas
les poids de référence.

**Décision : revoir le système est prioritaire, avec des défauts désormais
localisés et reproductibles. Aucun mécanisme de remplacement n'est encore
validé.**
