# V27.3 — Pourquoi le sélecteur surestime ses corrections

Audit terminé : les deux rejeux à poids figés ont réussi, puis les 88
instantanés ont été vérifiés. Le résultat actuel reste **584 corrections,
546 régressions, soit +38 net**, pour **+397,42 annoncé** par les probabilités
du catalogue à huit sélections. Aucun nouveau modèle n'a été entraîné.

Le constat central est un **fort écart de généralisation du sélecteur**.
Ses décisions réussissent sur les événements utilisés pour son apprentissage.
Sur les folds exclus, le bénéfice réel s'effondre, tandis que le bénéfice
annoncé reste presque identique. La sélection des scores positifs concentre
ensuite les erreurs optimistes. La formule du score est cohérente ; le seul
nombre de combinaisons et l'écart de taille des producteurs ne suffisent pas
à expliquer les résultats observés.

## 1. Ce qui a été rejoué

- Modèles et poids du [run d'apprentissage 37789646292](https://github.com/Andriamarosoa/note/actions/runs/37789646292),
  commit `3f83c2b07308c9114f1db0b5a904c0d80a91b91a`.
- [Rejeu terminé 37797488453](https://github.com/Andriamarosoa/note/actions/runs/37797488453),
  commit `78cbade5ffaa168bb959eb381a9124f25742d56a`.
- Deux bras : contrôle à sept sélections et catalogue avec S8.
- 7 493 événements éligibles, folds 0/1/2/4 uniquement. Fold3/player05 exclus.
- Poids figés, aucun optimiseur, calibrateur, nouveau seuil ou choix du meilleur
  modèle. Les historiques locaux restent désactivés comme dans le modèle audité.

Chaque événement est évalué une fois par le modèle dont il était exclu, et
trois fois par les modèles qui l'ont vu à l'entraînement. Les 22 479 lignes
d'entraînement sont donc des évaluations événement/modèle, **pas 22 479
événements indépendants**. Les événements et la prévalence du verdict initial
correct sont les mêmes dans les deux agrégats : 33,565 %.

Les décisions externes reproduisent toutes les décisions archivées. Les
sorties S8 sont identiques bit à bit. Pour le contrôle, elles peuvent différer
de `4,47e-08` au maximum sur le fold 0, sans changement de décision.

## 2. La confiance ne suit pas la chute du résultat

Le gain net vaut le nombre de corrections moins le nombre de régressions.
Un changement faux vers faux vaut zéro, même si le verdict change.

| Modèle / population | Changements retenus | Net annoncé pour 100 changements | Net réalisé pour 100 changements |
|---|---:|---:|---:|
| S8, lignes vues à l'entraînement | 6 139 | +21,33 | +31,16 |
| S8, folds exclus | 1 896 | +20,96 | +2,00 |
| Contrôle 7, lignes vues à l'entraînement | 5 694 | +21,56 | +31,31 |
| Contrôle 7, folds exclus | 1 800 | +21,29 | +3,33 |

La surestimation externe existait donc avant S8. Ce n'est pas non plus une
surestimation uniforme sur toutes les données : le réseau sous-estime le
bénéfice de ses décisions sur les lignes vues à l'entraînement.

Les pertes probabilistes calculées avec les **mêmes poids finaux** confirment
l'écart. Une perte plus basse est meilleure. La perte conditionnelle concerne
les événements où le verdict initial est faux ; elle répartit la probabilité
entre les autres verdicts disponibles et OTHER, qui signifie qu'aucun ne convient.

| S8 | Perte jointe par événement | Perte de justesse du verdict initial | Perte conditionnelle si verdict initial faux |
|---|---:|---:|---:|
| Entraînement | 0,7633 | 0,5600 | 0,3059 |
| Folds exclus | 1,0775 | 0,6317 | 0,6710 |

La perte conditionnelle fait plus que doubler. Pour le contrôle 7, elle passe
de 0,2881 à 0,6234 : le même phénomène est présent.

Les ensembles de verdicts disponibles peuvent varier selon les producteurs.
Pour vérifier que cette différence ne suffit pas à créer le constat, on
restreint aussi la comparaison aux **3 863 événements S8 dont l'ensemble des
verdicts disponibles est identique dans les quatre modèles**. La perte
conditionnelle passe encore de 0,2722 à 0,6142 et la perte jointe de 0,7327 à
1,0358. L'écart persiste sur cette population comparable.

Ces mesures établissent une mauvaise généralisation finale. Elles sont
compatibles avec du surapprentissage et/ou un transfert insuffisant entre
enregistrements ; elles n'identifient pas à elles seules une cause unique.
Seuls les poids finaux sont disponibles : on ne peut pas dater le début
du problème au cours des 30 époques.

## 3. Le réseau surestime surtout la capacité à corriger

Sur les 1 896 changements externes retenus par S8 :

| Quantité | Somme annoncée | Nombre réel |
|---|---:|---:|
| Corrections | 872,92 | 584 |
| Régressions | 475,50 | 546 |
| Gain net | +397,42 | +38 |

L'excès de gain annoncé est de **359,42**. Arithmétiquement, 288,92 viennent
des corrections surestimées et 70,50 des régressions sous-estimées. Cette
décomposition des résultats n'est pas une attribution causale aux branches
du réseau.

Sur les changements dont le verdict initial était effectivement faux, la
probabilité conditionnelle annoncée de réussir la correction est de
**61,71 %**, pour **43,26 %** observés. À l'entraînement, ces valeurs sont
64,12 % et 71,21 % : la confiance change peu, la réussite beaucoup.

Les 1 350 changements externes partant d'un verdict faux se répartissent ainsi :

- 584 corrections réussies ;
- **625 événements sans aucun autre verdict correct disponible** ;
- 141 événements avec un verdict correct disponible, mais un mauvais choix.

Le réseau ne distingue donc pas assez les cas corrigeables des cas où il
faudrait conserver le verdict faute d'alternative utile. Dans cette population
retenue, il annonce 32,68 % de OTHER, contre 46,30 % observés. Une correction
peut être impossible avec les propositions présentes, même si son score est positif.

La branche estimant la justesse du verdict initial contribue également :
elle annonce 25,08 % de verdicts initiaux corrects parmi les changements
retenus, contre 28,80 % observés. Chaque changement sur ces événements crée
nécessairement une régression.

## 4. Retenir les scores positifs révèle une mauvaise calibration

Sur la même population de 5 231 événements ayant au moins une alternative,
on compare les politiques descriptives suivantes. Aucun de ces diagnostics
ne devient une nouvelle règle de décision.

| Population / choix | Événements | Gain moyen annoncé | Gain moyen observé |
|---|---:|---:|---:|
| Choix uniforme parmi les verdicts distincts, sans filtre | 5 231 | −0,0964 | −0,0731 |
| Meilleur score, sans filtre | 5 231 | −0,0681 | −0,0635 |
| Meilleur score positif, changements retenus | 1 896 | +0,2096 | +0,0200 |
| Meilleur score non positif, changements rejetés | 3 335 | −0,2260 | −0,1109 |

La moyenne du meilleur score sur toute la population semble presque juste.
Elle masque pourtant des erreurs de signes opposés : les décisions retenues
sont trop optimistes, les décisions rejetées trop pessimistes. **Une moyenne
globalement proche du résultat ne prouve pas la calibration des décisions
effectivement prises.**

Le catalogue contient 255 groupes, mais les logits des groupes proposant le
même K sont moyennés avant la probabilité finale. Il ne s'agit donc pas de
prendre le maximum de 255 probabilités de verdicts indépendantes. Les 7 493
événements présentent respectivement 0/1/2/3/4 autres verdicts distincts dans
2 262/3 854/1 235/139/3 cas.

Même avec **un seul autre verdict disponible**, les 1 280 changements retenus
annoncent +20,66 net pour 100, contre **+0,86 réel**. Il n'y a alors aucun
choix du maximum entre plusieurs destinations. Cela exclut cette maximisation
comme explication suffisante, sans exclure un effet des descripteurs des
groupes sur l'apprentissage.

Les scores élevés ne sont pas fiables non plus : sur les 140 décisions au
score supérieur ou égal à 0,5, le gain moyen annoncé est 0,623 contre 0,093
réel. Les intervalles complets sont publiés dans la preuve JSON ; aucun seuil
n'a été choisi à partir de ces résultats.

## 5. Deux hypothèses vérifiées directement

### La formule du score et la perte sont cohérentes

Avec `r = P(verdict initial correct)` et
`q_K = P(K correct | verdict initial faux)`, changer vers K a pour gain prévu :

```text
g(K) = (1 − r) × q_K − r
```

La perte est `BCE(r, B) + (1 − B) × CE(q, cible)`, avec
`B = 1` si le verdict initial est correct. Elle est exactement la
log-vraisemblance négative d'une distribution jointe : verdict initial,
autres verdicts disponibles, ou OTHER. Le vérificateur reconstruit cette
distribution depuis les logits et contrôle l'identité de perte ainsi que
les décisions depuis les probabilités sauvegardées.

Il y a une contribution par événement. Les 255 groupes ne deviennent pas
255 exemples indépendants dans la perte. Apprendre `q` seulement quand le
verdict initial est faux est cohérent avec sa définition conditionnelle.
Cela ne garantit cependant pas la calibration sur de nouveaux événements.

Pour mémoire, l'écart de gain sur les décisions retenues se décompose aussi
exactement en `(B−r)×(1+q) + (1−B)×(q−Y)`, où Y indique que le K choisi est
correct. Ces termes valent 110,28 et 249,14. C'est une identité algébrique
avec un conditionnement fixé, **pas une mesure de l'effet causal d'une
correction de chaque branche**.

### L'écart de régime des producteurs n'explique pas tout

Le sélecteur apprend sur des votes issus de producteurs ajustés sur deux
folds ; à l'évaluation normale, les producteurs utilisent trois folds.
Les historiques changent aussi de régime. Le protocole rejoue chaque modèle
figé en alignant les votes, les historiques, ou les deux sur le régime
d'entraînement, toujours en excluant le fold évalué.

Les trois omissions possibles sont toutes conservées. Les lignes ci-dessous
pour les interventions sont des moyennes sur ces omissions ; elles ne
décrivent **ni un ensemble de modèles ni une nouvelle performance déployable**.

| S8, votes / historiques | Perte conditionnelle | Net annoncé pour 100 changements | Net réel pour 100 changements | Net réel équivalent sur 7 493 événements |
|---|---:|---:|---:|---:|
| Normal, 3 / 3 folds | 0,6710 | +20,96 | +2,00 | +38,00 |
| Votes alignés, 2 / 3 folds | 0,6882 | +20,62 | +1,18 | +22,33 |
| Historiques alignés, 3 / 2 folds | 0,6720 | +20,84 | +1,28 | +25,00 |
| Deux régimes alignés, 2 / 2 folds | 0,6893 | +20,75 | +0,91 | +17,33 |

La surestimation persiste dans toutes les conditions. Le contrôle 7 présente
le même comportement. Ces interventions ne démontrent pas qu'un entraînement
homogène n'aurait aucun effet ; elles montrent qu'aligner les entrées à poids
figés ne résout pas l'écart et ne permet pas d'en faire la cause unique.

## 6. Priorité justifiée par l'audit

La priorité est de contrôler la généralisation et la confiance du sélecteur.
L'ajout d'une sélection peut apporter des verdicts utiles ; il ne corrige pas
automatiquement ce défaut déjà présent avec sept sélections.

Le code actuel entraîne pendant 30 époques fixes, sans suivi d'une validation
interne pour arrêter l'apprentissage, et sans calibration externe du score
final. Les blocs denses du sélecteur n'utilisent ni dropout ni pénalité L2.
Ce sont des choix à examiner par expériences contrôlées, **pas des causes
démontrées individuellement par ce rejeu**.

La suite expérimentale devrait :

1. Réserver une validation interne au sélecteur lui-même pour choisir la durée
   d'apprentissage et tester une régularisation. Les votes hors fold des
   producteurs ne rendent pas hors entraînement les lignes vues par le sélecteur.
2. Mesurer puis, si nécessaire, calibrer les scores sur des prédictions du
   sélecteur obtenues sans avoir appris sur les labels correspondants. Toute
   calibration et tout choix de règle doivent rester à l'intérieur du protocole
   d'entraînement, avant l'évaluation externe.
3. Évaluer la calibration **après le choix du verdict et sur les décisions
   retenues**, en suivant séparément corrections, régressions et OTHER.
4. Valider enfin sur des données inédites. Les folds de cet audit ont déjà
   servi au développement et ne fournissent pas cette preuve indépendante.

Une calibration peut réduire des changements injustifiés ; elle ne crée pas
une capacité à distinguer les bonnes corrections que le réseau n'aurait pas
apprise. L'audit ne valide donc ni une correction automatique de la confiance,
ni une nouvelle architecture, ni l'activation des historiques locaux.

## 7. Preuves et reproduction

- [Protocole figé](README_V273_PROTOCOLE_AUDIT_CONFIANCE.md).
- [Vérification et métriques complètes](evidence/v273-confidence-audit/verification.json) :
  identifiants d'artefacts, SHA-256, 44 instantanés par bras, pertes par fold,
  calibration, omissions et contrôles de population.
- [Comparaison des dix conditions](evidence/v273-confidence-audit/confidence-comparison.csv).
- [Rejeu](../scripts/replay_v273_confidence.py),
  [calcul des métriques](../scripts/v273_confidence_metrics.py) et
  [vérification](../scripts/verify_v273_confidence_audit.py).
- [Workflow et dépendances](../.github/workflows/v273-confidence-audit.yml).

Le premier run `37797057773` a échoué à l'export final du JSON sur un entier
NumPy, après avoir produit les instantanés. La conversion explicite en entier
Python a permis le run complet référencé ici ; aucun poids ou protocole
numérique n'a été modifié.

Après téléchargement des deux ZIP du rejeu sous `confidence-results/` et
extraction dans `control7/` et `corrector8/`, écrire leur manifeste
`artifacts.json` à partir du champ `artifact_manifest` de la preuve publiée.
Placer les archives d'origine extraites dans `catalogue-results/control7/`
et `catalogue-results/corrector8/`. Depuis la racine du dépôt :

```bash
python -m scripts.verify_v273_confidence_audit \
  --artifacts ../confidence-results \
  --original ../catalogue-results \
  --output analysis/evidence/v273-confidence-audit/verification.json
```

Le vérificateur contrôle les sommes SHA, identités des poids, partitions et
labels, reconstruit les probabilités et la perte, reproduit les décisions
archivées, vérifie la multiplicité des événements et recalcule toutes les
comparaisons. Le résultat vérifié est `status: verified`.

**Aucune promotion.** Le résultat mesuré reste +38 net ; l'audit explique
où la confiance diverge du résultat et prépare une correction contrôlée.
