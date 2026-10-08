# V27.3 — Calibration des réseaux figés, protocole préalable

Objectif : réduire la surestimation des décisions, puis mesurer ce que cette
correction révèle des propositions manquantes, des mauvais choix et des
changements d'un verdict initialement correct. Les résultats seront publiés
même si le gain net diminue ou si la calibration reste insuffisante.

## Population et indépendance

Utiliser exclusivement les sorties externes du rejeu figé `37797488453`,
vérifiées dans `analysis/evidence/v273-confidence-audit/verification.json`.
Le réseau de chaque fold a été entraîné sur les trois autres folds. Ses poids,
ses propositions et ses entrées restent inchangés. Les folds 0/1/2/4 sont déjà
exposés au développement ; fold3/player05 restent exclus.

Une calibration apprise sur les prédictions externes des autres folds serait
insuffisamment indépendante : les réseaux qui les ont produites ont vu le fold
évalué. Une calibration sur les prédictions d'entraînement du réseau serait
également inadaptée au défaut de généralisation observé.

On effectue donc une **calibration supervisée à l'intérieur du fold exclu du
réseau**, avec exclusion complète du morceau évalué. Un morceau est la partie
du nom située entre le numéro du musicien et le suffixe `comp/solo`, par
exemple `Funk2-119-G`. Tous ses musiciens et ses enregistrements comp/solo
appartiennent au même bloc. Les noms et les métadonnées, jamais les labels ni
les scores, déterminent les blocs.

Pour chaque morceau d'un fold :

1. Ajuster quatre paramètres de calibration sur les autres morceaux de ce
   même fold, que le réseau n'a donc pas vus pendant son apprentissage.
2. Appliquer cette calibration au morceau exclu, sans utiliser ses labels.
3. Recommencer pour chaque morceau, sans choisir la meilleure partition.

Il y a 5/4/5/5 morceaux dans les folds 0/1/2/4, soit 19 calibrateurs par bras.
Chaque événement est évalué une seule fois après calibration. Le protocole
est identique pour le contrôle à sept sélections et le catalogue S8, qui est
le bras principal. Les enregistrements d'un morceau ne sont jamais divisés
entre ajustement et évaluation d'un même calibrateur.

**Portée précise :** ce test mesure l'apport de labels de calibration provenant
d'autres morceaux du même fold, avec un réseau figé. Il ne démontre pas le
transfert d'un calibrateur appris uniquement sur les trois folds d'entraînement,
ni une validation sur des données totalement inédites. Les différentes
versions musicales éventuellement présentes dans les autres folds restent
celles du découpage d'origine. Aucun résultat ne sera présenté comme une
validation indépendante sur un nouveau domaine.

## Transformation fixée avant les résultats

Avec `z` le logit de justesse du verdict initial et `l_K` le logit déjà agrégé
des groupes proposant K :

```text
r' = sigmoid(a_r × z + b_r)
q' = softmax([a_q × l_K + b_q pour les K disponibles, 0 pour OTHER])
gain'(K) = (1 − r') × q'_K − r'
```

Les quatre paramètres sont initialisés à `(1, 0, 1, 0)`. Les pentes sont
bornées à `[0.05, 20]` et les intercepts à `[-10, 10]`. Leur positivité conserve
l'ordre des verdicts alternatifs d'un même événement. La calibration ajuste
la confiance et la décision de changer ; elle ne crée aucun nouveau verdict
et ne départage autrement les combinaisons proposant des K différents.

Minimiser la même perte jointe par événement que le réseau :
`BCE(r', B) + (1−B) × CE(q', cible)`, augmentée d'une pénalité fixe
`0.001 × [(a_r−1)² + b_r² + (a_q−1)² + b_q²]`. Cette petite pénalité stabilise
les quatre paramètres vers la transformation identité. Aucune recherche de
coefficient, de borne, de transformation ou de seuil sur les résultats évalués.
L'objectif est convexe dans ces paramètres ; une optimisation L-BFGS-B avec
gradient analytique doit réussir, sinon le calcul s'arrête pour examen.

La règle de décision reste le meilleur gain strictement positif, avec KEEP
en cas d'égalité à zéro. Les 127/255 combinaisons restent présentes ; aucun
veto sur une sélection individuelle. Le contrôle de cette étape conserve
les bonnes interactions déjà apprises.

## Mesures et audits à publier

- Avant/après sur les mêmes 7 493 événements : pertes, Brier, probabilités et
  fréquences observées, gain annoncé, corrections, régressions, neutres et net.
- Exact-K global, polyphonique et K0 à K6 sur les 59 309 événements natifs ;
  les événements non éligibles conservent leur verdict initial.
- Résultats par fold et par morceau, paramètres et convergence de tous les
  calibrateurs. Aucune sélection de la meilleure partition.
- Changements conservés, annulés et ajoutés ; corrections perdues et
  régressions évitées. Une baisse d'activité n'est pas une amélioration de
  discrimination démontrée.
- Trois familles de changements ratés : verdict initial correct ; verdict
  initial faux sans alternative correcte ; verdict initial faux avec bonne
  alternative disponible mais mauvais choix.
- Scores résiduels trop confiants : erreurs encore retenues, avec leurs
  identifiants, morceau, K initial/vrai/proposé et scores avant/après.
  Les seuils descriptifs `gain >= 0.1` et `gain >= 0.5` sont fixés ici ; ils ne
  deviennent pas des règles. Les intervalles de gain de l'audit précédent
  sont également conservés.
- Synergies conservées ou perdues où aucun membre du groupe correct ne
  réussissait seul, en réutilisant les propositions de groupes complètes.

Le script de vérification contrôlera les SHA des sources, l'exclusion des
morceaux et des événements, l'identité des sorties brutes, la reconstruction
des probabilités, le décodage et les comptes. Les gradients de l'objectif
seront contrôlés numériquement avant l'exécution. Les prédictions et les
paramètres seront exportés pour reproduction sans TensorFlow.

**Aucune promotion automatique.** Le succès de la calibration et le progrès
du gain net sont deux résultats distincts. La faiblesse des propositions ou
du classement peut rester entière après correction de la confiance.
