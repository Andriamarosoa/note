# NOTE S632 — Apprentissage réel du coût de l'attente après une attaque

Statut : **entraîné, validé sur DEV, non promu**.

Sources : [S632 Q-cost training (SUCCESS)](https://github.com/Andriamarosoa/note/actions/runs/38803974784),
[S632 runtime strict LIVE parity (SUCCESS)](https://github.com/Andriamarosoa/note/actions/runs/38804294952).

## Hypothèse

S631 prédit trois actions (WAIT/EMIT/REJECT) selon des étiquettes approximatives ;
S632 apprend maintenant **le coût futur estimé de ces actions à partir de
l'observation causale courante**.

- \(Q_{\mathrm{emit}}\) : prix d'émettre maintenant une fausse hauteur
  ou de dépasser la tolérance temporelle ;
- \(Q_{\mathrm{reject}}\) : prix de manquer une vraie attaque ;
- \(Q_{\mathrm{wait}}\) : coût du temps supplémentaire + opportunité
  estimée d'une meilleure décision après observation.

Coûts de la supervision oracle sur TRAIN :
FP = 1,20, FN = 1,35, attente = 0,020 par trame, retard
supplémentaire = 0,10 par 80 ms. Une programmation dynamique
rétrograde donne les cibles coût-à-venir sur **TRAIN uniquement**.
L'inférence est causale : l'oracle et la vérité MIDI n'interviennent
jamais dans les trois prédictions courantes. Ceci est une approximation
**supervisée** de Q, et non une preuve d'optimisation globale par RL.

S632 conserve les propositions NOTE générées par S614, les
caractéristiques harmoniques natives et les décisions par hauteur.
L'observation d'un événement candidat peut durer au plus
28 trames × 256 / 44 100 ≈ 162,54 ms.
Aucune détection n'est antidatée ; NoteOff est conservé.

## Entraînement et vérifications

- 35 enregistrements TRAIN (17 URMP, 18 GuitarSet), sans compositions
  partagées avec les 17 enregistrements DEV ;
- 135 373 états de candidats échantillonnés ;
- modèles HistGradientBoostingRegressor distincts pour les 3 Q :
  - EMIT SHA256 `207329be553ad179b461b725749e69244c080ac562b71364df6dc58066620218` ;
  - REJECT SHA256 `e57c818431e2544eba3d0b22d59d6df621cbac7637c7874df98424d8fb0a6a5b` ;
  - WAIT SHA256 `b54f631eaaeb0c31bc9f072bbfe2f6b76b9dcdd795f06401389c76a61824a4c6`.
- Les tests de parité coût par lots vs prédiction LIVE bloc par bloc,
  mutation du futur, respect des empreintes, original S614 sans
  nouvelle décision, et fin de NoteOff ont réussi.

Le modèle original S614 SHA256
`7ccf046e3a1870eaca3a108e6f9565298aeeadf43732a2f83559542c03f5f17f`
et S80 historique restent strictement inchangés.

## Résultats sur DEV connus historiquement (NoteOn ±80 ms)

| Variante | GuitarSet TP/FP/FN, F1 | URMP TP/FP/FN, F1 |
|---|---|---|
| S614 | 998 / 585 / 582, **63,1046 %** | 243 / 149 / 256, **54,5455 %** |
| S632 assistance, coût attente +0,10 | **1001 / 585 / 579, 63,2344 %** | **244 / 149 / 255, 54,7085 %** |
| S632 assistance, autres 5 grilles | 998 / 585 / 582, 63,1046 % | 243 / 149 / 256, 54,5455 % |
| S632 remplacement, coût émettre max 0,65 | 467 / 115 / 1113, 43,2007 % | 109 / 12 / 390, 35,1613 % |

L'aide à la décision avec coût d'attente augmenté (plutôt que
prolonger indéfiniment la décision) est **seule positive sur les
deux corpus DEV** :

- GuitarSet : +3 TP net, 0 FP net, +0,1297 point F1 ;
- URMP : +1 TP net, 0 FP net, +0,1631 point F1 ;
- 2 extraits GuitarSet progressent, 1 régresse ;
  1 composition URMP progresse, aucune ne régresse.

Ce n'est pas encore une validation indépendante :
les jeux DEV ont déjà été examinés et réutilisés lors de
S613/S614 et des boucles suivantes. Aucun score sur le groupe
TEST S614 déjà publié n'a été modifié ni réévalué pour sélectionner
S632.

## Latences ajoutées des émissions assistées

Variant `S632_assist_wait_p010` :
- GuitarSet : médiane **5,805 ms**, p95 **17,415 ms**,
  maximum **63,855 ms** depuis la *création du candidat* ;
- URMP : médiane **5,805 ms**, p95 **26,413 ms**,
  maximum **46,440 ms** depuis la *création du candidat*.

Ces latences ne sont PAS l'intervalle attaque physique→sortie MIDI
et ne mesurent pas le calcul temps réel de l'ensemble de la chaîne.
Les émissions assistées déclenchées par la tête ne se traduisent
pas nécessairement en notes MIDI **supplémentaires** : nombre
important de décisions redondantes avec S614, à auditer.

## Verdict

**Conserver S632 comme expérience positive limitée, sans le promouvoir.**

- La supervision des risques réduit les régressions du remplacement
  complet de S631 lorsqu'elle sert d'**assistant** à S614.
- Le gain F1 reste minime ; aucune réussite vers l'objectif 80 % LIVE
  n'est revendiquée.
- La politique de remplacement est nettement négative.
- Le test indispensable est une source nouvelle hors des corpus
  réexaminés, avec modèle, coût et sélecteur gelés ; reproduire aussi
  chaque correction et chaque régression indépendamment.
- Le contrôle logiciel de causalité est passé ; il reste à mesurer la
  performance machine complète (temps p95 par bloc et micro→MIDI).

## Reproduction

```bash
export PYTHONPATH=.:src
python -m unittest -v test.test_s632_risk_policy test.test_s632_streaming_runtime test.test_strict_live
python -m scripts.s632_train_decision_cost --out /tmp/S632
```

Le code, les trois modèles originaux et le rapport JSON complet sont
archivés dans l'artifact de l'exécution S632. La référence officielle
reste **S614 à 69,03 % F1 sur ses 22 enregistrements TEST historiques**.
