# S645 — Recherche d'attaques nouvelles par montée causale du score S642

## Pourquoi une deuxième boucle ?

[S644 a prouvé](README_V273_S644_WEAK_CANDIDATE_GATES.md)
que le seuil appris q≥0,965 ne reçoit aucune observation qui
respecte simultanément le garde acoustique original
onset S614≥0,13 et sustain≥0,26. Le maximum q après ce filtre
était 0,959 sur TRAIN et 0,962 sur DEV.

Relâcher le seuil q à 0,50–0,85 augmente le nombre de vrais
positifs mais dégrade massivement la précision et le F1.
**Le score q élevé sans notion d'attaque nouvelle ne suffit pas.**

## Architecture expérimentale

Conserver strictement le modèle S614 (incluant sa logique de
sélection Exact-K native), la tête d'attaque S642 déjà entraînée,
S632 et S80, sans modifier aucun checkpoint.

Pour chaque hauteur inactive et à chaque trame :
- observation q_t provenant des caractéristiques S614 **causales** ;
- mémoire des **six scores précédents uniquement** ;
- tester q_t supérieur à un seuil élevé (0,98, 0,99 ou 0,995)
  et q_t − min(q_{t−6},...,q_{t−1}) ≥0,10/0,20 ;
- confirmer une activité physique minimale d'attaque et maintien
  (plus faible qu'en S642, mais non nulle), parfois un signal
  spectral dans les colonnes du frontend native ;
- présenter au plus **une seule** nouvelle hauteur au
  sélecteur causal S614 par trame ;
- distinguer tentative, émission MIDI effective, vrai positif,
  faux positif, faux négatif et régression sur chaque source.

Les garde-fous sont expérimentalement prédéfinis dans
`scripts/s645_causal_score_rise_loop.py`.

## Prévention des fausses promotions

- 35 enregistrements TRAIN (17 URMP, 18 GuitarSet) séparés
  par composition des 17 DEV ;
- choisir l'éventuelle politique positive sur TRAIN **uniquement**,
  exigence d'amélioration F1 dans les deux corpus et au plus
  deux sources en régression par corpus ;
- après verrouillage, mesurer DEV avec les mêmes politiques
  pour comprendre les défauts, sans sélection a posteriori ;
- révision TRAIN/DEV déjà exposée, donc pas de nouveau test
  indépendant ;
- aucun remplacement S614 sans amélioration vérifiée,
  absence de régressions, source réellement inédite et
  vérification stricte du comportement LIVE matériel.

[Workflow GitHub S645](https://github.com/Andriamarosoa/note/actions/workflows/v273-s645-learned-rise-rescue.yml).

## Résultats réels S645 et S646 — verdict figé

[S645 TRAIN/DEV : workflow réussi](https://github.com/Andriamarosoa/note/actions/runs/38818392769),
[rapport JSON complet](https://github.com/Andriamarosoa/note/actions/runs/38818392769/artifacts/12012102143).

**AUCUNE des cinq variantes de montée du score n'a émis de
nouveau NoteOn** sur les 35 TRAIN ou 17 DEV.
Aucune modification des TP, FP, FN, F1 originaux
sur GuitarSet ou URMP :

| Corpus DEV, métrique historique ±80ms | S614 | S645 toutes variantes |
|---|---:|---:|
| GuitarSet TP/FP/FN | 998/585/582 | 998/585/582 |
| GuitarSet F1 | **63,10 %** | 63,10 % |
| URMP TP/FP/FN | 243/149/256 | 243/149/256 |
| URMP F1 | **54,55 %** | 54,55 % |

La règle TRAIN a verrouillé **S614 original**.
Aucune promotion.

[S646 : audit indépendant des compteurs figés S645 réussi](https://github.com/Andriamarosoa/note/actions/runs/38818968835),
[artefact de preuve S646](https://github.com/Andriamarosoa/note/actions/runs/38818968835/artifacts/12011173578).

Exemple de variante q≥0,99 avec augmentation q_t − minimum
des six précédentes ≥0,10 :

| Corpus | Montées q détectées | Intersection avec garde acoustique et hauteur inactive | NoteOn supplémentaires |
|---|---:|---:|---:|
| TRAIN GuitarSet | **17 684** | **0** | **0** |
| TRAIN URMP | **35 483** | **0** | **0** |
| DEV GuitarSet | **17 204** | **0** | **0** |
| DEV URMP | **7 390** | **0** | **0** |

Même en diminuant onset S614 minimal à 0,02
et sustain minimal à 0,05, aucun score q atteignant
0,99 et montant rapidement ne coïncide avec le
garde et l'état inactif. Les autres quatre
variantes préfixées présentent le même problème,
y compris le garde spectral.

**Défaut de conception confirmé, plus profond qu'un seuil
mal fixé :** les activations élevées q de la tête
d'attaque S642 ne correspondent pas aux observations
que le réseau principal S614 et l'état du sélecteur
considèrent admissibles pour une nouvelle note.

Cet audit ne peut pas encore répartir avec certitude
l'échec entre (a) hauteur déjà active, (b) p_onset très
faible, (c) p_sustain très faible et (d) hypothèse
acoustique incorrecte. Pour cela, la prochaine étude
doit comptabiliser **les intersections conditionnelles
pour chacun des quatre facteurs séparément**.

### Implication de conception

Éviter une nouvelle boucle de seuils q ou une simple
copie du décodeur MIDI : entraîner une tête de
*naissance de candidate* directement conditionnée par
l'état courant S614 et par la trajectoire de
nouveauté acoustique. La supervision doit utiliser
des candidats réellement admissibles, ainsi que des
hard négatifs de hauteur active, harmoniques et
résonances, et non l'ensemble des trames d'attaque
du corpus indépendamment de la sélection.

Vérifier par audit indépendant la conversion
candidat→NoteOn, le coût des faux positifs et les
régressions des accords polyphoniques. Conserver la
référence officielle **S614 69,03 % F1 TEST historique**
tant qu'un nouveau score TEST indépendant n'est pas
reproduit.
