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


## S647 — Isolation de la porte exacte (confirmée par les données)

[Exécution S647 réussie](https://github.com/Andriamarosoa/note/actions/runs/38819248595)
et [rapport complet avec détails par source](https://github.com/Andriamarosoa/note/actions/runs/38819248595/artifacts/12012292801).

Audit sans changement de modèle : la sortie S614 a été
rejouée pour calculer l'état actif de chaque hauteur avant
chaque trame. Le modèle S642 a prédit q sur les mêmes
caractéristiques natives. La montée vérifiée est
q≥0,99 avec q − min(q des six précédentes) ≥0,10.

| Mesure | GuitarSet TRAIN | URMP TRAIN | GuitarSet DEV | URMP DEV |
|---|---:|---:|---:|---:|
| Montées fortes q | 17 684 | 35 483 | 17 204 | 7 390 |
| Montées q dans les fenêtres de vraie attaque (40ms) | **30** | **31** | **24** | **1** |
| Montées q sur hauteur déjà active | 0 | 0 | 0 | 0 |
| Montées avec onset S614 ≥0,02 | 24 | 0 | 20 | 0 |
| Montées avec sustain S614 ≥0,05 | 21 | 0 | 19 | 0 |
| Montées satisfaisant onset et sustain minimums | 21 | 0 | 19 | 0 |

Sur DEV GuitarSet, seules 24/17 204 (0,14 %)
des montées q se situent dans une fenêtre annotée de
vraie attaque. Sur DEV URMP, seulement 1/7 390
(0,014 %). Les occurrences sont des observations
trame-hauteur, **pas des événements MIDI uniques**,
et ces chiffres ne sont pas une précision de
classification à seuil optimisé.

Conclusions plus fines :
1. La porte `~active` **n'est pas le problème** :
   100 % des fortes montées de q sont sur des hauteurs
   qui ne sont pas actives dans le sélecteur natif.
2. La tête prédit une forte montée q essentiellement
   lorsque les sorties du réseau S614 pour attaque
   et maintien sont **toutes deux proches de zéro**.
3. Même la nouvelle garde physique très douce
   onset≥0,02 / sustain≥0,05 n'aurait que 19
   observations admissibles sur GuitarSet DEV et
   aucune sur URMP DEV.
4. Plus important, presque toutes les montées q ont
   lieu en dehors des attaques annotées : retirer
   entièrement les portes fabriquerait beaucoup
   de faux candidats. Une montée q forte **n'est
   pas actuellement un détecteur fiable de naissance
   de note**.

### Défaut de conception et suite

Le problème ne peut pas être corrigé durablement
par un seuil q fixe ou une simple attente après
l'attaque. La supervision de S642 a entraîné la
tête à distinguer des trames positives étiquetées,
mais la politique l'utilise comme un nouveau
déclencheur événementiel sans calibration
sur les véritables naissances et réattaques du
sélecteur original. La correspondance sémantique
entre `q` et la décision NoteOn est insuffisante.

Prochaine architecture à développer : entraîner une
nouvelle tête **événementielle** plutôt que
réutiliser ce q : cible de nouvelle attaque et
durée d'état passée, hard négatifs des notes
soutenues/résonances et accords, entrée de
l'état actif du S614 ; calibrer sur TRAIN
par composition ; proposer plusieurs classes
de candidats mesurées avec vraie précision
NoteOn et délai causal. Aucun q de l'actuelle
S642 n'est à promouvoir.

Les scores et checkpoints officiels restent :
S614 historique 69,03 % F1 sur test ±80ms ;
S632 et S642 expérimentaux, S80 inchangé.
