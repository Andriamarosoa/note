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

## Résultats

À renseigner uniquement avec les nombres issus de la CI
après réussite complète de l'exécution.
