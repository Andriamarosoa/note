# NOTE — Audit transversal S745 → S748 (19 octobre 2026)

**Statut** : audit expérimental ; S708 reste signé et officiel. Aucun
pourcentage >80 % ni promotion de S747/S748 n'est établi ici.
Pas de changement du microphone LIVE, de S80/S397 ou de S675.

## Défauts vérifiés / constatés

| Priorité | Axe | Preuve et effet |
|---|---|---|
| P0 | Parité de référence | S747 a terminé FIT80/CAL32 mais a échoué dans `scripts/s746_instance_safe_fit_cal_loops.py` sur `AssertionError: S746 drifted signed S708 metric: exact_k` ; run 39202362375. Le contrôle interdit de publier ses résultats comme valides. |
| P1 | Qualité des offres | Artefact S747 du run 39202362375 : 6 217 instances proposées sur FIT80, seulement 410 événements complets jugés corrects (6,5948 %). Taux de *positifs parmi les candidats FIT*, non F1 de reconnaissance et non précision de la sortie sélectionnée. |
| P1 | Conflit des têtes | S747 stockait un délai réfractaire par tête et hauteur uniquement. Deux têtes pouvaient donc proposer la **même hauteur** à deux fragments voisins ; S748 introduit une déduplication au niveau de l'arbitre, pas dans les poids ou états individuels. |
| P1 | Reprise après relâchement | S747 pouvait représenter une queue spectrale d'une note S708 récemment relâchée comme nouvelle note ; S748 introduit une garde causale courte après NoteOff natif. |
| P1 | Offset et appariements | S746 a conservé les objets MIDI S708 mais a néanmoins perdu 3 correspondances d'Offset auparavant correctes sur son meilleur essai d'Active-K. L'invariance des événements *n'implique pas* l'invariance d'appariement F1 après ajout d'attaques concurrentes. |
| P1 | K5/K6 | Audit antérieur S742 : exactitude vraie K5 9,39 %, K6 6,40 %, sous-comptages 36 890 et surcomptages 6 940 sur CAL32. Les têtes de nouveaux NoteOn seules ne guérissent pas ce biais. |
| P2 | Sélection | 32 masques H0-ancrés existent, un masque choisi **par fragment** ; cela n'est pas encore une architecture totalement indépendante par caractéristique, avec son propre apprenant. |
| P2 | Domaines | FIT80 et CAL32 sont disjoints par composition GuitarSet ; aucune démonstration sur guitaristes/instruments réellement externes ni sur le microphone LIVE pour S747/S748. |
| P2 | Causalité | Les tests de préfixe temporel et d'événements natifs passent, mais l'inférence en boucle microphone avec les six spécialistes reste non validée. |
| P2 | Évaluation | L'agrégat seul masque les échecs des registres grave/médium/aigu, K0..K6, solos/accompagnement, notes longues et faux partiels d'octave/12e. Ajouter ces strates à l'audit indépendant. |

## Boucles exécutées et nouvelles

- **S745** : 12 essais réels, rejetés (Onset/Offset/notes complètes).
- **S746** : 12 essais réels, CI verte, mais 0 variante admissible :
  variante `depth5_risk55_tail20` = Onset 74,3394 %,
  Exact-K 62,3084 %, Active Exact-K 56,6134 %, Offset 77,2243 %,
  notes complètes 57,1138 %, Frame Pitch 76,6449 %. 3 Offsets perdus.
- **S747** : 80 FIT et 32 CAL effectivement traités, vérification signée
  `exact_k` en échec. Un second run instrumente les scores de référence
  par enregistrement et conserve l'écart avant refus.
- **S748** : deux nouvelles barrières **causales d'arbitrage**
  (inter-têtes et queue NoteOff) et 12 nouvelles variantes CAL32,
  strictement comparées à S747 et S708, sans transfert en LIVE.

## Cadence diagnostique non destructive

1. Corriger la parité signée **avant d'interpréter les nouveaux résultats**.
   Toute différence d'Exact-K ou de flux source est enregistrée par
   enregistrement et bloque la promotion.
2. Mesurer l'éligibilité d'événements (correctement reconnus / proposés),
   rejets, re-déclenchements, notes qui expirent sans Offset, et conflits
   d'appariements avec les événements S708 natifs.
3. Séparer les strates K0–K6 : rappel des vrais K≥2, sous/surcomptages
   et matrices de confusion sans mélanger tête K brute et comptage MIDI.
4. Mesurer Onset pitch±80 ms, Offset±50 ms après appariement, notes
   complètes, fréquence/trames, graves/aigus, solos/accords, notes longues.
5. Comparer S747/S748 sur les **mêmes** 32 enregistrements et les
   **mêmes** annotations. Ni amélioration d'une seule métrique,
   ni CI verte ne vaut promotion.
6. Si une vraie variante sûre apparaît, **figer paramètres avant HELD**
   puis vérifier sur nouveaux guitaristes, sons non vus et microphone causal.
   Ne pas entraîner/choisir sur HELD et ne pas ouvrir TEST.

Fichier d'audit calculable :
`scripts/s748_six_axis_failure_audit.py` (post-hoc seulement, jamais
dans le prédicteur), pour K0..6, confusions, Onset par registre,
Frame Pitch par registre, vrais/predits K>6, faux partiels, Offset
raison et delta six métriques.

Références :
- S746 : https://github.com/Andriamarosoa/note/actions/runs/39200842305
- S747 original : https://github.com/Andriamarosoa/note/actions/runs/39202362375
- S747 diagnostic : https://github.com/Andriamarosoa/note/actions/runs/39204464834

**Règle finale** : six métriques complètes ≥ référence *et* aucune
instance d'Offset ou Onset précédemment correcte perdue, puis
validation externe indépendante. Aucune distillation avant >80 %.
