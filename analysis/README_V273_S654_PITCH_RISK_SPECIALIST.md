# S654 — Une véritable sélection spécialisée de risque dans NOTE S614

## Motivation : conclusions vérifiées des boucles précédentes

[S653 : audit événementiel des vraies attaques](https://github.com/Andriamarosoa/note/actions/runs/38823590757)
confirme les échecs de S648 sur GuitarSet DEV :
- stratégie naissance 0,85/0,15 : **8 notes nouvelles correctes, 0 note correcte historique perdue, 114 FP nets supplémentaires** ;
- stratégie 0,90/0,15 : **2 notes correctes nouvelles, 0 perdue, 15 FP nets** ;
- les 8 corrections au premier seuil : 3 dans des groupes K1, 1 K2, 4 K3 ;
- la tête S648 ne déclenche aucune correction URMP aux seuils 0,85/0,90.

Il ne faut donc plus simplement augmenter ou diminuer le seuil
de la même tête. Il faut **une sélection spécialiste du risque d'une
nouvelle NoteOn**, qui reçoive explicitement le contexte Exact-K,
l'occupation active et les concurrents harmoniques.

## Architecture construite, sans écraser les têtes d'origine

Original : PCM LIVE → frontal original 49×126 → MLP K0..K6 +
3 sorties par hauteur (onset, sustain, release) hybrides MLP/arbres →
sélecteur causal S614 (armement, confirmation, réfractaire, NoteOff).

Branche expérimentale uniquement :
1. S648 propose un événement de naissance (seuil 0,70 / hausse 0,10).
2. Nouvelle tête d'arbitrage `S654` reçoit les descripteurs acoustiques
   du pitch à cet instant, l'état de S614, les scores K, la marge
   par rapport aux autres hauteurs, les scores des voisines aux
   décalages harmoniques ±12/19/24 demi-tons et l'historique de `q`.
3. Action **ACCEPT** si risque appris favorable, sinon **ABSTAIN**.
4. Après acceptation, le sélecteur S614 original applique ses
   protections temporelles, capacité et réfractaire : pas de NoteOn
   antidatée, pas d'audio futur.
5. NoteOff, S614, S632, S80, poids S648, et référence officielle
   restent entièrement intacts.

Le code n'est pas une simple règle de seuil q : il entraîne un second
classifieur supervisé sur les **NoteOn supplémentaires et réellement
appariés** par rapport aux références reconnues par S614, ainsi que
sur les propositions parasites. C'est un prototype d'arbitrage
événementiel inspiré des contrôleurs de risque S70–S72.

## Jeu d'entraînement anti-fuite

- Les 24 sources originales ayant entraîné S648 sont **exclues**
  du nouvel apprentissage du risque (leurs scores ne sont pas OOF).
- S654 utilise les **11 CALIBRATION S648** non vues par la tête S648,
  réparties à nouveau **par composition**.
- 7 enregistrements servent à entraîner la tête de risque.
- 4 sources de compositions disjointes servent à verrouiller un
  seuil parmi 0,35 / 0,55 / 0,75 / 0,90, déclarés avant leurs scores.
- Aucune annotation des 17 DEV n'intervient dans ce verrouillage.
- Le score des 17 DEV historiques est **exploratoire** (ces
  enregistrements ont déjà été vus dans les autres expériences).
- Toujours aucun accès aux TEST originaux S614, aucun test externe
  authentiquement inédit et aucune promotion automatique.

## Audit **dans** la boucle

Chaque option S654 est évaluée à nouveau sur le PCM original avec
l'appariement officiel S614 via `audit_event_round` S652.
Le rapport est un registre immuable par source :
- vrais NoteOn retrouvés vs références précédemment ratées ;
- notes historiquement correctes devenues erronées ;
- FP additionnels, origine temporelle et hauteur ;
- ventilation localement par K1/K2/K3/K4+, hauteur et
  cause candidate (harmonique, mauvaise hauteur, absence d'attaque).
  Ce K local ne constitue pas une mesure S80 d'Exact-K.

La politique ne serait retenue à titre expérimental sur la
sous-calibration TRAIN que si :
1. F1 historique supérieur à S614 dans GuitarSet **et** URMP ;
2. au plus **3 FP** supplémentaires par corpus ;
3. au plus **1 source** régressée par corpus.

En cas d'échec, le verrou reste sur **S614 original**.

## Fichiers exécutables

- `src/causal_note/s654_pitch_risk_selector.py`
- `scripts/s654_train_risk_on_unseen_birth_events.py`
- `test/test_s654_pitch_risk_selector.py`
- [GitHub Actions S654](https://github.com/Andriamarosoa/note/actions/workflows/v273-s654-learned-event-specialist.yml)

## Limitations

Le réseau d'Exact-K historique S80 compte lui-même de nombreuses
politiques spécialisées ; S654 n'importe **pas** ses 193 politiques
ni ses poids dans le réseau de notes. Il en reprend la structure
d'arbitrage conditionnel, désormais pour une décision événementielle
nouvelle NoteOn, ce qui évite de supprimer les trois têtes de notes
S614 et de casser la causalité du LIVE.

Aucune mesure de latence physique MIDI Windows n'a encore été
réalisée sur les nouveaux checkpoints. Ne pas confondre ce
prototype avec une nouvelle architecture officiellement promue.
