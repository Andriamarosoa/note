# S653–S654 — Boucles Tone vers un objectif ambitieux de 90 %

## Score cible et métriques strictement distinctes

Objectif de recherche : atteindre ou dépasser 90 % sur une mesure
**NoteOn F1 GuitarSet indépendante**, et mesurer séparément le
diagnostic de classification Exact-K (K0-K6).

Nous ne confondons pas :
- **NOTE S614** : 998 TP, 585 FP, 582 FN, F1 **63,10465 %**
  sur 12 GuitarSet DEV (1580 vraies attaques) ;
- **S641** : 1009 TP, 594 FP, 571 FN, F1 **63,39931 %**,
  meilleure sélection expérimentale choisie sur TRAIN ;
- **S652** : moyenne de 3 vues Tone, 974 TP, 509 FP, 606 FN,
  F1 **63,59778 %** DEV exploratoire mais pas retenue sur TRAIN ;
- **Exact-K diagnostic S652** : original 668/1576 = 42,39% ;
  filtres 2800Hz 674/1576 = 42,77% ; 1100Hz 671/1576 = 42,58%.

La cible à 90% est très éloignée de nos bases : même si les FP de
S614 étaient tous éliminés, il faudrait au moins 1293 TP parmi 1580
annotations pour obtenir 90% NoteOn F1. Cela nécessite **au moins 295
attaques correctement récupérées** par rapport aux 998 TP actuels
même dans le scénario improbable FP=0.

## Constat quantitatif pré-S653 issu de l'artefact S652

Un choix **oracle** de la meilleure des 13 politiques S652 pour
chacun des 12 enregistrements DEV, fait a posteriori avec les
annotations (donc *interdit en production*), obtient seulement :
- TP 996, FP 514, FN 584, **F1 64,466%**.

Ce score est une borne empirique sur un **sélecteur choisissant une
politique fixe par enregistrement** dans la famille S652.
Ce n'est **pas** une limite supérieure à NOTE, ni à une tête
variant par note, ni à une méthode apprenant de nouvelles
fondamentales. Cela démontre qu'on ne peut atteindre 90% en
changeant seulement un « Tone global » par morceau.

## Boucle S653 : les vues Tone proposent, une tête décide

- Même checkpoint natif, 2 filtres causal Butterworth `sosfilt`
  2800/1100Hz, 256 échantillons par hop.
- **Aucune transposition**, anticipation, normalisation future ni
  antidatage NoteOn.
- Une attaque Tone que NOTE natif n'a pas déjà confirmée devient
  un **candidat** ; le signal original reste chemin principal.
- Chaque proposition réellement émise par la chaîne ungated est
  étiquetée sur TRAIN comme *correction inédite vs faux NoteOn*.
- Les entrées de sélection : probas on/off/sustain des 3 vues,
  différentiels, compétition à l'octave, registre, posterior
  original Exact-K et les K des vues pour **contexte seulement**,
  et évolutions 4 hops passés.
- Quatre têtes entraînées séparément, validation GroupKFold par
  composition sur 9 compositions TRAIN et abstention autorisée.
- Seuls les choix désignés avant DEV peuvent compter comme
  résultats prévalidés. Toutes les alternatives seront présentées
  comme exploratoires, pas de sélection par maximum DEV.
- Sortie : score réel NoteOn TP/FP/FN/F1 par morceaux, polyphonie,
  registres graves E2/E3, et gains/pertes exacts sur les attaques.
- Aucun changement du modèle LIVE/Exact-K officiel.

## Boucle S654 : spécialisation de la tête Exact-K

- Fusion de probas `K0...K6` sur le même audio original, Tone
  doux et Tone fort ; quatre têtes de fusion simples/non linéaires.
- Même annotations et même procédure diagnostic S652 : lecture
  de la prédiction 30ms après l'attaque **annotée** ; la vérité
  K est le nombre de hauteurs actives à cet instant.
- Le timestamp annoté n'est **pas** un générateur d'attaque LIVE.
  Ce test mesure les possibilités de la tête K mais ne constitue
  pas un système autonome de reconnaissance.
- Composition GroupKFold TRAIN ; exigence de +3 points K accuracy
  et amélioration d'au moins 3/5 plis pour envisager un candidat ;
  DEV examiné seulement après la sélection sur TRAIN.
- Signal original K et sortie NOTE officiels inchangés.

## Pistes scientifiques complémentaires

L'estimation multi-F0 moderne utilise la **polyphonie et le timbre**
pour conditionner les notes concurrentes (2025,
`Multi-pitch estimation with polyphony per instrument information`,
DOI 10.1186/s13636-025-00398-2). Un travail ICASSP 2026 explore la
reconstruction et la nouveauté temporelle (DOI
10.1109/ICASSP55912.2026.11462294). Ces approches motivent des
**têtes conjointes H0/H1** comparant un accord avec ou sans nouvelle
fondamentale, sans prétendre reproduire les scores d'autres travaux.

## Règles de validation et de promotion

Ne jamais promouvoir sur :
- les mêmes compositions GuitarSet déjà utilisées pour choisir
  les seuils ou les caractéristiques ;
- des variantes choisies après lecture du jeu DEV ;
- un simple score Exact-K au timestamp annoté présenté comme LIVE ;
- un F1 sur des accords synthétiques présenté comme F1 des vrais
  fichiers GuitarSet polyphoniques ;
- des simulations de Tone présentées comme mesure physique de
  Stratocaster.

90 % ne sera revendiqué que si atteint sur le protocole,
avec comparatifs S614/S641, Exact-K K1/2/3/4/5/6,
résultats par morceau/registre, FP et latence, puis
confirmation indépendante sur un autre jeu réellement inconnu.
