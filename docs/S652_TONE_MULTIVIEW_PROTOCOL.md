# S652 — Multi-view Tone de guitare : hypothèse et protocole

## Question

Le contrôle Tone d'une Stratocaster **atténue des fréquences
aiguës** ; il ne produit pas normalement de transposition +2 ni
un écho. Notre hypothèse : un filtrage fréquentiel peut rendre plus
facile la distinction d'une fondamentale nouvelle et des harmoniques
partagées, mais peut aussi **masquer de vraies notes** et provoquer des
**faux NoteOn** si les têtes NOTE changent de confiance.

Nous testons **une approximation numérique**, et non une mesure
du vrai circuit passif Fender :
- signal original GuitarSet (signal témoin) ;
- filtre causal Butterworth 1er ordre, coupure 2800 Hz ;
- filtre causal Butterworth 1er ordre, coupure 1100 Hz.

Ces deux filtres sont des *proxies de réponse fréquentielle*, pas des
équivalents reconnus d'une position du potentiomètre Tone. Sans
relevé de la guitare, de ses micros, du potentiomètre et de la charge,
on ne peut pas convertir l'un des filtres en « Tone 3 » ou « Tone 7 ».
**Aucune transposition ni délai n'est appliqué**.

## Méthode : vrais audios + architecture NOTE originale

Les trois signaux ont exactement le même horodatage PCM 44,1 kHz.
Le filtrage `scipy.signal.sosfilt` est **strictement avant vers
l'avant** : aucun `filtfilt` bidirectionnel, anticipation ou
décalage artificiel des attaques. La phase d'un filtre causal peut
néanmoins provoquer du **retard de groupe dépendant de la fréquence**,
qu'il faudra mesurer sur un vrai microphone.

Toutes les vues utilisent **la même chaîne NOTE gelée** :
`CausalPitchFeatures` → `NativeLiveHeads` → copies indépendantes
de `CausalSelection`. Chaque politique reçoit toujours le
**posterior Exact-K du signal original**, identique au baseline,
et conserve les règles de notes actives/onset/offset natives.
Les K prédits sur les signaux filtrés sont uniquement diagnostiqués
et ne commandent jamais la sélection.

Les **13 politiques fixées avant validation DEV** :
1. original S614 ;
2. Tone simulé doux 2800 ;
3. Tone simulé fort 1100 ;
4. moyenne original+doux ;
5. moyenne des trois ;
6. vote de deux vues sur trois ;
7. original confirmé par filtre doux ;
8. original confirmé par filtre fort ;
9–10. original ou secours doux, seuils 0,70 / 0,85 ;
11–12. original ou secours fort, seuils 0,70 / 0,85 ;
13. original ou consensus de deux vues filtrées.

## Partitions et décisions

- 36 enregistrements GuitarSet TRAIN, **neuf compositions** :
  cinq compositions pour choisir une politique fixée, quatre
  compositions entières comme contrôle interne sans réajustement ;
- 12 enregistrements GuitarSet DEV (historiquement exposés) pour
  comparer les politiques figées et vérifier les erreurs ;
- aucun enregistrement GuitarSet TEST ni IDMT/URMP TEST ouvert ;
- sélection obligatoirement basée sur les scores TRAIN +
  contrôle interne, pas sur le meilleur résultat DEV.

Un candidat doit battre S614 d'au moins **+0,4 point F1**
sur TRAIN calibration, puis +0,2 point F1 sur les compositions
TRAIN tenues à l'écart, et être positif sur au moins 3 des 4
compositions. Faute de preuves, la politique retombe sur
**le signal original S614**.

Le rapport comporte :
- TP, FP, FN et F1 NoteOn ±80ms de toutes les variantes ;
- corrections et régressions par enregistrement GuitarSet ;
- détection E2 (MIDI40), E3 (MIDI52), K2/K3+,
  et notes réellement superposées selon annotations ;
- Exact-K original constant pour tous les sélecteurs et prédictions
  K des filtres séparées pour analyse ;
- comparaison contextuelle avec S641 à F1 63,3993 % DEV, qui reste
  expérimental et non promu.

### Limites

Le corpus n'est **pas une campagne dédiée de micros Stratocaster
avec potentiomètre Tone documenté**. On ne peut pas revendiquer un
résultat propre au matériel Stratocaster. Les rares notes E2 isolées
dans les annotations TRAIN empêchent une vraie mesure équilibrée
« E2 seul » contre « E2+E3 » sans enregistrement complémentaire.

Un filtrage passif linéaire ne *crée* pas des hauteurs dans
le signal idéal, mais les réseaux NOTE peuvent fabriquer des
fausses notes parce que leurs probabilités changent.
Un gain local de F1 sur DEV ne suffit pas à promouvoir
un nouveau sélecteur LIVE.

## Code

- `scripts/s652_tone_filter_multiview.py`
- `test/test_s652_tone_filter_multiview.py`
- `.github/workflows/v273-s652-tone-multiview.yml`

Toutes les valeurs numériques d'évaluation sont dans l'artefact
`S652_GUITARSET_TONE_FILTER_MULTIVIEW_REAL/report.json`.
