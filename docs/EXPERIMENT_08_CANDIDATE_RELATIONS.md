# Expérience 08 — relations entre candidats onset/offset

État figé le 31 août 2026. L'audit est terminé. Il n'a modifié ni le modèle,
ni les seuils, ni le décodeur, ni le chemin live. Il explique la sur-prédiction
restante du diagnostic N=16 sans sélectionner encore de nouveau filtre.

## Question

Les candidats supplémentaires sont-ils principalement :

1. plusieurs détections successives autour d'une même frontière réelle ;
2. des réactions de plusieurs canaux au même voisinage ;
3. des détections isolées, éloignées de toute frontière annotée ;
4. ou un mélange de ces mécanismes ?

La sortie publique reste strictement :

```text
type, position
```

Le canal interne est observé uniquement pendant l'audit. Il ne devient ni une
corde publiée, ni une cible publique, ni une association par `eventId`.

## Protocole et correction avant exécution

Le protocole a été préenregistré au commit `cf03b32`. Le pré-audit des
annotations a été enregistré au commit `e09f8ad`, avant l'instrumentation. Une
revue indépendante a ensuite trouvé deux ambiguïtés méthodologiques avant la
passe complète :

- être proche d'une seule référence ne prouve pas que deux candidats sont le
  même événement acoustique ;
- une fenêtre de ±50 ms pouvait attribuer un même candidat à deux notes
  voisines.

L'amendement `1c22733`, lui aussi antérieur à la passe complète, impose donc :

- une classe répétition forte exigeant même type, même canal interne, candidats
  successifs et même unique frontière admissible ;
- une classe séparée de simple proximité ;
- un appariement un-à-un par type et canal pour mesurer la relation onset–offset
  d'une même note annotée.

## Pré-audit des références

Sur les 60 pistes et 9 541 notes de validation :

| Relation annotée | Minimum |
|---|---:|
| Deux onsets successifs du même canal | 3 421 échantillons = 77,57 ms |
| Deux offsets successifs du même canal | 2 844 échantillons = 64,49 ms |
| Deux onsets de canaux différents | 0 ou 1 échantillon possible |
| Deux offsets de canaux différents | 0 ou 1 échantillon possible |

Il existe 45 onsets et 54 offsets simultanés supplémentaires. Un délai global
serait donc invalide : il fusionnerait de vraies frontières inter-canaux. Toute
future consolidation temporelle devra préserver cette multiplicité.

## Intégrité

- checkpoint V7-e8 inchangé, SHA-256
  `5634ADD0E112A6889B65D5245AD051AD850A1FFFE66FEB8D9E5E74472BA114BF` ;
- source Exp07 inchangée, SHA-256
  `4E5249C4AB2E6C3B170AA450BE8E27F6DFCF95692DF0917623C268485E20AD88` ;
- joueurs `00` à `04`, joueur `05` non lu ;
- 60/60 pistes et 162 030 blocs reproduisent exactement Exp07 pour N=1 et
  N=16 ;
- le même objet de scores est remis aux deux décodeurs et à l'observateur ;
- la projection anonyme de l'observateur reproduit exactement les deux
  décodeurs sur chaque piste ;
- aucun score brut ni candidat individuel n'est écrit ;
- 158/158 tests réussis dans l'environnement TensorFlow réel ;
- durée murale : 2 727,49 s, soit 45,46 minutes ;
- résultat SHA-256 :
  `C117BA2C15B0FF9B5A907831C38CEDB71B4BAB67E7158FBA8DE88F743BBBE25B`.

## Définition simple des quatre classes de faux positifs

| Classe | Signification |
|---|---|
| Isolé | aucune frontière réelle du même type à ±50 ms |
| Répétition forte | candidat successif du même type et canal, relié à la même unique frontière |
| Proche d'une seule référence | proximité réelle, mais relation insuffisante pour parler de répétition |
| Ambigu | au moins deux frontières réelles admissibles à ±50 ms |

La proximité reste une relation statistique aux annotations, pas une preuve
d'identité acoustique.

## Résultat principal

### Faux onsets N=16

| Relation | Nombre | Part des 148 886 faux onsets |
|---|---:|---:|
| Isolés | 90 918 | **61,07 %** |
| Répétition forte | 25 865 | 17,37 % |
| Proches d'une seule référence | 9 328 | 6,27 % |
| Ambigus | 22 775 | 15,30 % |

### Faux offsets N=16

| Relation | Nombre | Part des 250 647 faux offsets |
|---|---:|---:|
| Isolés | 152 126 | **60,69 %** |
| Répétition forte | 52 616 | 20,99 % |
| Proches d'une seule référence | 5 389 | 2,15 % |
| Ambigus | 40 516 | 16,16 % |

Globalement, la majorité des faux candidats n'est donc pas une répétition
autour d'une vraie frontière : elle apparaît dans des régions sans frontière
annotée du même type à ±50 ms.

La conclusion n'est toutefois pas uniforme. La classe isolée dépasse 50 % dans
8/12 groupes famille–arrangement pour onset et 9/12 pour offset, ainsi que dans
32/60 et 38/60 pistes. La classe répétition forte ne dépasse 50 % dans aucun
des 12 groupes. Selon la règle préenregistrée et amendée, le mécanisme global
doit donc être déclaré **mixte**, avec une majorité isolée agrégée, et non une
cause universelle unique.

## Temporalité des répétitions

Même après N=16, les candidats restent très rapprochés sur un même canal :

| Mesure | Onset | Offset |
|---|---:|---:|
| Écart médian entre candidats successifs | 5,96 ms | 4,33 ms |
| Successeurs à 50 ms ou moins | 120 468/157 673 = 76,40 % | 227 706/259 272 = 87,83 % |
| Successeurs reliés à la même unique référence | 20 183 | 43 984 |
| Successeurs avec au moins un candidat isolé | 100 343 | 163 268 |

Il existe donc bien des candidats temporellement rapprochés, et beaucoup de
leurs paires successives comprennent au moins un candidat isolé. Une
consolidation de pics peut réduire le nombre de sorties, mais sa capacité à
corriger les activations isolées devra être mesurée séparément.

Le premier score au franchissement est aussi très proche du seuil `0,55` :

| Score d'entrée | Onset | Offset |
|---|---:|---:|
| Médiane | 0,55274 | 0,55126 |
| p90 | 0,56047 | 0,55474 |

Les scores sont donc très concentrés juste au-dessus du seuil au moment précis
du franchissement, surtout pour offset. Cette mesure ne prouve pas à elle seule
la cause des répétitions et ne décrit pas la hauteur maximale ultérieure du pic.

## Relation aux canaux internes

À ±50 ms :

- 42,47 % des onsets et 41,41 % des offsets sont proches d'au moins une
  référence du même type ;
- seulement 13,97 % des onsets et 13,95 % des offsets sont proches d'une
  référence du même canal interne ;
- 45 038 onsets et 71 292 offsets sont proches uniquement de références
  appartenant à d'autres canaux.

De nombreux candidats sont donc proches uniquement de références appartenant
à d'autres canaux. Cette relation de proximité ne doit pas être interprétée
comme une réaction acoustique causale, ni être transformée en sortie
corde/case.

La multiplicité exacte reste fortement surproduite :

| Type | Événements simultanés supplémentaires prédits | Référence |
|---|---:|---:|
| Onset | 10 521 | 45 |
| Offset | 5 844 | 54 |

## Relation onset–offset d'une même note annotée

Après appariement un-à-un par type et canal :

| Support trouvé pour la note | Notes | Part |
|---|---:|---:|
| Onset et offset | 7 270 | 76,20 % |
| Onset seulement | 1 609 | 16,86 % |
| Offset seulement | 411 | 4,31 % |
| Aucun des deux | 251 | 2,63 % |

Au total, 93,06 % des notes ont un onset proche sur le même canal et 80,51 %
ont un offset proche sur le même canal. Il existe donc un signal relationnel
utile pour de nombreuses notes, mais il est noyé dans un nombre beaucoup plus
grand de candidats supplémentaires. Cet audit ne construit aucune association
live entre onset et offset.

## Conclusion et arrêt

La sur-prédiction restante possède au moins deux composantes :

1. des rafales de candidats successifs, dont une partie est fortement reliée à
   la même frontière réelle ;
2. une composante plus grande de détections isolées, sans frontière réelle du
   même type à ±50 ms.

La correction ne peut donc pas se limiter à fusionner les répétitions d'une
vraie note. Il faudra tester séparément une consolidation causale des rafales,
puis mesurer ce qui reste des activations isolées avant de décider si un nouvel
entraînement de détection `type, position` est nécessaire.

Conformément au protocole, l'expérience s'arrête ici. N=16 reste diagnostique,
V7-e8 associé reste la baseline live officielle, et aucune nouvelle règle n'est
activée sans une nouvelle approbation.

## Artefacts

- protocole :
  `model/causal-boundaries-weight28-window512-v7-epoch08.candidate-relations-protocol.json` ;
- amendement méthodologique :
  `model/causal-boundaries-weight28-window512-v7-epoch08.candidate-relations-protocol-amendment-01.json` ;
- pré-audit :
  `model/causal-boundaries-weight28-window512-v7-epoch08.candidate-relations-preaudit.json` ;
- évaluateur : `scripts/audit_boundary_candidate_relations.py`, SHA-256
  `2C0E732CD3DB4534C00840D79A55C3970DBC025F1DC588DC7EDB390DC2758402` ;
- tests : `test/test_boundary_candidate_relations.py`, SHA-256
  `05237175A0A1F086E75040BAE5A4DD795BB16B918DC650AD5DC3E0C247C55CA9` ;
- résultat :
  `model/causal-boundaries-weight28-window512-v7-epoch08.candidate-relations.json`.
