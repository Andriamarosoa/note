# NOTE — Détection des notes polyphoniques, Exact-K et Live
## README d'état du projet : historique S0–S584 et mises à jour S585–S609

> **Mise à jour S606–S609 — le transfert de S601 est maintenant mesuré : GuitarSet microphone 80,06 %, IDMT Dataset3 82,11 %, MAESTRO 76,26 % de F1 NoteOn ±80 ms**, contre respectivement 76,11 %, 72,15 % et 70,70 % pour Basic Pitch par défaut. **77 enregistrements, 10 887 vraies notes MIDI40–88, même modèle et mêmes seuils figés avant les annotations. Les 80 % ne sont pas validés de façon générale : accompagnement GuitarSet 76,08 %, Rock 74,26 % et piano 76,26 %.** Le Rock et deux extraits piano régressent face à BP. IDMT Dataset3 ne couvre qu'un seul instrument, pas les sept guitares de la collection. Audit : 693 vérifications d'appariement concordantes et toutes les 784 vraies notes perdues face à BP conservées. **Aucune promotion S80/Exact-K ni intégration Live.** [Résultats, limites et audit complet](analysis/README_V273_S606_S609_TRANSFER_RESULTS.md) · [CI de transfert réussie](https://github.com/Andriamarosoa/note/actions/runs/38688389092).

> **Mise à jour S598–S605 — objectif franchi sur le test réservé : 84,91 % de F1 NoteOn ±80 ms**, contre 73,51 % pour Basic Pitch par défaut sur les mêmes **cinq œuvres URMP, 120 s et 626 notes MIDI40–88**. Candidat S601 appris sur 37 autres enregistrements, avec exclusion des titres test et gel avant leurs WAV/annotations. **TP 484, FP 30, FN 142 ; précision 94,16 %, rappel 77,32 %.** Audit : 38 notes récupérées, 55 perdues, 206 FP de moins. Les cinq morceaux progressent en F1, mais Miserere reste à 75 %. **Résultat hors ligne, sans promotion S80/Exact-K ni intégration Live.** [Rapport et toutes les boucles](analysis/README_V273_S598_S605_RESERVED_80_RESULTS.md) · [Première CI réservée réussie](https://github.com/Andriamarosoa/note/actions/runs/38685400020).

> **Mise à jour S592–S597 — transfert URMP :** sur **cinq ensembles, 120 s et 1 119 vraies notes MIDI40–88**, S585 strict atteint **64,78 % de F1 NoteOn ±80 ms**, contre **62,23 % pour S578** sur les mêmes extraits. Gain : 35 notes récupérées, 15 perdues, 31 FP de moins. **Basic Pitch direct fait mieux : 73,54 %.** L’audit trouve 180 notes correctes de BP absentes de S585, dont 176 sans raccordement protégé à V88 ; S585 récupère 40 autres notes. Le gain de la tête ne suffit donc pas à valider la chaîne complète. Audio préparé 16 kHz, modèles figés avant annotations, aucune promotion ni modification du Live. **Objectif 80 % non atteint.** [Résultats et diagnostic S592–S597](analysis/README_V273_S592_S597_URMP_RESULTS.md) · [CI réussie](https://github.com/Andriamarosoa/note/actions/runs/38681794058).

> **Mise à jour S585–S591 — 15 octobre 2026 :** une tête apprise à quatre états (attaque, réattaque, maintien, absence/résonance), entraînée sur les **mêmes 28 sources que S578**, atteint **75,31 % de F1 NoteOn ±80 ms contre 72,46 % pour S578 sur les mêmes deux nouveaux morceaux** (Enescu et Moussorgski, 227 vraies notes). Audit : **8 notes récupérées, 4 perdues ; 14 faux positifs retirés, 5 ajoutés**. Les deux morceaux progressent à ±80 ms, mais le registre grave régresse. C'est une sonde de **deux extraits de 12 s**, pas une hausse de 59,34 à 75,31 sur un jeu constant. La variante autorisant les réattaques rapides n'est pas retenue ; aucune vraie réattaque <100 ms n'est présente dans ces deux extraits. **Pas de promotion S80 ni d'intégration Live ; rejeu complet Exact-K non effectué.** [Résultats, causes des pertes et preuves S585–S591](analysis/README_V273_S585_S591_FOUR_STATE_RESULTS.md) · [CI terminée avec succès](https://github.com/Andriamarosoa/note/actions/runs/38679314946).


**Date du bilan : 15 octobre 2026**  
**Dépôt de recherche :** https://github.com/Andriamarosoa/note  
**Dépôt de comparaison pour le temps réel :** https://github.com/Andriamarosoa/midi  
**Branche expérimentale du présent README :** codex/s533-s535-note-fundamental-loops  
**PR de recherche :** https://github.com/Andriamarosoa/note/pull/41 (**DRAFT, non fusionnée**)  

> **Bilan précédent S584 :** la transcription de notes est réellement évaluée sur des morceaux MAESTRO nouveaux pour les modèles entraînés. Le meilleur modèle de la dernière cohorte S577 est **S578 Global28 : 59,34 % de F1 NoteOn** (272 vraies notes, TP=162, FP=112, FN=110, ±80 ms), contre 58,65 % pour S569 sur **les mêmes morceaux**. Le mélange avec une tête dédiée au médium atteint 59,16 % et n'est **pas retenu**. Les différents meilleurs scores des cohortes (52,62 %, 60,02 %, 56,38 %, 59,90 %, 59,34 %) **ne constituent pas une courbe de progression à jeu constant**. Aucun modèle S533–S584 n'a remplacé la référence officielle **freeze_local_combo**, ni déployé le nouveau système dans le Live.

---

## 1. But du projet

Détecter les **vraies notes musicales** présentes dans un son, même si plusieurs notes sont jouées simultanément. La sortie cible n'est pas nécessairement un fichier MIDI : un événement peut simplement être représenté par son instant et ses noms de notes.

~~~text
Signal audio (microphone ou WAV)
    ↓
Détection d'une nouvelle attaque / événement sonore
    ↓
Exact-K : combien de NOUVELLES notes naissent à cet instant ? K0..K6
    ↓
Hauteurs de ces notes : C4, E3, G3, ...
    ↓
Suivi de la note : active, nouvelle réattaque, extinction
    ↓
Événements nommés et, facultativement, NoteOn / NoteOff MIDI
~~~

Exemples : un Mi3 seul donne **E3** ; un accord donne une liste de hauteurs ; **K0 signifie aucune nouvelle attaque, pas nécessairement le silence**. Les notes antérieures peuvent continuer et s'éteindre pendant K0.

**Objectif de recherche demandé : au moins 80 % de F1 de détection des notes**, évalué en dehors des morceaux utilisés pour décider des variantes. Il s'agit de **F1 hauteur exacte + instant NoteOn**, distinct de l'exactitude du nombre de notes, de l'exactitude d'un accord complet, du simple F1 par frame ou d'un F1 avec NoteOff.

### Définitions de mesures (à ne jamais confondre)

- **Exact-K** : proportion d'événements où le nombre prédit de *nouvelles* notes est exactement le bon. On rapporte séparément K0..K6, global et événements polyphoniques vrais K≥2.
- **Précision NoteOn** = TP/(TP+FP). Une note inventée de trop compte comme FP.
- **Rappel NoteOn** = TP/(TP+FN). Une note réelle oubliée compte comme FN.
- **F1 NoteOn** = 2TP/(2TP+FP+FN). L'évaluation MAESTRO récente impose **hauteur MIDI exacte** (ici 40..88), **apparition ±30/50/80 ms**, appariement 1:1 et comptage de *tous* les faux événements.
- **Accord exact** : toutes les hauteurs de l'accord sont bonnes, pas seulement K.
- **F1 MIDI complet** : hauteur + NoteOn + NoteOff, avec sa propre tolérance.
- **Latence Live** : délai réel entre audio capturé et émission MIDI/audio. Le temps d'un hop ou d'une seule inférence réseau n'est **pas** la latence bout-en-bout.

---

## 2. Évolution historique : S1 à S80, puis reconstitution

### Premier système : caractéristiques, têtes et sélections

Le projet a exploré plusieurs façons de distinguer fondamentales, harmoniques, résonances, nouvelles attaques et continuités de note. Les familles successives comprennent les producteurs historiques **S1/S2/S3/S4/S5**, les réseaux S18 et les étapes de décision suivantes ; les approches PR16 contiennent **36 propositions de têtes/routages**, et une reconstruction ultérieure a retrouvé **16 familles S45, 304 modèles Ridge de risque et 193 politiques S46**.

Les décisions des différentes têtes sont combinées par des **sélections** : une correction sur une famille d'événements peut devenir candidate à une nouvelle sélection, mais elle doit conserver son audit individuel (corrections, régressions, vrais K, registre, composition, sources, provenance). Les sélections ne sont pas nécessairement applicables à tous les K. Leur ordre et leurs combinaisons peuvent changer le résultat ; **une correction nette sur des données observées ne suffit pas pour promouvoir**.

Chaîne historique générale de comptage :

~~~text
WAV → représentation acoustique → ensembles de propositions S18/PR16
                                         ↓
                                 familles S45 → politiques S46
                                         ↓
                                  S69 → S70 → S71 → S72
                                                   ↓
                                        critiques/portes S79
                                                   ↓
                                          décision S80 K0..K6
~~~

Il ne faut pas interpréter **193 politiques S46** comme 193 notes reconnues ni comme 193 réseaux Live. Ce sont des variantes/règles de sélection de **comptage K**, construites avec des dépendances historiques. Le nombre des têtes dans une expérience de NoteOn récente est différent.

La reconstitution des chaînes S4/S5/S18/S45/S46 puis S69..S80 a été difficile parce que des combinaisons, modèles, caractéristiques et chemins de fichiers originaux devaient être retrouvés et réconciliés. Une série de workflows a restauré et rejoué des producteurs avec contrôle d'identité des sorties. La reconstruction reproductible d'une chaîne n'implique cependant pas, à elle seule, que son déploiement sur de nouvelles compositions est mathématiquement identique à une ancienne expérience évaluée avec ses folds d'origine.

### S76–S80 : référence expérimentale originelle

Sur **59 309** événements historiques GuitarSet, dont **7 385** événements de vérité K≥2 :

| Variante historique | K exact global | K exact polyphonique |
|---|---:|---:|
| S72 | 49 238 / 59 309 = 83,02 % | 3 202 / 7 385 = 43,36 % |
| S79 | 49 244 / 59 309 | 3 232 / 7 385 |
| **S80 initial** | **49 281 / 59 309 = 83,09 %** | **3 240 / 7 385 = 43,87 %** |

Ce **43,87 %** est le chiffre historique S80 à ne pas perdre : il ne mesure **pas** la précision du nom des notes. Le fichier de registre historique donne explicitement :

- **Référence officielle de ce jalon : freeze_local_combo**.
- **Statut S80 : candidat rejoué, non promu**, choix de certaines routes faits après audit des folds historiques.
- Audit : [S80 traceable repair](analysis/README_V273_S80_TRACEABLE_REPAIR.md) et [registre S80](analysis/v273_s80_candidate_registry.json).

### S221–S229, S272–S398 : chaîne complète, corrections et garde K5

La chaîne de producteurs réels et les étapes S69→S80 ont ensuite été rejouées dans de nombreux workflows. Les résultats de développement ne doivent pas être mélangés entre protocoles. Sur la reconstruction **S272 et la garde K5 S397–S398**, le protocole rapporte :

| Développement GuitarSet | S272 original rejoué | S397/S398 avec garde K5 |
|---|---:|---:|
| Exact-K global | 53 327 / 59 309 = 89,91 % | **53 529 / 59 309 = 90,25 %** |
| Exact-K poly | 5 852 / 7 385 = 79,24 % | **6 054 / 7 385 = 81,98 %** |
| Vrai K5 correctement reconnu | 99 / 355 | **301 / 355** |

L'intervention S397 a supprimé 203 conversions K5→K4 ; sur ce corpus exposé, 202 prédictions deviennent correctes, une reste erronée. **Ces 81,98 % poly ne sont pas le 43,87 % du S80 initial : ce sont des résultats d'un autre replay avec des protections supplémentaires et un régime d'usage des folds différent.** Une part des données GuitarSet a servi historiquement au développement ou à l'entraînement et la règle K5 a été choisie après examen des erreurs. **Ce n'est pas une précision poly indépendante certifiée ; pas de promotion de la référence officielle.**

Preuves : [S221–S229](analysis/README_V273_S221_S229_FULL_S80_INDEPENDENT_PROMOTION_GATE.md), [S397–S398](analysis/README_V273_S397_S398_FULL_S80_REPLAY_K5_GUARD_VERDICT.md), [run S397/S398](https://github.com/Andriamarosoa/note/actions/runs/38352988801).

---

## 3. Vers les vraies notes : hauteurs, attaques, fins et noms musicaux

Après le comptage K, les travaux ont recherché la **hauteur exacte** des notes, et non seulement le nombre de notes jouées.

| Étape | Réalisation vérifiée | Limite |
|---|---|---|
| **S399–S419** | Têtes MIDI40..88 conditionnées par K, émission des hauteurs nommées, scores par composition et tests d'événements | La justesse du K ne garantit pas la justesse de la hauteur |
| **S407 / S408** | Localisation de NoteOn et estimation des durées/NoteOff | Durées et source K pas nécessairement causales |
| **S422** | Estimateur hors ligne de durée exploitant jusqu'à **5 secondes d'audio futur**, F1 note complète 47,76 % dans le développement GuitarSet | **Impossible de l'insérer tel quel dans un Live à faible latence** |
| **S425** | Sélection apprise entre deux estimateurs de NoteOff ; F1 MIDI complet **48,35 %**, 16 543 notes complètes correctes, 0 modification K | Gain de développement, pas encore validé bout en bout sur musique neuve |
| **S428/S429** | Classement de noms de notes par caractéristiques acoustiques/harmoniques ; **3 025 / 7 385 = 40,96 %** d'accords poly complètement exacts, +250 vs S419 | K hérité de la chaîne ; GuitarSet de développement |
| **S430–S433** | WAV ou spectre calculé → E3 / [E3,G3,...] avec K prévu et identités d'événement préexistantes ; exemple WAV testé | Le module seul ne reconstruit **pas** toute la chaîne S80 depuis n'importe quel WAV |
| **S434–S456** | Tests réels guitare MIC/IDMT, audit des erreurs K, fondamentales, octaves et harmonique | Échecs de généralisation selon sources ; K de référence non garanti |
| **S469 et suivantes** | Nouveau protocole de NoteOn depuis **de vrais WAV MAESTRO**, vrai modèle Basic Pitch préentraîné, labels MIDI récupérés **après** gel des prédictions | Travail hors ligne, piano, morceaux courts ; pas encore le Live |

Le **score 48,35 %** de S425 évalue **NoteOn + hauteur + NoteOff** ; le **59,34 %** de S583 évalue **NoteOn + hauteur** seulement. Ils ne mesurent pas le même objet et ne se comparent pas comme une hausse de 10,99 points.

Sources : [S425–S427](analysis/README_V273_S425_S427_NATIVE_COMPLETE_MIDI_KLOCKED_RESULTS.md), [S428–S430](analysis/README_V273_S428_S430_SOUND_TO_NOTE_NAMES_KLOCKED.md), [S433–S435](analysis/README_V273_S433_S435_REAL_WAV_TO_NOTE_NAMES_AND_JAMS_AUDIT.md).

---

## 4. S469–S584 : détection expérimentale de notes sur WAV réels

### Corpus, indices et règles communes

- **MAESTRO v3** : enregistrements de piano audio+MIDI servant d'abord de développement, puis de cohortes séparées par enregistrement et composante de compositeur.
- Nouveaux extraits évalués de **12 secondes** commençant à un offset constant de 15 secondes, dans le split **officiel MAESTRO train** pour la plupart des nouveaux tests. Cela donne des *sources non utilisées pour entraîner le modèle testé*, **pas** un résultat sur le split officiel TEST ni une validation universelle hors piano.
- Chaque cohorte est choisie à partir des **métadonnées seulement** (composition, chemin, durée, SHA), en excluant les compositeurs/co-compositeurs déjà présents dans les sources utilisées par le modèle. Les premières cohortes du protocole ont aussi employé le split MAESTRO test ; les contrôles ne sont pas les mêmes entre toutes les étapes.
- **Ordre obligatoire** : fixer liste des morceaux → télécharger seulement WAV → reconstruire les événements V88 K0..K6 → inférer les notes Basic Pitch → calculer caractéristiques des 49 pitchs → charger et contrôler les modèles déjà entraînés → figer fichiers de prédictions et SHA → **seulement alors** ouvrir les MIDI pour noter TP/FP/FN.
- **49 hauteurs candidates MIDI40..88 par événement** ; Basic Pitch conserve ses naissances originales. La déduplication 80/100ms, les veto d'harmoniques et les autres correcteurs peuvent supprimer une vraie réattaque : leur effet est audité au lieu d'être proclamé sûr.
- Les prédictions de ces expériences NoteOn **ne sont pas toutes verrouillées pour émettre exactement le K S80** ; elles doivent être distinguées du chemin K-locked S428 et du pont Live K-locked.
- Les décisions, SHA, statistiques par source, couches sélectionnées, exécutions réussies/échouées et motifs de rejet restent dans des artefacts GitHub Actions.

### Évolution des caractéristiques et des têtes

| Génération | Caractéristiques par hauteur/event | Idée évaluée |
|---|---:|---|
| **S527–S530** | 49 × 11 | Force spectrale, attaque, nouveauté et indices physiques initiaux, plus note Basic Pitch ; modèles de hauteur |
| **S533–S536** | Même modèle, différents filtres | Supprimer les répétitions fautives, éviter les sous-harmoniques/harmoniques trompeurs ; audit des vraies réattaques rapides |
| **S535** | 11 ou 38 variables avec compétition harmonique | Des modèles réellement entraînés supplémentaires ont **régressé**, donc pas de substitution |
| **S546** | 7 variables acoustiques d'une hauteur par candidat, ensemble multi-source | Généralisation sur les 5 sources suivantes |
| **S562** | 34 variables | Fondamentales concurrentes inférieures (−12, −19, −24 demi-tons), attaques, nouveauté, maintien, voisinage acoustique |
| **S569** | **52 variables** | S562 + dynamique du même pitch entre événements sonores récents, perte/récupération d'énergie, tempo acoustique, réattaque et maintien |
| **S578** | 52D global + 52D médium spécialisé | Tête formée exclusivement pour MIDI60..71 ; comparaison remplacement/blend de son vecteur de probabilités |

Ce sont des caractéristiques **physiques calculées depuis l'audio** pour la prédiction. Le **MIDI vrai** sert uniquement à l'entraînement des morceaux autorisés et aux scores finaux après gel des prédictions.

### Résultats de développement et validations sur de nouvelles sources

**Attention : chaque ligne ci-dessous correspond à un ensemble différent de morceaux. Les variations verticales entre lignes ne sont pas des gains/pertes du même modèle sur le même test.**

| Étape / cohorte | Notes réelles de vérité | Meilleur F1 hauteur exacte + NoteOn ±80ms | Modèle/règle évaluée | Source |
|---|---:|---:|---|---|
| **S530 (3 premiers MAESTRO)** | 416 | 51,02 % | S528 original | [S530](https://github.com/Andriamarosoa/note/blob/codex/s530-full-maestro-noteon-f1/scripts/s530_full_maestro_stream_noteon_pitch_event_F1_audit.py) |
| **S536 (même cohorte exposée)** | 416 | 55,50 % | Dédup100 + veto harmonique ; **rétrospectif**, pas un nouveau test aveugle | [S533–S536](analysis/README_V273_S533_S536_NOTEON_REAL_DUPLICATE_RESTRIKES_HARMONIC_LOOPS.md) |
| **S545, mesuré S551** | 693 | **52,62 %** | S546 + 100ms/harmonique | [S545–S552](analysis/README_V273_S545_S552_SOURCE_DISJOINT_NOTEON_VALIDATION_AND_REGISTER_AUDIT.md) |
| **S554, mesuré S559** | 513 | **60,02 %** | S546 + 100ms/harmonique | [S559](https://github.com/Andriamarosoa/note/actions/runs/38663995955) |
| **S561, mesuré S567** | 310 | **56,38 %** | S562 34D + 100ms/harmonique | [S567](https://github.com/Andriamarosoa/note/actions/runs/38668197551) |
| **S570, mesuré S575** | 299 | **59,90 %** | **S569 52D** (23 anciennes sources d'entraînement), 100ms/harmonique | [S575](https://github.com/Andriamarosoa/note/actions/runs/38671119840) |
| **S577, mesuré S583** | 272 | **59,34 %** | **S578 Global28 52D**, 100ms/harmonique | [S583](https://github.com/Andriamarosoa/note/actions/runs/38675826715) |

**Comparaisons scientifiquement interprétables sur LES MÊMES nouveaux morceaux :**

| Vrai nouveau test | Ancien modèle même test | Challenger même test | Gain apparié |
|---|---:|---:|---:|
| **S554 / S559** | S528 + dédup80 : 57,88 % | S546 + 100ms/harmonique : 60,02 % | **+2,13 points** |
| **S561 / S567** | S546 + 100ms/harmonique : 55,60 % | S562 physical34 + même filtre : 56,38 % | **+0,78 point** |
| **S570 / S575** | S562 physical34 + 100ms/harmonique : 56,75 % | S569 temporal52 + même filtre : 59,90 % | **+3,15 points** |
| **S577 / S583** | S569 23 sources + 100ms/harmonique : 58,65 % | S578 global28 + même filtre : 59,34 % | **+0,69 point** |

Les tests comportent tous de petites cohortes (cinq nouveaux morceaux). Pour les nouveaux S545/S554/S561/S570/S577, la sélection n'a pas utilisé les **étiquettes MIDI** avant le gel des prédictions ; en revanche, **le statut d'« inconnu » se rapporte au modèle entraîné et aux protocoles d'exclusion documentés**, pas à une vérification exhaustive de chaque usage historique du dépôt. Les améliorations demeurent des **résultats de recherche**, pas une précision certifiée pour tous instruments, morceaux, micros et environnements.

### Corrections qui ont fonctionné et qui ont échoué

- **S533–S536** : supprimer des duplications proches élimine des faux NoteOn, mais il existe des **réattaques authentiques à moins de 80/100ms** ; une suppression aveugle est dangereuse.
- **S535** : des têtes ExtraTrees d'événement global entraînées avec différentes caractéristiques ont eu un F1 inférieur à S530/S533 : **rejetées**.
- **S553** : tenter le routage médium par simple vote, union ou intersection d'anciennes têtes ne résout pas la régression ; des tentatives de restaurer les réattaques ont ajouté trop de FP : **rejetées**.
- **S560** : amélioration du grave, mais régression **médium −0,97 point** dans la comparaison S546 vs S528 avec dédup80 sur S554.
- **S562** : 34 indices harmoniques réduisent les notes fantômes ; sur S561, **+0,78 point** de F1 global vs S546 avec le même filtre, mais perdent des vraies notes médiums, malgré une hausse de F1 dans le registre.
- **S569** : ajouter l'histoire acoustique de la même hauteur redonne des TP, surtout dans le grave. Sur S570, **+13 vraies notes détectées à FP inchangés** vs S562. Mais S576 vérifie **−4 TP médiums et −1,25 point de F1 médium** malgré une hausse globale.
- **S578 (28 sources)** : l'apprentissage spécialisé médium a un petit gain **interne rétrospectif** avec mélange 50/50 (62,78 % F1 contre 62,37 % global), **non confirmé** sur S577 : le mélange 59,16 % régresse face au global 59,34 %. Le remplacement pur donne 58,96 %.
- **S584, audit individuel** : le mélange au médium supprime **15 FP** au prix de **7 TP véritables** sur S577 : MIDI60/Do4 −1, 63/Mi♭4 −1, 65/Fa4 −2, 67/Sol4 −1, 70/Si♭4 −1, 71/Si4 −1. L'audit réouvre les anciens vrais MIDI et reproduit exactement le score : **diagnostic après étiquettes, pas nouvelle validation indépendante**.

**Décision du bilan S584 :** ne pas remplacer la référence avec S578 spécialiste médium. Garder S569 temporal52 et S578 Global28 comme candidats expérimentaux aux rôles distincts ; améliorer un classifieur explicite **véritable réattaque / sustain / harmonique** sans sacrifier les notes réelles. À ce stade S584, le score **80 % NoteOn F1** n'était pas atteint. **S605 le franchit ensuite sur cinq œuvres URMP réservées, à 84,91 % ; ce nouveau corpus ne constitue pas un rejeu des cohortes piano précédentes.**

Pour les sources et chiffres détaillés : [S533–S536](analysis/README_V273_S533_S536_NOTEON_REAL_DUPLICATE_RESTRIKES_HARMONIC_LOOPS.md), [S553–S560](analysis/README_V273_S553_S560_TWO_SOURCE_HELD_NOTEON_TESTS_MEDIUM_FAILURE.md), [S561–S568](analysis/README_V273_S561_S568_EIGHTEEN_SOURCE_PHYSICAL_NOTEON_AND_THIRD_PROSPECTIVE_TEST.md), [S569–S576](analysis/README_V273_S569_S576_TEMPORAL_RESTRIKE_23_SOURCES_FOURTH_BLIND_NOTEON_TEST.md), [S577–S584](analysis/README_V273_S577_S584_MIDDLE12_SPECIALIST_FAILS_FIFTH_HELD_F1.md).

---

## 5. LIVE dans le dépôt note : ce qui fonctionne réellement

Le travail de S424 a **explicitement préservé les anciens modes causaux**. Il ne faut pas écrire « Live supprimé » ou « Live entièrement entraîné S80 » : les deux affirmations seraient fausses.

### Entrées existantes

1. **live-boundaries** : [scripts/detect_live.py](scripts/detect_live.py), [src/causal_note/pipeline.py](src/causal_note/pipeline.py). Le détecteur causal traite des blocs WAV ou PCM16 44 100 Hz, sort des **onset/offset avec identités d'événement** ; taille de bloc par défaut **512 échantillons**, soit **11,61 ms**. Il conserve son modèle et sa logique historique.
2. **live-v8** : [scripts/detect_v8_live.py](scripts/detect_v8_live.py) ; modèle causal de frontières avec association d'événements, WAV ou PCM16, **512 échantillons par défaut**.
3. **live-midi** : [scripts/stream_k_midi_live.py](scripts/stream_k_midi_live.py) et [src/causal_note/live_k_midi_bridge.py](src/causal_note/live_k_midi_bridge.py). **Pont événementiel MIDI causal**. Il reçoit les prédictions **déjà calculées en amont** : horodatage en échantillons, valeur K0..K6, liste de **exactement K** hauteurs MIDI, et éventuellement des identifiants NoteOff. Il émet NoteOn/NoteOff, peut envoyer vers un port MIDI et sauvegarde un fichier .mid à fin de flux validée.

Entrée commune : [scripts/note_runtime.py](scripts/note_runtime.py).

~~~bash
PYTHONPATH=.:src python -m scripts.note_runtime --mode status

# Le modèle causal de frontières doit déjà exister comme .keras :
PYTHONPATH=.:src python -m scripts.note_runtime \
  --mode live-boundaries --model /chemin/causal.keras \
  --stdin-pcm16 --chunk-size 512

# Le pont attend des décisions déjà prédites, PAS du WAV :
PYTHONPATH=.:src python -m scripts.note_runtime \
  --mode live-midi --stdin-jsonl --midi-file session.mid
~~~

Exemple **de protocole**, pas une prédiction acoustique vérifiée :

~~~jsonl
{"type":"decision","sample":128,"s80_K":2,"midi_pitches":[52,55]}
{"type":"decision","sample":256,"s80_K":0,"midi_pitches":[]}
{"type":"decision","sample":384,"s80_K":0,"midi_pitches":[],"note_off_ids":["note-000000001"]}
{"type":"decision","sample":512,"s80_K":1,"midi_pitches":[52]}
{"type":"eof","end_sample":1024}
~~~

### Protections Live déjà présentes

- **K0 = aucune nouvelle note**, mais la note déjà active peut rester ou recevoir un NoteOff.
- **Exactement K NoteOn** par décision d'événement, avec des notes toutes distinctes dans un même accord.
- Une **réattaque de même hauteur** ferme d'abord l'instance précédente par NoteOff, puis ouvre un nouveau NoteOn.
- Pitches MIDI protégés **40..88**, K0..K6, ordre croissant des timestamps, identifiants de note stables.
- Fin de flux explicite ; NoteOff restants émis au vrai EOF, sans inventer des fins de notes en absence de mesure.
- Le pont événementiel n'importe pas et ne remplace pas les modèles du Live historique.

### CE QUI N'EST PAS implémenté/certifié dans note

- Pas de chaîne **microphone → S80 K causal sur chaque hop → modèle causal de notes S569/S578 → NoteOff fiable → MIDI** fonctionnant et testée de bout en bout.
- Le pont **live-midi ne calcule ni K ni les hauteurs** : une entrée JSONL qui les fournit reste nécessaire.
- Les caractéristiques V88 et les têtes NoteOn récentes sont utilisées dans des workflows sur **WAV de 12 s hors ligne** ; cela ne certifie ni un calcul par bloc en temps réel ni une latence de production.
- **S422 utilise jusqu'à 5 s de futur pour NoteOff** et n'est donc PAS un estimateur Live à faible latence dans cette forme.
- **Aucune latence micro→notes NoteOn/NoteOff bout-en-bout de S80/S569/S578 dans note n'est encore mesurée**. Le calcul 512/44 100 = 11,61 ms donne uniquement la durée du bloc d'entrée par défaut, pas le délai musical d'attaque.

Références : [protection S424](analysis/README_V273_LIVE_OFFLINE_KLOCKED_PRESERVATION.md), [manifest Live/Offline](analysis/v273_live_and_offline_protected_modes.json), [tests du pont](test/test_live_k_midi_bridge.py).

---

## 6. Comparaison pratique du Live : repository midi

**Référence étudiée directement dans** [Andriamarosoa/midi](https://github.com/Andriamarosoa/midi), branche **main** (snapshot inspecté en octobre 2026).

Le dépôt **midi** dispose d'une chaîne **guitare → MIDI réellement exécutable depuis un périphérique audio**, notamment :
- [src/polyphonic/live.py](https://github.com/Andriamarosoa/midi/blob/main/src/polyphonic/live.py) : application Live polyphonique à capture microphone et sortie notes ;
- [src/polyphonic/model.py](https://github.com/Andriamarosoa/midi/blob/main/src/polyphonic/model.py) : réseau causal multi-sorties **frame + onset par hauteur**, support d'amplitude/écart des **20 partiels harmoniques** ;
- [src/polyphonic/decoder.py](https://github.com/Andriamarosoa/midi/blob/main/src/polyphonic/decoder.py) : machine à états par hauteur, activation, release, détection d'attaque récente, retrigger, extinction, contrôle des pics harmoniques ;
- [src/stream/onset_detector.py](https://github.com/Andriamarosoa/midi/blob/main/src/stream/onset_detector.py) et [src/stream/ring_buffer.py](https://github.com/Andriamarosoa/midi/blob/main/src/stream/ring_buffer.py) : onset adaptatif, flux spectral/RMS, buffer causal ;
- [src/product/live.py](https://github.com/Andriamarosoa/midi/blob/main/src/product/live.py), [src/product/backpressure.py](https://github.com/Andriamarosoa/midi/blob/main/src/product/backpressure.py), [src/product/audio_io.py](https://github.com/Andriamarosoa/midi/blob/main/src/product/audio_io.py) : capture I/O, file bornée, retard mesuré et récupération de surcharge ;
- [src/product/midi_output.py](https://github.com/Andriamarosoa/midi/blob/main/src/product/midi_output.py) : port MIDI WinMM, panorama des notes actives et **panic All Notes Off**, synthétiseur optionnel FluidSynth ;
- [métadonnées V2.2.1](https://github.com/Andriamarosoa/midi/blob/main/artifacts/guitar_midi_polyphonic_v2_2_0/metadata.json) et [README midi](https://github.com/Andriamarosoa/midi/blob/main/readme/README.md).

### Comparaison vérifiée

| Point | **note** | **midi** |
|---|---|---|
| Objectif historique dominant | Exact-K, reconnaissance des vraies notes, architecture par têtes et sélections, audits de régression | **Guitare → événements MIDI causaux en Live** |
| Capture microphone directement intégrée à la chaîne de notes récente | **Non** pour S80/S569/S578 ; le Live historique lit WAV/PCM16 | **Oui**, via PortAudio/sounddevice, endpoint audio et callbacks |
| Fréquence | 44 100 Hz pour Live original | 44 100 Hz |
| Unité de traitement audio Live | **512 échantillons par défaut**, soit ~11,61 ms | **256 échantillons**, soit ~5,80 ms |
| Mémoire de contexte | Modèle historique causal ; fenêtre complète récente S80 non consolidée en moteur live | **Ring buffer 4 096**, fenêtres progressives **512 / 1 024 / 2 048 / 4 096** |
| Réseau note poly Live réellement branché | **Non** : les têtes S569/S578 ont surtout été évaluées hors ligne ; le pont attend des notes déjà prédites | **Oui** : réseau causal par hauteur avec sorties frame, onset et indices harmoniques |
| Cardinalité | Recherche K0..K6 ; le pont Live impose **exactement K nouvelles notes** déjà fournies | Maximum de **6 notes polyphoniques actives** dans le décodeur, mais **pas** le même estimateur Exact-K S80 |
| Plage du détecteur de hauteurs | 49 pitchs **MIDI40..88** dans les expériences NoteOn / pont Live | Produit poly V2.2.1 : **MIDI40..76** |
| NoteOff | Code Live d'événements historique et pont à IDs ; estimateur S422 avec futur **hors ligne** | **Machine à états causale par pitch**, release/silence, retrigger et panic |
| Protection harmonique | Caractéristiques 34D/52D, correcteurs séparés et audits des vraies notes perdues | Indices 20 partiels, gate d'attaque physique récente, seuil fort **0,90** pour notes créées sans attaque récente |
| CPU et inférence | **Aucun benchmark p95 de la chaîne complète S80→notes Live** | TFLite modèle poly, **3 threads recommandés** après essais de stabilité |
| Surcharge / audio manquant | Propriétés de flux du Live historique ; pas de contrat micro↔S569 complet | **Backpressure borné**, suivi queue, abandon de vieux blocs, réinitialisation/panic lors des incidents |
| Latence musicale end-to-end directement comparable | **Non mesurée** | Mesures partielles matérielles et runtime publiées ; loopback acoustique complet reste distinct de ces mesures |

### Latence : ce qui est réellement mesuré dans midi

Le fichier [latency_report V2.2](https://github.com/Andriamarosoa/midi/blob/main/artifacts/guitar_midi_polyphonic_v2_2_0/latency_report.json) indique pour **l'inférence TFLite du réseau polyphonique** avec trois threads recommandés :
- **p95 = 2,597 ms**, p99 ~**3,468 ms** sur les validations de saturation mentionnées ;
- Budget de **5,805 ms par hop de 256 échantillons** ;
- Critères de runtime satisfaits dans ce protocole ; TFLite float16/entrées-sorties float32. Cela n'inclut **pas** toutes les attentes audio, la période de confirmation, le décodeur et la sortie.

Le fichier matériel du moteur monophonique V1 [hardware_live_report_20s.json](https://github.com/Andriamarosoa/midi/blob/main/artifacts/guitar_midi_v1_0_0/hardware_live_report_20s.json) donne, sur une capture réelle de 20 s :
- **pipeline p95 ~31,66 ms** dans le protocole instrumenté ;
- boucle CPU p95 **~7,97 ms** ;
- **0 blocs audio perdus**, **2,01 % des inférences sautées** pour tenir la surcharge.

Ce rapport V1 monophonique **n'est pas** la latence mesurée du modèle polyphonique V2.2.1 et n'inclut pas nécessairement l'instant acoustique réel de frappe ni toute la chaîne jusqu'à l'auditeur. Le README de midi rapporte également ~**20,79 ms d'I/O négociée** dans une configuration Windows donnée, et précise que la **latence acoustique exacte nécessite une mesure loopback**.

**Comparaison honnête :** nous pouvons conclure que midi a déjà un *runtime temps réel complet*, un hop 5,80 ms et des instruments de mesure ; nous ne pouvons **pas** conclure que « midi répond en 2,60 ms bout-en-bout » ni calculer une latence exacte de note simplement à partir du hop 11,61 ms. Il faut exécuter les deux moteurs sur le **même matériel**, même micro, même protocole d'attaque et vraie sortie MIDI.

### Particularité de l'architecture midi à conserver pour note

La reconnaissance du son ne se résume pas à identifier des pics de fréquences :
- **Frame par hauteur** : indique qu'une note est encore acoustiquement présente.
- **Onset par hauteur + attaque physique** : indique qu'une nouvelle note commence.
- **NoteOff par hauteur** : condition de release/silence, disparition de la note ou vraie extinction.
- **Réattaque** : un nouveau NoteOn sur la même hauteur doit pouvoir arriver même si cette hauteur était déjà active.
- **Partiels harmoniques** : permettent de différencier une fondamentale d'un pic dû à une autre note.
- **État causal + hystérésis** : évite de traiter le son comme des observations isolées.
- **Backpressure** : interdit qu'une meilleure précision hors ligne se paie par une file de traitement qui grossit sans limite.

Le runtime poly V2.2.1 règle entre autres : seuil activation frame ~0,535, onset ~0,665, note non attaquée ~0,90, minimum de réattaque 14 frames et maximum de 6 notes actives. **Ces valeurs sont propres à midi et à son domaine GuitarSet/GAPS/Guitar-TECHS ; elles ne doivent PAS être copiées arbitrairement comme seuils universels dans note**, dont la plage et les cibles sont différentes.

Commande **midi** (Windows, application déjà implémentée) :

~~~powershell
.\START_LIVE_POLYPHONIC.bat --console-midi --duration-s 60
~~~

La présence du Live complet dans **midi** ne veut pas dire que son modèle de hauteur ou son décodeur donnera de meilleurs scores sur les 49 pitchs et les K0..K6 de **note** ; cela doit être testé sans contaminer les jeux de référence.

---

## 7. Comment intégrer le meilleur des deux, sans perdre Exact-K ni Live

La bonne solution n'est pas de remplacer aveuglément le Live de **note** par le réseau de **midi** : la tâche est différente. Il faut **réutiliser l'architecture de flux causal de midi** et **conserver les têtes/routages, preuves et règles musicales de note**.

### Architecture cible proposée (PAS encore livrée)

~~~text
MIC 44.1 kHz
  │
  ├─ capture/queue/backpressure : architecture causale inspirée de midi
  │
  └─ ring buffer 4096, hop 256 (nouveau chemin expérimental, sans remplacer
      le live historique 512)
        │
        ├─ tête d'attaque physique : RMS + flux spectral + preuve temporelle
        │
        ├─ tête d'activité par pitch : notes déjà actives, sustain, harmoniques
        │
        ├─ tête de nouvelle attaque / réattaque par pitch : séparée du sustain
        │
        └─ têtes Exact-K causal : K0..K6, S80 de comparaison PROTÉGÉ
                     │
           politiques entraînées et auditées
                     │
     sélection des K nouvelles notes MIDI40..88
                     │
       invariant : EXACTEMENT K nouveaux NoteOn
                     │
      machine d'état NoteOn / NoteOff / retrigger
                     │
          MIDI / noms de notes / synthé audio
                     │
     télémétrie timing, F1, erreurs et régressions
~~~

La tête de hauteur peut utiliser la physique **34D/52D S562/S569/S578** ; elle devra être **adaptée et mesurée en contexte strictement causal**. Elle ne doit pas dépendre d'une fenêtre future de cinq secondes comme S422. Le système devra différencier :
1. **fondamentale déjà présente sans attaque** ;
2. **nouvelle fondamentale avec attaque physique récente** ;
3. **réattaque réelle de la même hauteur active** ;
4. **harmonique/sous-harmonique/résonance étrangère** ;
5. **NoteOff réel ou note étouffée**.

Les règles de sécurité et de scoring Exact-K restent des **garde-fous séparés de la sélection des hauteurs** : K0 peut coexister avec des notes actives, et la prédiction de hauteur ne doit **jamais** recalculer K en comptant simplement ses pics.

### Ordre d'intégration recommandé

**A. Sécuriser le comportement actuel :**
- conserver sans modification les quatre fichiers Live historiques, les poids actifs, le profil S424, **freeze_local_combo**, toutes les archives S80/S397/S425/S428 et les scores ;
- rejouer les tests d'entrées WAV/PCM16, ordre des timestamps, K0, réattaque, extinction, EOF et sortie MIDI ;
- imposer audit individuel à chaque tête et à chaque nouvelle sélection, avec liste complète des corrections/régressions.

**B. Construire un vrai frontal causal sur une branche expérimentale :**
- isoler capture audio et inférence comme dans midi (**le callback ne fait pas le calcul lourd**), ring, file bornée et timestamps en échantillons ;
- utiliser le hop **256** en nouvelle expérimentation et garder **512** comme contrôle historique ; ne pas imposer la taille 256 tant que ses effets sur la précision ne sont pas mesurés ;
- produire les caractéristiques de chaque note en causal sans voir les trames futures ;
- ajouter deux sorties conditionnelles distinctes **frame-active** et **new-onset/restrike**, avec indices de concurrence harmonique et de transitoire ;
- conserver le futur K-first si sa version réellement causale est validée.

**C. Conserver le meilleur des têtes S569/S578, mais corriger leur échec réel :**
- former une tête *nouveau NoteOn* explicitement annotée (positive = vraie attaque, même en présence de la même hauteur ; négative = sustain/résonance) au lieu d'un simple filtre d'identité ;
- ne pas effacer automatiquement une nouvelle note sous 80/100ms ;
- inclure des réattaques et des notes étouffées de **guitare GuitarSet/IDMT** et piano MAESTRO, avec compositeurs/pièces/instruments strictement disjoints ;
- suivre *par MIDI60..71* les vrais TP perdus et les FP supprimés. Rejeter une « amélioration » qui recrée le problème de S584 ;
- les multiples têtes et sélections doivent apprendre quand s'abstenir, et l'addition d'une sélection n'est acceptable que si un test indépendant confirme son gain et quantifie les régressions.

**D. Mesurer la validité musicale ET la latence :**
- sur morceaux inédits, rapporter **F1 NoteOn ±30/50/80ms**, accord exact, Exact-K par classe, K5/K6, par instrument, registre et compositeur, les NotesOff/notes longues ;
- comparer à la référence et aux challengers sur **les mêmes WAV** avec **les mêmes vrais labels** et modèle gelé avant MIDI ;
- sur matériel Live, mesurer p50/p95/p99 de capture, attente de fenêtre, temps d'inférence, décodeur, émission MIDI, temps audio→NoteOn total, profondeur de file, pertes/blocs sautés, déclenchements du panic, et coût CPU ;
- mesurer aussi l'instant d'attaque via loopback ou événement source instrumenté : le temps d'inférence seul ne suffit pas ;
- analyser note frappée, accord dense, répétition rapide, sustain long et note étouffée.

### Conditions de promotion (bloquantes)

Un nouveau **référent officiel** ne doit être adopté que si :
1. il est réellement inférable sur des WAV/micros **sans entrée de vérité MIDI/JAMS** ;
2. le protocole de test est décidé avant tout accès aux annotations, et tient des œuvres/instruments réellement exclus de tous les entraînements/choix d'hyperparamètres ;
3. il améliore **le F1 NoteOn** sans régressions majeures de registre et détaille tous les TP/FP/FN ;
4. il ne dégrade pas l'**Exact-K K0..K6**, y compris les classes poly rares ;
5. s'il remplace un modèle Offset, il améliore **la note complète NoteOn + hauteur + NoteOff**, pas simplement les fenêtres ;
6. il respecte le contrat causal et un budget de **latence mesurée bout-en-bout**, pas une estimation fictive ;
7. les fichiers et scores du Live original sont inchangés jusqu'à un basculement explicitement validé avec possibilité de rollback.

**Aucune de ces conditions ne peut être déclarée globalement remplie par le seul score S583.**

---

## 8. État résumé : acquis, prototypes, blocages

| Composant | État au 15 octobre 2026 |
|---|---|
| Recherche historique Exact-K, têtes et sélections | **Réalisée, plusieurs rejouements et audits** |
| Référence officielle de départ freeze_local_combo | **Conservée** |
| S80 initial et S397 gardé K5 | **Résultats de développement retrouvés ; nouvelle promotion non certifiée** |
| K-locked et noms des notes E3 / accords | **Implémentations/prototypes et audits réalisés** |
| Estimateurs de fin de note S408, S422, S425 | **Développement hors ligne**, S422 non causal |
| MAESTRO WAV → nouveaux événements de notes MIDI40..88 | **Entraîné et mesuré** jusqu'à S584 sur plusieurs nouvelles cohortes |
| Meilleur modèle récent en NoteOn sur cohorte S577 | **S578 Global28, F1 59,34 %**, non officiel |
| S569 temporal52 sur nouvelle cohorte S570 | **F1 59,90 %**, non officiel, autre jeu |
| S585 quatre états + dédup100, sonde S591 à deux nouveaux compositeurs | **75,31 % F1 vs S578 72,46 % sur les mêmes morceaux**, candidat non promu ; grave en régression |
| S601, filtre appris sur candidats BP natifs, test S605 | **84,91 % vs BP 73,51 %**, 38 notes récupérées/55 perdues ; non promu |
| Tête 12 notes médiums S578 | **Rejetée** après validation S583 et audit S584 |
| Note Live historique 44,1k / 512 | **Préservé, fonctionnel avec ses propres modèles**, non S80 complet |
| Pont note Live MIDI K-locked | **Fonctionnel sur entrée K+notes déjà prédites**, pas détection audio complète |
| Micro → S80 → hauteurs → NoteOff en vrai Live | **Non réalisé / non validé** |
| Latence bout-en-bout note de ce nouveau pipeline | **Inconnue** |
| Pipeline microphone → poly MIDI causal dans midi | **Implémenté** ; qualité et latence dépendantes du profil/matériel |
| 80 % de F1 NoteOn sur cinq œuvres URMP réservées | **Atteint : S601/S605, 84,91 %**, 626 notes, 120 s ; Miserere 75 %, hors ligne |
| Généralisation tous instruments, pièces et microphones | **Non certifiée** ; validation réservée de portée limitée |
| Fusion dans la référence officielle | **Non. PR #41 DRAFT.** |

---

## 9. Liens de travail et reproductions

### Dépôts, PR et états protégés

- Dépôt **note** : https://github.com/Andriamarosoa/note
- Dépôt **midi** : https://github.com/Andriamarosoa/midi
- PR expérimentale **note #41** : https://github.com/Andriamarosoa/note/pull/41
- [Référence officielle/challengers et protections Live](analysis/v273_live_and_offline_protected_modes.json)
- [README Live protégé S424](analysis/README_V273_LIVE_OFFLINE_KLOCKED_PRESERVATION.md)
- [S80 historique](analysis/README_V273_S80_TRACEABLE_REPAIR.md)

### Exécutions principales et rapports

| Série | Exécution GitHub | Rapport |
|---|---|---|
| S601/S605, premier test réservé | [38685400020](https://github.com/Andriamarosoa/note/actions/runs/38685400020) | [S598–S605](analysis/README_V273_S598_S605_RESERVED_80_RESULTS.md) |
| S397/S398 K5 | [38352988801](https://github.com/Andriamarosoa/note/actions/runs/38352988801) | [S397/S398](analysis/README_V273_S397_S398_FULL_S80_REPLAY_K5_GUARD_VERDICT.md) |
| S425 NoteOff | [38513725784](https://github.com/Andriamarosoa/note/actions/runs/38513725784) | [S425/S427](analysis/README_V273_S425_S427_NATIVE_COMPLETE_MIDI_KLOCKED_RESULTS.md) |
| S428 notes | [38517551029](https://github.com/Andriamarosoa/note/actions/runs/38517551029) | [S428/S430](analysis/README_V273_S428_S430_SOUND_TO_NOTE_NAMES_KLOCKED.md) |
| S559 nouveau NoteOn | [38663995955](https://github.com/Andriamarosoa/note/actions/runs/38663995955) | [S553/S560](analysis/README_V273_S553_S560_TWO_SOURCE_HELD_NOTEON_TESTS_MEDIUM_FAILURE.md) |
| S562 physique34 | [38667229236](https://github.com/Andriamarosoa/note/actions/runs/38667229236) | [S561/S568](analysis/README_V273_S561_S568_EIGHTEEN_SOURCE_PHYSICAL_NOTEON_AND_THIRD_PROSPECTIVE_TEST.md) |
| S569 temporal52 | [38670001388](https://github.com/Andriamarosoa/note/actions/runs/38670001388) | [S569/S576](analysis/README_V273_S569_S576_TEMPORAL_RESTRIKE_23_SOURCES_FOURTH_BLIND_NOTEON_TEST.md) |
| S575 nouvelle validation | [38671119840](https://github.com/Andriamarosoa/note/actions/runs/38671119840) | [S569/S576](analysis/README_V273_S569_S576_TEMPORAL_RESTRIKE_23_SOURCES_FOURTH_BLIND_NOTEON_TEST.md) |
| S578 spécialiste12 | [38675146653](https://github.com/Andriamarosoa/note/actions/runs/38675146653) | [S577/S584](analysis/README_V273_S577_S584_MIDDLE12_SPECIALIST_FAILS_FIFTH_HELD_F1.md) |
| S583 cinquième validation | [38675826715](https://github.com/Andriamarosoa/note/actions/runs/38675826715) | [S577/S584](analysis/README_V273_S577_S584_MIDDLE12_SPECIALIST_FAILS_FIFTH_HELD_F1.md) |
| S584 audit perte vraie note | [38676593804](https://github.com/Andriamarosoa/note/actions/runs/38676593804) | [S577/S584](analysis/README_V273_S577_S584_MIDDLE12_SPECIALIST_FAILS_FIFTH_HELD_F1.md) |

### Lecture du dépôt midi Live

- [README et commandes Windows](https://github.com/Andriamarosoa/midi/blob/main/readme/README.md)
- [Spécification de streaming causal](https://github.com/Andriamarosoa/midi/blob/main/docs/v5/STREAMING.md)
- [Application Live polyphonique](https://github.com/Andriamarosoa/midi/blob/main/src/polyphonic/live.py)
- [Réseau frame/onset/harmoniques](https://github.com/Andriamarosoa/midi/blob/main/src/polyphonic/model.py)
- [Décodeur d'événements par hauteur](https://github.com/Andriamarosoa/midi/blob/main/src/polyphonic/decoder.py)
- [Latence TFLite validation poly](https://github.com/Andriamarosoa/midi/blob/main/artifacts/guitar_midi_polyphonic_v2_2_0/latency_report.json)
- [Rapport matériel Live monophonique](https://github.com/Andriamarosoa/midi/blob/main/artifacts/guitar_midi_v1_0_0/hardware_live_report_20s.json)

---

## 10. Conclusion

**Dernier résultat S605 : 84,91 % de F1 NoteOn sur cinq œuvres URMP réservées, après apprentissage sur 37 autres enregistrements et validation groupée par titre.** L’objectif de 80 % est franchi sur cette cohorte hors ligne ; le rappel reste à 77,32 %, et le test ne valide ni NoteOff, ni les réattaques rapides, ni le Live. Les pertes et les 30 faux positifs sont publiés.

**Ce qui a avancé :** reconstruction des têtes et sélections Exact-K, réalisation des véritables sorties **son → hauteurs musicales**, audit NoteOn et NoteOff, expérience de physique spectrale 34D, puis temporelle 52D, apprentissage sur **28 musiques** et plusieurs tests sur de nouvelles compositions avec gel SHA avant les MIDI. La détection des notes ne se réduit plus à deviner K. Chaque correction est conservée avec ses régressions.

**Ce qui reste non résolu :** le médium (60..71), l'arbitrage entre véritable réattaque et simple persistance harmonique, la généralisation entre instruments, les notes très longues et courtes pour NoteOff, la généralisation indépendant Exact-K, la **chaîne complète Live**.

**Ce que le dépôt midi apporte réellement :** une **méthode opérationnelle de capture et d'inférence causale au microphone**, un décodeur actif par hauteur, des protections temps réel et des mesures de latence. Sa conception doit servir de base à une **nouvelle implémentation Live expérimentale de note**, mais ses poids, seuils, résultats et sa plage MIDI ne doivent pas être confondus avec ceux de note.

**Règle de sécurité du projet :** ne jamais effacer une piste ou un correcteur utile, conserver ses audits, ne jamais confondre un bon score rétrospectif avec un résultat indépendant, **ne pas promouvoir sans mesure de non-régression**, et laisser **freeze_local_combo / S80 / Live historique** intacts tant que le successeur n'a pas démontré son avantage dans les mêmes conditions.
