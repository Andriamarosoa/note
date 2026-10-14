# S485–S490 — Résultats multi-instrument : audio réel → noms des notes, Exact-K et audit individuel des régressions

**Date : 14 octobre 2026. Expérimental, non promu.** Ne modifie ni le vrai `freeze_local_combo`, ni S80 officiel, ni Live, ni les onsets/offsets et le décodeur des noms de notes historique.

## Motivation
Les audits de V88 révèlent un effondrement vers K0 sur 233 vraies attaques de piano MAESTRO, et les accords guitare IDMT comportent aussi des surcomptages. Les correcteurs purement GuitarSet S478–S483 montrent des gains par validation interne (+64 et +7 vrais poly avec zéro régression poly) mais **aucun transfert contrôlé sur 45 attaques IDMT**. On a évalué séparément **Basic Pitch 0.4.0 préentraîné multi-instrument**, sans l'entraîner ou l'ajuster sur ces enregistrements. L'objectif final est de reconnaître les notes réelles, par ex. `E3`, `G3`, `B3`, et non de produire un fichier MIDI.

## Vrai piano MAESTRO (trois extraits, 36 s)
- [S485 SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/38568724837) : 317 notes prédites sur 3 vrais WAV de piano vérifiés SHA ; les sorties du modèle sont figées avant le scoring.
- [S486 SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/38569142478) : les **mêmes 322 attaques** MIDI/MAESTRO déjà appariées aux événements V88, vérité des **nouvelles notes** regroupées sur 20 ms et appariement à ±80 ms. **V88 K correct = 42/322 (13,04 %)** contre **Basic Pitch = 151/322 (46,89 %)**. Sur 73 vrais accords poly, **7/73 (9,59 %) contre 21/73 (28,77 %)**. Ensembles complets exacts : **125/322 (38,82 %)** dont **18/73** accords. Note précision sur attaques appariées ~75,08 %, note rappel ~55,39 %. Les groupes prédits du réseau sont 234, dont 220 appariés à 322 vraies attaques ±80 ms : la couverture ne doit pas être oubliée lorsque l'Exact-K est calculé sur tous les 322 événements.
- [S490 SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/38574162349) : contre-factuel d'un basculement global **V88 → Basic Pitch** sur les 322 *mêmes événements* : **124 corrections**, **15 régressions**, **net +109** ; poly **17 corrections**, **3 régressions**, **net +14**. Parmi **233 vraies attaques classées K0 par V88**, **105** sont correctement récupérées par Basic Pitch. Ces 15 régressions interdisent un remplacement sans conditions.

## Vraie guitare IDMT, mêmes cinq WAV originaux (50 s)
- [S487 SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/38569589535) : cinq véritables WAV d'origine IDMT, extraits 10 s et SHA original S277 vérifié, Basic Pitch pré-entraîné avec seuils fixes ; note-name predictions figées **avant** les XML.
- [S488 SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/38570219836) : exactement **45 attaques XML** dans MIDI40..88, dont **21 vrais accords poly** selon les notes actives à +30 ms, comme pour le K source S80. **S80 K 25/45 = 55,56 %** contre **Basic Pitch 30/45 = 66,67 %**. Poly **S80 7/21 = 33,33 %** contre **Basic Pitch 11/21 = 52,38 %**. Sur les ensembles de notes complets, **S430 historique 21/45** contre **Basic Pitch 25/45**, et vrais poly **S430 4/21** contre **Basic Pitch 7/21**. Cela ne signifie pas que la fusion est évaluée ; seule la substitution hypothétique est comparée.
- [S490 SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/38574162349) : **12 corrections, 7 régressions, gain net +5** en remplaçant le K S80 sur les 45 attaques vraies. Sur les 21 accords, **7 corrections, 3 régressions, net +4**. Toutes les **7 vraies attaques** que S80 avait classées K0 sont récupérées en K exact par Basic Pitch. Mais les 7 régressions individuelles parmi des cas S80 déjà corrects interdisent le déploiement inconditionnel. Les **459 événements IDMT sans attaque XML appariée** ne sont PAS des négatifs sûrs, ni des vrais K0, et ne peuvent pas valider un filtre sans une vérité d'événements positifs/négatifs complète.

## Interface son → notes (utilisable expérimentalement)
- [S489 SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/38571220996) : `scripts/s489_sound_to_note_names_pretrained_cli.py` a été exécuté avec un vrai fichier WAV. Il rend un **JSON strict** avec pour chaque groupe d'attaque `note_names`, `K_new_notes`, `onset_seconds`, les `offset_seconds` des notes et un score non calibré ; **pas de fichier MIDI exporté**. Le débogage a corrigé les impressions parasites du modèle sur stdout, dirigées vers stderr.
- Exemple local après installation de Basic Pitch et de ses dépendances :
  `python -m scripts.s489_sound_to_note_names_pretrained_cli --audio mon_son.wav --pretty`
- L'interface n'est **pas encore une chaîne Live officiellement intégrée**. Les notes E3, G3, B3 sont les noms réellement fournis par le réseau à partir d'une piste audio ; l'exemple de test `E3,G3,B3` est un **selftest**, pas une preuve que l'accord se trouve dans le WAV testé.

## Verdict de sécurité avant toute fusion

1. **Ne pas remplacer S80 directement par Basic Pitch** : les deux sources affichent des régressions (7/45 IDMT et 15/322 piano) malgré des gains nets.
2. **Prioriser une tête de récupération des vrais K0 à l'attaque**, car Basic Pitch a retrouvé 7/7 K0 erronés IDMT et 105/233 MAESTRO, mais attention : on ne connaît pas de véritables négatifs pour les 459 candidats IDMT non appariés ; un sauvetage aveugle des K0 pourrait accroître le bruit faux-positif.
3. **Construire une fusion sélective apprenante** à partir de données d'entraînement indépendantes des 5 WAV IDMT et 3 MAESTRO test, sans choisir les règles sur leurs annotations maintenant exposées. Bloquer toute promotion tant que les régressions par vraie classe K et source ne sont pas maitrisées sur des **nouveaux enregistrements**.
4. **Ne pas confondre V88 et S80** : comparaison IDMT avec la véritable chaîne S80, comparaison MAESTRO avec V88 uniquement. La validation de S80 complet sur ce troisième corpus reste à faire avant toute affirmation officielle.

**Résultat :** une piste instrument-générale forte de transcription des notes existe et un prototype WAV→noms des notes fonctionne, mais **une amélioration sans régression de S80 reste non démontrée**.
