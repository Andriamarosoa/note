---
serie: S792
statut: termine
type: experience
reference: S708
branche: codex/note-s792-framepitch-six-metric-midi-replay
run_id: 39328341071
commit_sha: aec47655049d89ddf297348ca1ec89b08c8d1daf
promotion: false
---

# S792 — Résumé

## Objectif

Évaluer **sur le véritable flux de notes MIDI NoteOn/NoteOff** si le gain Frame Pitch SHADOW S790 (85,12 % sur CAL32 à horloge 1024) peut être transféré dans la mesure officielle `scripts.s704_joint_metrics.metric` à horloge 256, sans perte d'Onset, Exact-K, Active Exact-K, Offset ni Full Note.

## Hypothèse

Le gain SHADOW S790 peut améliorer l'occupation des notes lorsque le décodeur officiel coupe certaines durées trop tôt ; cependant, une prolongation naïve peut produire une fin de note incorrecte et dégrader Full Note. Cette expérience doit tester et, si nécessaire, réfuter cette hypothèse sur de vrais événements MIDI.

## Modifications

- Référence S708 immuable, modèle S707 immuable.
- Tête S790 indépendante, entraînée sur GuitarSet FIT80, politique de tête gelée `cut0.45_harm0.00`.
- Rejeu des **NoteOff seulement**, jamais de NoteOn inventé, avec rétention causale bornée de 0, 1, 2 ou 4 décisions 1024 échantillons.
- Retenue accordée seulement si la tête S790 indique que la hauteur est encore active au moment du NoteOff ; sortie retardée jusqu'à extinction de la preuve, nouveau NoteOn de la même hauteur, limite des six voix, échéance ou EOF.
- Mesure commune des six axes sur les événements MIDI effectivement émis ; vérification des comptes et scores de la référence signée S708.
- Sélection d'une durée sur **FIT80 uniquement**, validation hors composition en cinq plis, garde contre chaque régression. CAL32 appliqué après gel du choix ; HELD/TEST jamais ouverts.

## Protocole

FIT80 GuitarSet par cinq plis de compositions disjointes pour entraîner et choisir une politique. CAL32 réservé exclusivement au test du modèle déjà sélectionné. La source S708/S707 signée est vérifiée et le flux de notes est mesuré sur la grille officielle 256 échantillons, sans ouvrir HELD ni TEST.

## Exécutions

1. [Run scientifique #39328081673](https://github.com/Andriamarosoa/note/actions/runs/39328081673) : FAIL pendant les tests unitaires ; défaut d'indexation pitch MIDI 40 vers case 0, identifié avant tout réentraînement.
2. Correction `aec47655049d89ddf297348ca1ec89b08c8d1daf` : indexation, vérification des cas de rejouement MIDI, préparation de FIT S790 et cache de prédictions.
3. [Run #39328341071](https://github.com/Andriamarosoa/note/actions/runs/39328341071) : **SUCCESS**, FIT80 + CAL32 complets, six audits indépendants **6/6 PASS** et consensus **SUCCESS**. Artefact `S792_REAL_CAUSAL_MIDI256_SIX_METRICS`, rapports des six auditeurs et consensus conservés.

## Résultat

Le meilleur Frame Pitch FIT isolément vient de la rétention de quatre blocs (80,4416 % contre 78,7278 %), mais ses pertes d'Offset (66,1887 % contre 69,8862 %) et de Full Note (47,9729 % contre 50,6528 %) rendent la stratégie inadmissible. Les durées 1 et 2 blocs sont également inadmissibles. FIT sélectionne donc **0 bloc / aucune intervention**, avant l'ouverture CAL.

Le replay CAL32 reproduit exactement **les six scores S708** (Onset 74,2639 %, Exact-K 62,0455 %, Active Exact-K 56,2762 %, Offset 77,3292 %, Full Note 57,1319 %, Frame Pitch 76,4626 %). **Aucun nouveau score officiel au-delà de 80 %.** Aucun modèle ni pipeline LIVE n'a été promu.

## Audits obligatoires

- Données : 80 FIT, 32 CAL, séparation compositions, SHA S708.
- Acoustique : tête S790 gelée, observations contemporaines ou passées, sans oracle.
- Architecture : événements MIDI réels, NoteOff causaux, zéro NoteOn créé et six voix au plus.
- Apprentissage : quatre politiques et choix sur cinq plis FIT seulement.
- Métriques : recomptage des six métriques S704 et parité signée exacte.
- LIVE : pipeline production inchangé ; validation sur replay causal, pas mesure de latence microphone.

## Statut

**Terminé, six audits PASS, consensus PASS et aucune promotion.** Les prolongations de notes expérimentales ont été rejetées sur FIT pour régression des fins de notes. La référence MIDI reste S708.

## Verdict

**Terminé ; six audits et consensus validés ; S792 rejeté comme amélioration.** Les données montrent qu'améliorer la persistance des hauteurs sans prédiction des attaques et gestion indépendante des fins de notes détériore Offset et Full Note.

Voir `series/S792/resultat.md` pour les scores et les statistiques de modifications MIDI.

## Suite expérimentale

Ne plus prolonger globalement les NoteOff ; auditer les cas de vrais NoteOn absents et les notes dont la sortie est trop courte. Tester des candidats conçus pour préserver les vraies fins de notes, avec une tête/stratégie dédiée aux attaques manquées si nécessaire. À chaque tentative, sélectionner sur FIT et vérifier les six mesures du même flux MIDI causal sur CAL, sans toucher HELD/TEST.
