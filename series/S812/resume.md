---
serie: S812
statut: en_cours
type: infrastructure
reference: S708
branche: codex/note-s812-protect-six-native-h0-axes
run_id: en_attente
commit_sha: en_attente
promotion: false
---
# S812 — conservation du véritable historique H0 dans six chaînes indépendantes

## Objectif

Corriger la faute d'architecture S810 : ne jamais confondre séparation de six décodeurs avec suppression des performances antérieures. Le vieux système validé S708 et sa politique d'offset S707 restent le témoin H0 ; chaque caractéristique récupère une **projection privée et exacte de leurs événements MIDI**, sur l'horloge native 256 échantillons. Les spécialistes nouvellement entraînés S810 restent H1 *shadow*, sans promotion automatique. Zéro lookahead, hold ou réassignation d'attaques/offset.

## Hypothèse

On doit conserver un replay certifié bit-exact des décisions publiées sans reconstruire six anciens réseaux neuronaux inexistants. Six copies séparées de l'état historique natif, une par axe, peuvent être comparées aux nouvelles prédictions privées ; le moteur H0 d'origine reste exécutable et protégé. L'isolement de H1 ne prouve pas son niveau de reconnaissance musicale.

## Modifications

`src/causal_note/s812_historical_axis_h0.py` définit `HistoricalH0Axis` et `SixHistoricalH0`. Six objets indépendants reçoivent uniquement les événements S708/S707 déjà disponibles ; chaque objet garde **son propre ensemble des notes actives et ses propres événements**. Exact-K et Active-K projettent les cardinalités du même flux historique (sans se lire mutuellement), Onset les NoteOn, Offset les NoteOff originaux, Frame Pitch l'occupation et Full Note le flux complet. Chaque note conserve timestamp, décision, disponibilité, raison et champ signé. Aucune approche ne modifie le processus de sélection officiel S708.

`checked_h1_view` isole un essai H1 d'une seule caractéristique ; il ne doit pas modifier H0 ni les autres coordonnées. C'est une interface de shadow, pas un arbitre MIDI faisant passer des logits S810 pour de vrais événements.

**Distinction cruciale :** les six projections H0 sont des vues fidèles du **même** système historique qui génère le MIDI officiel, pas des poids historiques recouvrés pour six nouvelles architectures indépendantes. Ce dispositif garantit la protection des performances officielles mais ne fait pas magiquement de six réseaux existants dans S708.

## Protocole

Tests adversariaux `test/test_s812_historical_axis_h0.py` : protection de la référence, duplications de pitch, notes orphelines, refus de données futures et rétrodatées, invariance de l'ordre d'exécution, copier-isoler les six états, injection hostile dans un axe et invariance stricte des cinq autres, huitième preuve de causalité via des suffixes modifiés. Replay signé S800 : 80 pistes FIT GuitarSet, cinq compositions, 12 compteurs et six ratios exacts MIDI256 recalculés sur la transcription originelle et sur H0. L'évaluation des cardinalités et de Frame Pitch est reproduite depuis les **six sorties individuelles** sur les frames de 256 échantillons, pas seulement depuis le MIDI complet. Les sorties Onset, Offset et Full Note sont comparées événement par événement. Les six tentatives de remplacement H1 par piste sont strictement limitées à l'axe cible et aucune ne change le MIDI officiel.

Interdire ouverture CAL32/HELD/TEST ou optimisation de seuil sur ces corpus. Une seul exécution FIT80 GitHub taguée `[run-fit80]` après les tests smoke.

## Audits obligatoires

Données : vérification S800 SHA256 de chaque JSON/NPZ et de 80 pistes ; Acoustique : disponibilité des événements seulement au temps courant ; Architecture : six objets, états, inputs, sorties distincts ; Apprentissage : H1 non déployé, H0 non remplacé ; Métriques : six scores S704 exacts et six projections comparées, aucun score SHADOW ne remplace l'officiel ; LIVE : 0 confirmation, 0 retiming, matériel iRig/CPU non mesuré. Ce sont des audits locaux/CI, pas les six agents Work ni un consensus NVIDIA 9/9.

## Statut

Sept tests unitaires et 80 traces FIT réelles ont passé un replay **local** des six copies MIDI H0 sans modification ; le contrôle officiel S704 GitHub reste à exécuter avant de clore. S708 reste officielle. Aucune promotion, aucune régression et aucun déploiement LIVE.
