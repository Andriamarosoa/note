---
serie: S824
statut: termine
type: experience
reference: S708
branche: codex/note-s824-s795-feature-family-ablation
run_id: 39582850126
commit_sha: 70570a4bd32527edf67960ccae0f6b3df84f7703
promotion: false
---
# S824 — Ablations causales des familles de 23 variables historiques S795/S810

## Objectif

Mesurer sur **les mêmes 249411 trames GT actives MIDI256**, en conservant les **vraies observations S795 à 1024 échantillons**, quelles familles de variables permettent à S823 RAW23 de réussir **190705/249411 = 76,462145 %**. Retirer séparément les sorties S708, les états MIDI, les indices audio et les sous-familles de postérieurs. Ne pas déclarer une cause acoustique à partir d'un score descriptif.

## Hypothèse

Le record S823 peut dépendre davantage des décisions intermédiaires S708/MIDI que des 7 coordonnées acoustiques indépendantes (6 audio + registre). **Falsification** : entraîner exactement le même algorithme, sur les mêmes lignes TRAIN et 5 partitions composition-excluses, en masquant les familles prévues AVANT l'évaluation; prendre les différences de scores sur les OUTER 80 pistes uniquement comme observations historiques.

## Modifications

`src/causal_note/s824_feature_family_ablation.py` définit l'ABI et des masques prédéclarés, tous à **23 coordonnées avec zéros pour les colonnes exclues**, ce qui isole la suppression d'information sans modifier le nombre de coefficients possible, l'ordre des features ni la valeur des autres features. `scripts/s824_retrain_feature_families.py` rejoue le code S823, y compris les 80 empreintes SHA S800, reproduit `full23` à l'unité près (attendu 190705 corrects) puis entraîne **des modèles distincts, aux poids privés et recalibrés sur TRAIN** pour les huit autres masques uniques ; le checkpoint d'origine S823 est restauré séparément, vérifié SHA, comme témoin immuable. 5 plis 48 TRAIN/16 inner-CAL/16 outer. Le choix éventuel d'une variante par pli utilise seulement inner-CAL (jamais les résultats outer); tous les résultats en mode fixe sont publiés en entier.

## Protocole

Canaux S795 originaux documentés par `causal_features` :
- **S708-model** = 0..11, 13,14 (14 colonnes): onset/sustain/release et deltas, native_post, conflits harmoniques calculés à partir des postérieurs, K prédit et entropie.
- **MIDI émis** = 12,15 (2 colonnes): carte des notes actuellement émises et nombre de notes.
- **Acoustique et registre** = 16..22 (7 colonnes): six colonnes native_features S708 plus indice pitch normalisé. Cela reste la DSP du S708 à fréquence 1024, **pas** une reconstruction PCM native 256.
- Ablations : `full23`, `without_midi`, `without_s708`, `acoustic_only`, `without_acoustic`, `s708_only`, `midi_only`, `s708_plus_acoustic`, `midi_plus_acoustic`, `without_native_post` (retire seulement 7..9), `without_primary_post` (retire 0..6 +10..11 +13..14). 11 variantes définies avant test.

Même `StandardScaler→LogisticRegression(C=0.5)`, mêmes 840 échantillons au maximum par piste et mêmes poids par classe. Une variante ne recevra jamais les entrées retirées, directement ou via l'agrégation; le modèle source S708 et toutes ses émissions MIDI sont figés.

## Audits obligatoires

Identité RAW23 S823 (190705/249411; scores par composition), SHA des 80 originaux, 5×(48/16/16), mêmes index as-of 1024→256 et sous/surcomptages. Causalité anti-futur; tester qu'une perturbation d'une famille retirée ne change jamais les features ni les résultats. Les canaux issus S708 ou MIDI restent des dépendances inter-axes, donc toute variante concernée reste SHADOW non promouvable. Aucun LIVE, S708, Onset, Offset, Frame Pitch, Full Note ou Exact-K n'est modifié. FIT80 a déjà été exploré, donc aucune revendication de validation indépendante.

## Statut

Audit GitHub [#39582850126](https://github.com/Andriamarosoa/note/actions/runs/39582850126) terminé avec succès, 80 sources signées et contrôle S823 original **190705/249411** reproduit. Les ablations confirment la forte dépendance aux 14 sorties du S708, score WITHOUT_S708 54,5682% contre FULL23 76,4621%; S708 ONLY 76,8487%. Aucune tête indépendante promue, aucun LIVE modifié.


## Preuves finales, audit de reproductibilité et interprétation

**Origine** : [run réel S824 #39582850126](https://github.com/Andriamarosoa/note/actions/runs/39582850126) SUCCESS, [artefact #12339045461](https://github.com/Andriamarosoa/note/actions/runs/39582850126/artifacts/12339045461). Les cinq checkpoints S823 historiques ont été importés uniquement comme contrôles, sans changer aucun enseignant original. Les neuf classifieurs de contrôle/ablation ont été entraînés avec la même source et la même recette; le modèle FULL23 réentraîné reproduit cette fois exactement l'original S823 à **190705/249411**, identique par composition. Le test S800 source signé conserve **306536** trames totales et **249411** GT actives, 5 plis 48/16/16 et 80 pistes. 

**Résultat** : 14 sorties S708 seules donnent **191669/249411 = 76,8487 %** ; enlever toutes les sorties S708 donne seulement **136099/249411 = 54,5682 %** (**−21,894 points** vs FULL23). Les features acoustiques S795 seules produisent **108269/249411 = 43,4099 %**; les deux coordonnées MIDI seules produisent **126121/249411 = 50,5675 %**. S708 + MIDI, sans les 7 features acoustiques directes, donne **191690/249411 = 76,8571 %**. La sélection inner-CAL des 9 variantes donne **191376/249411 = 76,7312 %**. Toutes les données sont FIT80 déjà explorées; le meilleur score diagnostique historique S823 CAL-selected reste **76,8775 %**.

**Précision** : les 14 sorties S708 agrègent un travail acoustique déjà accompli par un autre enseignant. Les retirer réduit fortement la prédiction, ce qui ne démontre pas que les données spectrales ou le MIDI sont inutiles dans une autre architecture. Même la condition ACOUSTIC ONLY utilise la sélection limitée 6 features natives ancien S708 et un index de registre; elle ne mesure pas tous les potentiels du frontend PCM256. En particulier, les snapshots d'origine sont à 1024, sans capacité de répondre aux nouvelles attaques chaque 256. L'exigence d'architecture Active-K **strictement indépendante** n'est pas satisfaite par ces modèles historiques, indépendamment de leur score.

**Veto** : aucun modèle n'est promu ni unifié ; S708 officiel et ses six métriques CAL32, le moteur LIVE causal et les 256 échantillons de décision sont inchangés. Prochaine recherche sur une tête sonore privée qui apprenne directement les preuves de la fondamentale et du nombre de notes à partir du PCM256 courant ou passé, avant toute validation matérielle et extérieure.
