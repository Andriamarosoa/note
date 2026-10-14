# S406–S407 — Améliorer les vraies attaques S80 sans dupliquer le détecteur de frontières ; stratégie Offset

**Date : 13 octobre 2026.** État : **S406 et S407 exécutés et mesurés sur des WAV/JAMS réels GuitarSet** ; aucun modèle S80 de comptage ni S402 d'identification MIDI modifié. **Offset : aucun gain mesuré ici.**

## 1. Pourquoi les anciennes Onsets/Offsets étaient médiocres

La version causale V7 effectuait des décisions binaires très fréquentes (six slots, échantillonnage très dense) et un décodeur à seuil. Sur les **60 pistes historiques V7**, seuil 0,55, tolérance ±50 ms, l'audit `docs/EXPERIMENT_06_UNASSOCIATED_BOUNDARY_CANDIDATES.md` donne :
- **Onset** : précision **7,9662 %**, rappel **92,8205 %**, F1 **14,6732 %** ;
- **Offset** : précision **7,1900 %**, rappel **83,6390 %**, F1 **13,2417 %**.
- Le modèle émet beaucoup trop de candidats ; l'association des frontières conditionnée par un slot déjà occupé masque des vrais onsets (réattaque possible avant l'ancien offset).
- La cible élémentaire de V7 représente des fenêtres de **512 échantillons** et ne résout pas le déséquilibre dans le flux réel : à 0,5, beaucoup de négatifs dépassent le seuil. Voir `docs/EXPERIMENT_04_CAUSAL_OFFSET_WINDOW.md` et `docs/AUDIT_03_TEMPORAL_TARGETS.md`.
- Expérience 09 : filtrer tous les candidats par une période réfractaire 50 ms (par canal et type) fait passer Onset à **18,80 % précision / 31,40 % F1** et Offset à **16,55 % / 28,05 %**, MAIS supprime des vraies frontières, perd le support simultané et a été **rejeté**. Ce n'est pas un décodeur à promouvoir.

**Il ne faut PAS comparer directement ces scores V7 (60 pistes) aux nouveaux S406/S407 (190 pistes) comme si la population, les poids ou les post-traitements étaient identiques.**

## 2. S406 : premier contrôle mesuré du filtrage des attaques par le K de S80

[GitHub Actions #38368600246](https://github.com/Andriamarosoa/note/actions/runs/38368600246), **success**, artefact `11797996849`, code `scripts/s406_score_S80_gate_native_onsets_real_GuitarSet.py`.

59 309 groupes acoustiques natifs, 190 pistes et 34 954 attaques JAMS (dont 1 788 sans rattachement aux groupes natifs selon la règle originale de ±20 ms).

**Protocole** : les prédictions S80 S397 étaient entièrement préexistantes, et les instants start, milieu et end ont été générés avant l'ouverture des annotations. On apparie une attaque prédite à une seule vraie attaque, au maximum une fois, sans utiliser la hauteur (chiffres temporaux seulement). La première ligne brute émet un candidat par groupe ; S80 émet exactement K attaques au même instant choisi pour le groupe. Ces 33 475 propositions respectent le K S80. Tolérances testées ±20/50 ms.

| Temps de prédiction | Tolérance | Politique | Attaques prédites | TP | FP | FN | Précision | Rappel | F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Début groupe | ±20 ms | Candidat brut | 59 309 | 19 171 | 40 138 | 15 783 | 32,32 % | 54,85 % | 40,68 % |
| Début groupe | ±20 ms | **S80 K** | 33 475 | 23 694 | 9 781 | 11 260 | 70,78 % | 67,79 % | 69,25 % |
| Milieu groupe | ±20 ms | **S80 K** | 33 475 | 26 108 | 7 367 | 8 846 | 77,99 % | 74,69 % | 76,31 % |
| Début groupe | ±50 ms | Candidat brut | 59 309 | 26 169 | 33 140 | 8 785 | 44,12 % | 74,87 % | 55,52 % |
| Début groupe | ±50 ms | **S80 K** | 33 475 | 30 493 | 2 982 | 4 461 | 91,09 % | 87,24 % | 89,12 % |

Répartition événementielle : K0 prédits **40 532** ; vrais K0 supprimés **38 432**, vraies attaques manquées lorsque S80 a prédit K0 **2 100 groupes** ; groupes vrais K0 déclenchés par erreur **1 220**. La suppression K0 augmente la précision, et le fait d'émettre K≥2 attaques simultanées (au lieu d'une seule candidate brute par groupe) améliore également le rappel.

**Limite majeure** : les scores ne prouvent pas des hauteurs MIDI correctes, les K attaques d'un groupe partagent la même position, et il n'y a aucun Offset prédit.

## 3. S407 : apprentissage de l'instant d'attaque avec audio S80

[GitHub Actions #38369185178](https://github.com/Andriamarosoa/note/actions/runs/38369185178), **success**, artefact `11798621000`. Source `scripts/s407_song_heldout_predict_onset_timestamp_from_S80_audio.py`.

- Déjà 38 **compositions** de GuitarSet ; quatre folds de validation : composition tenue à l'écart de l'apprentissage du régresseur S407 avec preuve de 0 composition commune pour chaque fold (les différentes prises/joueurs du même morceau ne traversent pas les splits).
- Features **source-only** = K prédit par S80, largeur native du groupe, 18 statistiques de croissance/décroissance temporelle extraites de la trajectoire acoustique `sequence42x49`. L'étiquette JAMS est utilisée pour l'entraînement sur d'**autres morceaux seulement**.
- Régresseur `HistGradientBoostingRegressor` entraîné pour estimer le déplacement en millisecondes par rapport au milieu de la fenêtre du groupe. La quantité d'Onsets S80 n'est **jamais** modifiée. Les nouveaux instants sont bornés à [début−20 ms, fin+20 ms]. Un seul instant partagé par K notes reste une limite du prototype.

| ±20 ms, 190 pistes | TP | FP | Précision | Rappel | F1 |
|---|---:|---:|---:|---:|---:|
| S80 début | 23 694 | 9 781 | 70,78 % | 67,79 % | 69,25 % |
| S80 milieu | 26 108 | 7 367 | 77,99 % | 74,69 % | 76,31 % |
| **S407 song-held** | **29 023** | **4 452** | **86,70 %** | **83,03 %** | **84,83 %** |

À ±50 ms : S80 début F1 **89,12 %**, S80 milieu **89,11 %**, S407 **89,21 %**. Le gain concerne avant tout la localisation fine, et la qualité à 50 ms est presque inchangée. Le traitement n'est pas encore validé sur de nouveaux enregistrements.

**Attention :** S80 S397 (en amont) a déjà été sélectionné avec les erreurs de GuitarSet et ses modèles reconstruits ont des recoupements d'entraînement. De plus, S407 a été développé après consultation des résultats exploratoires de S406 sur ce même corpus. Ainsi, malgré des folds séparés pour entraîner la tête S407, **aucun de ces pourcentages ne constitue une validation complètement indépendante de l'ensemble**. S407 est une **tête expérimentale de positionnement**, pas un nouveau score officiellement promu.

## 4. Recommandations classées par impact attendu et coût

### P0 — Réutiliser S80 pour le nombre de nouvelles attaques

Conserver `K=S80(prediction)` comme **entrée** de la prédiction MIDI et du décodeur d'événements. Si K0, ne générer aucun nouveau NoteOn ; permettre à la tête `frame` et à la présence harmonique de rester fortes car des notes précédentes résonnent. Si K≥1, exactement K attaques possibles par événement. L'usage direct K0 peut également masquer de vraies attaques : **2 100 groupes de vraie classe positive sont prédits K0** sur le diagnostic GuitarSet. Donc, pour de nouveaux audios, comparer durement le garde K0 à un mécanisme de secours *préappris* et calibré hors test.

### P1 — Localisation fine de chaque note, pas seulement du groupe

S407 est une amélioration mesurée à ±20 ms, mais les vrais accords comportent parfois plusieurs attaques à des instants différents dans la même fenêtre de 40 ms. Après S402/S404, prédire un **résidu de temps de début par note identifiée** (voire par corde), entraîné sur les mêmes annotations de notes, et comparer au S407 partagé. Des fenêtres spectrales précises sont déjà extraites ; il n'est pas nécessaire de réimplémenter l'apparition d'énergie `birth` ni la morphologie de S80.

### P2 — Remplacer l'Offset autonome par un Offset conditionné par note active

Un `NoteOff` n'est possible que si une note `(pitch, voice/string?, onset_time)` a été ouverte, y compris durant des événements K0. Évaluer l'activité des notes sur le **flux temporel continu du morceau**, pas seulement autour des événements K>0 ; déduire leur probabilité de libération ou leur résidu de temps à partir du flux, de l'amplitude temporelle et des harmoniques. Ne jamais confondre `end_samples` du **groupe de candidats** avec le véritable `offset` musical d'une note. À K0, une note existante peut parfaitement s'éteindre. Exiger un contrôle des notes longues, notes étouffées, retriggers et cas hors fenêtre avant déploiement.

### P3 — Corriger le déséquilibre des étiquettes temporelles, pas seulement le seuil

Le flux audio V7 possède des attaques extrêmement rares par échantillon. L'ancien sampler surreprésentait les positives et le seuil produisait du bruit même sur des régions sans attaque. Construire un sampler avec zones **attaque / avant-après attaque immédiats / fond lointain**, en utilisant les mécanismes déjà audités dans les expériences 12–13, puis recalibrer sur les morceaux d'entraînement ; garder les cas de 2 ou 3 événements au même instant. Ne pas appliquer la période réfractaire globale 50 ms rejetée, qui efface certains accords simultanés et des Offsets véritables.

### P4 — Évaluer les notes MIDI complètes et contrôler les sources

- Comparer Onset seul (temps) à Onset + hauteur MIDI (temps+pitch) et à Note complet (Onset+Pitch+Offset) avec appariement 1-à-1 et tolérances publiées, dont ±20/±50 ms pour Onset et la tolérance note-off du benchmark.
- Détailler faux nouveaux NoteOn pendant K0, vraie attaque ratée quand S80 K0, erreurs de doublons de voix, décalage médian en ms, Pitch F1, notes poly K2..K6, accord complet et Offset conditionné par hauteur.
- GuitarSet hexaphonique peut superviser séparément les cordes à l'entraînement, sans exiger 6 canaux à l'inférence. Pour généraliser, prévoir morceaux et guitaristes réellement inédits (p. ex. échantillons IDMT/GAPS non exposés, après preuve de provenance).

## 5. Sources méthodologiques externes vérifiées

1. Hawthorne et al., *Onsets and Frames* (ISMIR 2018) : conditionner les états de notes sur des attaques.
   https://research.google/pubs/onsets-and-frames-dual-objective-piano-transcription/
2. Kong et al., *High-resolution Piano Transcription by Regressing Onset and Offset Times* (2021) : affiner chaque instant plutôt que se limiter à la taille du pas temporel.
   https://arxiv.org/abs/2010.01815
3. Wei et al., *Streaming Piano Transcription Based on Consistent Onset and Offset Decoding* (ISMIR 2024) : le décodeur Offset examine les notes qui sont déjà actives.
   https://ismir2024program.ismir.net/poster_18.html
4. Riley et al., *GAPS: A Large and Diverse Classical Guitar Dataset and Benchmark Transcription Model* (ISMIR 2024) : données et généralisation sur guitare, avec prudence concernant les annotations de transcription disponibles.
   https://www.eecs.qmul.ac.uk/~simond/pub/2024/RileyEtAl-ISMIR-2024.pdf

**Décision technique :** conserver S80 pour K et S402 pour l'identité MIDI. Faire de S407 la meilleure **tête expérimentale** de localisation des Onsets à ±20 ms, puis implémenter et évaluer de façon indépendante un vrai Offset **par note active**. Ne pas modifier `freeze_local_combo` ou la référence historique formelle par ce résultat de développement.
