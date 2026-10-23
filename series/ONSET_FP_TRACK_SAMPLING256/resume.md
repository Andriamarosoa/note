# Onset PCM256 — boucles indépendantes H_FP_TRACK_SAMPLING_03 (#144)

**Deux contre-expériences neuronales réelles, aucune autre caractéristique ni MIDI officiel modifié.**

## Hypothèse enregistrée avant les nouvelles expériences

Sur GuitarSet FIT80, l'audit signé des 12465 attaques acceptées par ErrorPrediction69 montrait 8054 TP et 4411 FP, mais une concentration très inégale d'exemples FP entre les pistes (4 à 262 FP par piste, par exemple). Nous soupçonnons — SANS le tenir pour prouvé — que le modèle FP neuronal `control69` a été ajusté principalement aux pistes qui produisent le plus d'événements, compromettant le transfert à d'autres textures/compositions.

Référence signée Onset native256 80 WAV 306536 décisions : meilleur FP69 contrôle du run [#39775357133](https://github.com/Andriamarosoa/note/actions/runs/39775357133) **TP8127 FP4519 FN3140 F1=67.9713963%** ; quatre anciennes têtes `MainAttack S613 + RescueAttack S701 + ErrorPrediction69 MLP + FNRecovery` **TP8160 FP4605 FN3107 F1=67.9094541%**.

[Hypothèse préenregistrée et verdict à publier : GitHub issue #144](https://github.com/Andriamarosoa/note/issues/144). Nouvelle règle d'audit [#140](https://github.com/Andriamarosoa/note/issues/140).

## Deux boucles distinctes

Un workflow matrice exécute **deux branches expérimentales d'entraînement indépendantes** sous la même branche de code, dans deux jobs GitHub séparés, pour respecter l'isolement des états scientifiques :

1. `track_uniform` : tirer avec remise **exactement N_train exemples** avec une probabilité inverse du nombre d'événements de leur piste, pour équilibrer les 48 pistes TRAIN. Distribution TP/FP naturelle préservée autant que possible ; seuls les poids par piste changent.
2. `track_class` : tirer **exactement N_train exemples** en équilibrant moitié TP et moitié FP, puis tirer de façon uniforme les pistes à l'intérieur de chaque classe. La pression anti-FP est volontairement plus forte ; il faut prouver que TP ne se dégrade pas.

**Architecture et protocole invariants** : le `control69` est réellement ré-entraîné sous le **même modèle `StandardScaler + MLPClassifier(64,32)`** utilisé dans le run précédent, mêmes paramètres alpha=.012, batch_size=256, early_stopping et seeds, entrée **69D pré-decision** nativement PCM256, même TRAIN composé de trois compositions, même quatrième composition CAL pour seuil, cinquième OUTER strictement exclue de tout choix. Le nombre de lignes d'apprentissage est IDENTIQUE, la distribution seule diffère.

Les modèles 70D et 74D de contrôle sont chargés et recopiés **bit-à-bit avec vérification SHA256** des poids signés, sans jamais les réentraîner. S613/S701/MLP69 ErrorPrediction/FNRecovery restent figés ; cinq autres axes (Exact-K, Active-K, Frame Pitch, Offset, Full Note) et MIDI officiel intacts. Au replay chaque tête prédit avant le sélecteur et chaque décision consomme **256 nouveaux échantillons à 44.1kHz**, aucun lookahead/hold/wait. Aucune des fenêtres 2048/4096/8192 du frontal causal historique n'est remplacée par de l'audio futur.

La sélection du seuil `control69` réentraîné se fait sur **CAL uniquement** selon exactement les anciennes règles de proxy (F1 sur événements annotés, FP évités, TP perdus avec garde 2.5%) ; **le proxy n'est jamais le F1 scientifique du modèle**. Le replay des 80 véritables WAV GuitarSet recalcule TP/FP/FN pour chaque nouvelle variante. Les résultats de CAL et les poids figés sont enregistrés AVANT l'ouverture de OUTER.

## Critères de réfutation, PAS de tuning opportuniste

Pour chaque `mode` indépendamment :
- **F1 réel strictement >67.9713963%** ET **FP réel strictement <4519** sur le même 80 WAV GuitarSet / 306536 décisions PCM256 ;
- quatre anciennes têtes exactement inchangées TP8160 FP4605 FN3107, poids 70D/74D inchangés avec scores réels prouvés ;
- publication des comptes TP/FP/FN/F1 par composition et par piste, régressions incluses ;
- aucune promotion du modèle, ni déploiement MIDI sur la base de la seule métrique FIT80. Tests externes sur compositions réellement nouvelles toujours nécessaires.

Si un mode ne satisfait pas ces critères, considérer **H_FP_TRACK_SAMPLING_03 REFUTED_OR_INCOMPLETE pour ce mode**, conserver preuves et rouvrir audit, pas de seuil ajusté après OUTER. Si les deux échouent, ce mécanisme de déséquilibre par piste n'est pas une solution démontrée.

## Livrables

- `scripts/train_onset_fp_track_sampling256.py` : deux politiques d'échantillonnage TRAIN, mêmes réseaux et CAL.
- `scripts/replay_onset_fp_track_sampling256.py` : vérité du replay WAV causal, garde scientifique comparative FP/F1.
- `tests/test_onset_fp_track_sampling256.py` : expérience déterministe, taille de TRAIN conservée, 48 pistes TRAIN, aucune OUTER dans les tirages, poids MLP réellement entraînés.
- `.github/workflows/onset-fp-track-sampling256.yml` : jobs matrice `track_uniform` et `track_class`, vérification SHA, archive de toutes les métriques y compris les échecs.
- `series/ONSET_FP_TRACK_SAMPLING256/resume.md` / `resultat.md` : audit et score signé.

**Attention : les poids historiques S613/S701 ont déjà vu une partie de FIT80 dans leur entraînement antérieur.** Même un gain ne vaut pas une validation externe complète. La cadence 5.805ms est celle des 256 échantillons, **pas** une mesure de la latence p95 end-to-end. Aucun résultat scientifique nouveau annoncé avant la fin des deux jobs.
