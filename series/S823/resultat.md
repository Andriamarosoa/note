---
serie: S823
statut: termine
type: experience
reference: S708
branche: codex/note-s823-legacy-s795-features-causal256
run_id: 39578183788
commit_sha: 56094b6f8353a26b998faa29e3b3f043b081b611
promotion: false
---
# S823 — Vraies caractéristiques S795 reportées causalement à MIDI256

## Exécution

Replay diagnostique de 80 snapshots OOF S800 signés, avec leur cadence d'observation S795 1024, puis décisions et métriques MIDI256. Un tel replay ne constitue PAS une ré-extraction acoustique 256.

## Métriques

| Active Exact-K | Score FIT80 |
|---|---:|
| S810 1024 SHADOW, 62310 frames actives | 76,5656 % (protocole différent) |
| H0 MIDI256, 249411 frames actives | 57,019538 % |
| S822 RAW23 PCM256 indépendant | 54,454695 % |
| S822 AGG61 PCM256 indépendant | 54,385733 % |
| S823 vrais S795 RAW23 reportés 256 | 76,462145 % |
| S823 vrais S795 AGG61 reportés 256 | 75,612944 % |
| S823 CAL-selected | 76,877523 % |

## Six caractéristiques S708 CAL32

| Axe | S708 officiel |
|---|---:|
| Onset F1 | 74,2639 % |
| Exact-K | 62,0455 % |
| Active Exact-K | 56,2762 % |
| Offset accuracy | 77,3292 % |
| Full Note F1 | 57,1319 % |
| Frame Pitch F1 | 76,4626 % |

## Comparaison à S708

Aucune modification S708, MIDI final ou LIVE. Inputs S795 contiennent S708 postérieurs, K et notes émises : **dépendance inter-axes**. L'observation n'est actualisée qu'à 1024 même si chaque décision est rendue à 256.

## Audits

| Rôle | Statut | Critère |
|---|---|---|
| Données | PASS | 80 paires SHA S800, H0 142213 reproduit |
| Acoustique | FAIL | Données audio mélangées aux prédictions d'axes S708 et au MIDI émis, non indépendantes |
| Architecture | FAIL | Les entrées reçoivent des postérieurs S708 et l'occupation des notes, donc non indépendantes |
| Apprentissage | PASS | Cinq plis 48 TRAIN/16 inner-CAL/16 OUTER, sélection CAL |
| Métriques | PASS | 249411 trames GT actives, identité original H0, trois sorties recalculées |
| LIVE | UNKNOWN | Non intégré, pas de mesure micro; source réactualisée chaque 1024 |

## Verdict

Non promouvable malgré la précision diagnostique obtenue : sorties provenant des autres axes, non intégrées en MIDI final.

## Prochaine action

Comparer aux 23/61 variables S822 natives, identifier l'information additionnelle contenue dans les sorties S708 et tracer la voie d'un modèle indépendant qui ne reçoit que le son présent ou passé.


## Bilan réel vérifié — 21 octobre 2026

La preuve scientifique est le [run #39578183788](https://github.com/Andriamarosoa/note/actions/runs/39578183788) **SUCCESS**, artefact [#12335556216](https://github.com/Andriamarosoa/note/actions/runs/39578183788/artifacts/12335556216). Les 80 paires NPZ/JSON S800 originales ont été vérifiées SHA, avec 76600 observations historiques de 1024 samples, restituées via last-available uniquement; 240 premiers instants 256 échantillons (3 par morceau) n'ont pas d'observation 1024 et ont reçu une valeur neutre. Aucune trame future, aucun hold ajouté. Cinq plis 48 TRAIN/16 inner-CAL/16 OUTER, H0 reproduit: **142213/249411** décisions Active-K correctes, **306536** trames MIDI256 au total.

### Scores FIT80 à 256, 249411 trames actives

| Architecture | Correct / 249411 | Exact-K actif | Sous-comptage | Surcomptage |
|---|---:|---:|---:|---:|
| H0 MIDI émis officiel FIT80 | 142213 | 57,019538 % | 80654 | 26544 |
| S822 vraie DSP indépendante RAW23 | 135816 | 54,454695 % | 77833 | 35762 |
| S822 vraie DSP indépendante AGG61 | 135644 | 54,385733 % | 76568 | 37199 |
| **S823 véritables canaux S795 RAW23** | **190705** | **76,462145 %** | **36530** | **22176** |
| S823 véritables canaux S795 AGG61 | 188587 | 75,612944 % | 37073 | 23751 |
| **S823 sélection inner-CAL** | **191741** | **76,877523 %** | **34611** | **23059** |

Le dernier modèle obtient +49528 décisions correctes par rapport au H0 sur le **même dénominateur de comptage 256**, mais cette sortie est une tête SHADOW indépendante de la reconstruction d'événements MIDI : les autres cinq métriques du MIDI final ne sont pas réévaluées. La différence avec S810 76,5656 % à 62310 trames n'est pas un gain directement comparable. Aucun >85 %.

### Détail par composition : exactitude RAW23 / AGG61

| Composition | RAW23 | AGG61 | Choix inner-CAL |
|---|---:|---:|---|
| BN1 | 38601/47770 = 80,8059 % | 37719/47770 = 78,9596 % | RAW23 |
| Funk2 | 36567/48827 = 74,8909 % | 35927/48827 = 73,5802 % | RAW23 |
| Jazz1 | 31449/40548 = 77,5599 % | 30657/40548 = 75,6067 % | AGG61 |
| Rock3 | 38953/54917 = 70,9307 % | 40781/54917 = 74,2593 % | AGG61 |
| SS1 | 45135/57349 = 78,7023 % | 43503/57349 = 75,8566 % | RAW23 |

La politique inner-CAL choisit donc AGG61 pour Rock3 (gain par rapport à RAW23) mais aussi pour Jazz1 (régression locale). Aucun choix n'est refait a posteriori à partir de l'outer fold.

### Interprétation et causalité

**Le vrai domaine d'entrée de S810 est identifié :** les colonnes S795 0–6 décrivent les probabilités S708 onset/sustain/release et leurs différences temporelles ; 7–9 ses postérieurs natifs ; 10–11 les conflits harmoniques fondés sur ces postérieurs ; 12 contient l'occupation des notes **déjà émises** ; 13 la cardinalité prédite S708 ; 14 son entropie ; 15 le nombre de notes émises ; 16–21 les coordonnées acoustiques sélectionnées ; 22 la position de hauteur. Donc au moins seize variables transportent des décisions S708, et seulement six plus le registre décrivent directement l'audio.

Ce résultat **montre que le retour à la vraie représentation S795/S810 coïncide avec la récupération de performances ~76 %**, même lorsque la décision/évaluation est assurée toutes les 256 valeurs. Il **ne sépare pas** l'effet de ses différentes familles de features ; pour attribuer rigoureusement le gain à S708 et au MIDI, une **ablation avec réentraînement** et entrées identiques sauf les familles retirées est nécessaire.

La donnée source S795 reste réellement actualisée toutes les **1024 valeurs**, ce qui diffère d'une extraction native 256. Une décision toutes les 256 valeurs basée sur la dernière feature vieille de 0..768 valeurs est techniquement causale, mais ne garantit ni découverte instantanée des nouvelles notes, ni latence physique faible du modèle source. Les sorties du S708 et de son MIDI **violent l'indépendance du nouvel enseignant Active-K** recherchée par le projet. On ne peut donc pas promouvoir S823, y compris si son score atteignait >85 %.

## Conclusion

**S823 TERMINE — score diagnostic 76,8775 % à métrique MIDI256, PROMOTION REFUSÉE**. Les têtes utilisent d'autres axes via S795, aucune intégration MIDI commune, aucun score de Full Note/Onset/Offset vérifié, pas de test de microphone physique. Garder S708 officiel, ne pas modifier les six axes ni le LIVE. Prochaine ablation : retirer d'abord le MIDI émis et les postérieurs S708 individuellement, mesurer la contribution sans données nouvelles, puis chercher une architecture d'acoustique native 256 capable de retrouver ces signaux sans dépendances.
