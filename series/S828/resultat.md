---
serie: S828
statut: termine
type: experience
reference: S708
branche: codex/note-s828-activek-s708-private-readout-256
run_id: 39604850899
commit_sha: bec403685379a581554f4d6d2d1e6629370e477f
promotion: false
---
# S828 — S708 privé natif256, Active Exact-K sur GuitarSet FIT80 complet

## Exécution

**Résultat réel terminé et vérifié** — [GitHub Actions #39604850899](https://github.com/Andriamarosoa/note/actions/runs/39604850899), SUCCESS (les deux jobs `strict_isolation` et `signed_real_five`). Artefact signé `S828_FULL_FIT80_PRIVATE_S708_256_COMPOSITION_CALIBRATED_ACTIVEK_ONLY`, ID **12349523181**.

La procédure a traité **80/80 pistes GuitarSet originales, 5 compositions BN1/Funk2/Jazz1/Rock3/SS1, 249 411 trames de vérité active**, avec la grille **PCM, extraction S708 288D, inférence S708 et décision privées toutes les 256 valeurs**, sans prédiction différée de la grille 1024. L'audio est identifié par la somme SHA source originale ; événements S800 et référence H0 reproduits avec exactement **142 213/249 411**.

Les cinq plis évaluent chacun **48 enregistrements TRAIN, 16 inner-CAL, 16 outer-TEST** ; le sélecteur de la nouvelle tête privée utilise les labels CAL uniquement. À ne pas confondre avec une validation externe : la cohorte FIT80 a été historiquement explorée, et la provenance du pré-entraînement du checkpoint S708 gelé ne certifie pas une exclusion complète du pli TEST. Ce protocole confirme le split du **nouveau readout K**, pas le caractère entièrement indépendant du réseau S708 d'origine.

## Métriques

### Scores Active Exact-K comparables sur exactement les mêmes 80 pistes

| Architecture | Bons comptages sur 249 411 GT-actives | Active Exact-K | Sous-comptages | Surcomptages |
|---|---:|---:|---:|---:|
| H0 S708/S800 émis original | 142 213 | 57,019538 % | 80 654 | 26 544 |
| S827 privé — S708 original recalculé à 256, sans nouveau readout | 152 555 | 61,166107 % | 74 680 | 22 176 |
| **S828 privé — S708 gelé + readout K entraîné à 256** | **184 143** | **73,831146 %** | **41 647** | **23 621** |
| **S823 inner-CAL sélectionné — source S795 as-of 1024** | **191 741** | **76,877523 %** | **34 611** | **23 059** |

**Écart exact sur le même benchmark : S828 − S823 = −7 598 décisions et −3,046377 points.** S828 réalise +31 588 décisions correctes, soit +12,665039 points, par rapport à son témoin privé S827 à grille 256. La différence restante vs S823 correspond, en agrégat, à **+7 036 sous-comptages et +562 surcomptages**. Il ne s'agit pas d'une comparaison entre 5 et 80 pistes.

### Comparaison par composition FIT80 — mêmes dénominateurs

| Composition | Trames actives | S823 correct | S823 score | S828 correct | S828 score | Écart S828 − S823 |
|---|---:|---:|---:|---:|---:|---:|
| BN1 | 47 770 | 38 601 | 80,8059 % | 36 540 | 76,4915 % | −4,3144 pts |
| Funk2 | 48 827 | 36 567 | 74,8909 % | 34 871 | 71,4175 % | −3,4735 pts |
| Jazz1 | 40 548 | 30 657 | 75,6067 % | 29 889 | 73,7126 % | −1,8941 pts |
| Rock3 | 54 917 | 40 781 | 74,2593 % | 40 529 | 73,8005 % | −0,4589 pt |
| SS1 | 57 349 | 45 135 | 78,7023 % | 42 314 | 73,7833 % | −4,9190 pts |
| **FIT80** | **249 411** | **191 741** | **76,877523 %** | **184 143** | **73,831146 %** | **−3,0464 pts** |

Le déficit se concentre notamment sur **SS1 (−2 821 comptages corrects)**, **BN1 (−2 061)** et **Funk2 (−1 696)**. Les totaux `K1..K6` individuels sont enregistrés dans les 80 entrées `by_true_K` de l'artefact S828. Sans analyser ces entrées, aucune cause fine (harmoniques, sustain, attaque, etc.) ne doit être affirmée.

### Les cinq autres caractéristiques — **référence officielle inchangée**, autre cohorte CAL32

| Axe | S708 officiel CAL32 | Impact S828 sur code et MIDI signé |
|---|---:|---|
| Onset F1 | 74,2639 % | Gelé, invariant |
| Exact-K | 62,0455 % | Gelé, invariant |
| **Active Exact-K** | **56,2762 %** | **Candidat S828 FIT80 privé : 73,831146 %, NON PROMU** |
| Offset accuracy | 77,3292 % | Gelé, invariant |
| Full Note F1 | 57,1319 % | Gelé, invariant |
| Frame Pitch F1 | 76,4626 % | Gelé, invariant |

Ne pas comparer directement `S708 CAL32 56,2762 %` avec `S828 FIT80 73,831146 %` pour revendiquer une amélioration officielle : **cohortes et sorties différentes**. Une sortie Active-K SHADOW ne garantit pas le même score après fusion à un MIDI partagé. Les cinq autres axes ne changent pas parce que la sortie privée n'est **pas** intégrée au flux MIDI de production.

## Comparaison à S708

Le modèle préexistant S708 utilisé par S828 est réellement exécuté à chaque 256 mais ses **poids neuronaux temporels/harmoniques restent gelés**. Seul le `HistGradientBoostingClassifier` privé K0..K6 est entraîné à cette cadence ; ses entrées sont exclusivement des acoustiques S708 fraîchement recalculées et un état Active-K privé. Il n'utilise pas les sorties H0 des cinq autres axes ni le MIDI signé pendant l'inférence.

S823 reste devant en score (76,877523 %) mais ses 23/61 variables S795 sont actualisées à 1024 et contiennent des postérieurs S708 et des états MIDI historiques. **S823 n'est donc pas l'équivalent d'un modèle entièrement recalculé à 256 et indépendant des autres axes**. Le gain pratique de S828 est l'isolation et la cadence réelle, sans avoir encore restauré S823 ou franchi 85 %.

## Audits

| Domaine | Statut | Preuves et limites |
|---|---|---|
| Données | PASS | 80 WAV GuitarSet signés, cinq compositions, H0 FIT80 142 213/249 411 identique |
| Acoustique | PASS | Recalcul des 288D et inférence S708 sur chaque nouveau hop PCM256, pas de features S795 as-of1024 |
| Architecture | PASS | Diff limité à S828; les cinq axes S708/S812/S827 sont inchangés et aucune sortie MIDI H0 n'est modifiée |
| Apprentissage | PASS | Nouveau readout K entraîné à 256 sur 5 plis compositional TRAIN/CAL/OUTER disjoints; réseau S708 gelé, provenance des données de son entraînement amont non certifiée OOF |
| Métriques | PASS | 80/80; Active-K S828 184 143/249 411, S827 152 555, S823 191 741; sous/surcomptages et compositions comparés |
| LIVE | UNKNOWN | Aucun branchement production, aucune mesure iRig/CPU p99, pas de vérification matérielle d'un budget strict de 5,80 ms |

## Verdict

**S828 terminé, NON PROMU.** La comparaison sur **80 pistes identiques** prouve un résultat **73,831146 %** (et non le **44,41 %** du pilote 5 pistes). Résultat inférieur de **3,046377 points** à S823, mais réellement natif256 et isolé. L'objectif **Active Exact-K >85 %** n'est pas atteint. S708 reste la référence officielle, les cinq axes voisins et la production MIDI demeurent inchangés. **La réussite CI n'autorise ni promotion ni merge.**

## Prochaine action

Étudier l'erreur de sous-comptage (41 647 décisions) en commençant par **SS1/BN1/Funk2**, puis les `by_true_K` de l'artefact 80 pistes; tester des corrections **uniquement dans la tête et le sélecteur privés Active-K**, préenregistrer les variantes, sélectionner sur CAL interne et comparer **sur les 80 mêmes pistes**. Ne jamais annoncer de progrès à partir d'un pilote 5 pistes. Conserver S823 comme témoin expérimental 76,877523 %, et la référence S708 CAL32 officielle sans mutation.

Source primaire S823 : [GitHub Actions #39578183788](https://github.com/Andriamarosoa/note/actions/runs/39578183788), S828 : [GitHub Actions #39604850899](https://github.com/Andriamarosoa/note/actions/runs/39604850899).
