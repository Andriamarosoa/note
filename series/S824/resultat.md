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
# S824 — Résultats des ablations par famille de variables S795

## Exécution

Relecture des 80 traces d'origine S800, sans redécompresser de PCM et sans tourner le modèle S708. Les observations utilisées à MIDI256 sont les snapshots déjà disponibles à 1024 (max 768 échantillons d'âge). Scores authentiques uniquement après run GitHub validé.

## Métriques

| Source FIT80 | Active Exact-K | Commentaire |
|---|---:|---|
| H0 MIDI256 | 57,019538 % | MIDI complet historique, même dénominateur |
| S823 23 variables S795 à 256 | 76,462145 % | SHADOW, dépend S708+MIDI |
| S824 FULL23 contrôle de reproduction | **76,462145 %** | Identique S823 à l'unité, 190705/249411 |
| S824 S708 ONLY | **76,848655 %** | 191669/249411, dépend du modèle S708 |

## Six axes officiels S708 CAL32

| Caractéristique | S708 officiel | S824 |
|---|---:|---|
| Onset F1 | 74,2639 % | inchangé |
| Exact-K | 62,0455 % | inchangé |
| Active Exact-K | 56,2762 % | expériences SHADOW FIT80 séparées |
| Offset accuracy | 77,3292 % | inchangé |
| Full Note F1 | 57,1319 % | inchangé |
| Frame Pitch F1 | 76,4626 % | inchangé |

## Comparaison à S708

Aucun poids S708, aucune émission MIDI source ni moteur LIVE n'est changé. L'ablation ne transforme pas les vieilles features 1024 en une DSP audio 256; les ablations issues d'autres têtes **ne garantissent pas l'indépendance des architectures**.

## Audits

| Rôle | Statut | Motif |
|---|---|---|
| Données | PASS | 80 paires S800 originales, SHA, H0 142213/249411 |
| Acoustique | FAIL | Entrées S795 dépendantes des sorties S708, DSP à 1024, pas de frontend acoustique natif256 indépendant |
| Architecture | FAIL | 9 classifieurs SHADOW privés, mais les familles S708 et MIDI utilisent des caractéristiques d'autres axes |
| Apprentissage | PASS | 48 TRAIN / 16 inner-CAL / 16 OUTER pour chacun des 5 plis ; aucune sélection sur test |
| Métriques | PASS | 11 masques + checkpoint S823 SHA, 249411 GT actives, exact K1–K6 et under/over cohérents |
| LIVE | UNKNOWN | Zéro lookahead/hold ajouté démontré, mais CPU et latence matérielle non mesurés ; LIVE inchangé |

## Verdict

**Terminé, 11 conditions comparées : aucune promotion.** FIT80 est déjà exploré, et les architectures avec S708/MIDI ne sont pas indépendantes. Aucune métrique du MIDI final n'est modifiée.

## Prochaine action

Établir des substituts **audio-only natifs MIDI256**, avec des têtes privées apprenant les indices de fondamentales, de maintien et de cardinalité que S708 apportait, sans réutiliser les postérieurs ni le MIDI émis. Préenregistrer les variantes et les tester sur une source indépendante, sans changer les autres cinq axes ou ajouter de latence.


## Résultat scientifique final — CI SUCCESS

[Run scientifique S824 #39582850126](https://github.com/Andriamarosoa/note/actions/runs/39582850126) **SUCCESS** au commit `70570a4bd32527edf67960ccae0f6b3df84f7703`, [artefact modèle et rapport #12339045461](https://github.com/Andriamarosoa/note/actions/runs/39582850126/artifacts/12339045461). Signatures 80/80 S800, 5 checkpoints S823 d'origine SHA vérifiés, H0 FIT80 **142213/249411**, FULL23 original gelé et réentraîné **190705/249411**, **parité exacte sur chacune des cinq compositions**. Décisions/métriques tous les **256 échantillons**, mais données S795 provenant de snapshots réellement disponibles seulement tous les **1024 échantillons** (76600 snapshots, 240 premiers pas sans observation).

### Ablation exacte : 249411 trames Active Exact-K

| Modèle / entrées conservées | Corrects | Active Exact-K | Sous-comptages | Surcomptages | Δ vs FULL23 |
|---|---:|---:|---:|---:|---:|
| H0 MIDI historique | 142213 | 57,0195 % | 80654 | 26544 | Non applicable |
| FULL23, ancienne architecture S823 | **190705** | **76,4621 %** | 36530 | 22176 | — |
| **S708 ONLY** (14 features) | **191669** | **76,8487 %** | 35948 | 21794 | **+0,3865 pt** |
| WITHOUT ACOUSTIC (S708 + état MIDI) | 191690 | **76,8571 %** | 36052 | 21669 | +0,3949 pt |
| WITHOUT MIDI (S708 + acoustique) | 190768 | 76,4874 % | 36354 | 22289 | +0,0253 pt |
| WITHOUT NATIVE POST (retire colonnes 7–9) | 190974 | 76,5700 % | 36120 | 22317 | +0,1079 pt |
| WITHOUT PRIMARY POST (retire 0–6,10–11,13–14) | 145862 | 58,4826 % | 72716 | 30833 | **−17,9796 pts** |
| **WITHOUT S708** (état MIDI + acoustique) | 136099 | **54,5682 %** | 81280 | 32032 | **−21,8940 pts** |
| MIDI ONLY | 126121 | 50,5675 % | 94909 | 28381 | −25,8946 pts |
| ACOUSTIC ONLY (six canaux DSP + registre) | 108269 | 43,4099 % | 108094 | 33048 | −33,0523 pts |
| Sélection parmi masques via inner-CAL | 191376 | **76,7312 %** | 36171 | 21864 | +0,2690 pt |

Les deux alias déclarés `S708_PLUS_ACOUSTIC=WITHOUT_MIDI` et `MIDI_PLUS_ACOUSTIC=WITHOUT_S708` sont **identiques exactement** et ne représentent pas deux entraînements distincts. Neuf nouvelles variantes sont entraînées, plus le contrôle originel S823 gelé.

### Comparaison par composition et K

Deltas des décisions exactes `S708 ONLY − FULL23` : **BN1 +186**, **Funk2 −47**, **Jazz1 +389**, **Rock3 +120**, **SS1 +316** (total **+964**). Une ablation peut améliorer le score global mais dégrader une composition. Deltas par vérité K : **K1 −110, K2 +751, K3 −31, K4 +211, K5 +294, K6 −151**. Ce modèle ne respecte donc **pas** une exigence de non-régression stricte pour chaque K.

Inversement, `WITHOUT S708 − FULL23` perd **9018/10164/10241/10769/14414** bonnes décisions respectivement pour BN1/Funk2/Jazz1/Rock3/SS1, soit **−54606 décisions**, et son écart de −21,89 points est manifeste dans les cinq compositions. Les **postérieurs S708 primaire (onset/sustain/release + variations + contexte harmonique + K)** sont déterminants : leur retrait entraîne **−17,98 points**. Retirer seulement le native_post 7–9 ne provoque pas la même chute.

### Interprétation et limites

L'effet de suppression de familles est mesuré à protocole constant ; il établit la forte **dépendance prédictive** du classifieur historique aux sorties du S708. Il n'est pas une preuve de causalité physique ni de validation externe : les sorties S708 peuvent concentrer des informations acoustiques préalablement apprises, et la série FIT80 a été largement explorée. Les progrès de +0,39 point ne sont pas une nouvelle référence : le score S823 déjà retenu a atteint **76,8775 % avec une autre politique**, plus que 76,8571 %, mais reste lui-même non promouvable.

L'ablation ACOUSTIC ONLY est limitée aux six canaux `native_features` choisis parmi les 126 du frontend ancien, plus la coordonnée registre; son faible score **n'invalide pas les architectures acoustiques apprises modernes**. Les têtes S824 demeurent alimentées par des observations S795 **à 1024 échantillons**, même si les prédictions sont émises à 256. Aucun de ces modèles n'est prêt pour le LIVE strict indépendant.

**VERDICT : audit S824 terminé, pas de promotion, aucune unification et aucun LIVE modifié.**