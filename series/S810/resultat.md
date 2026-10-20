---
serie: S810
statut: en_cours
type: experience
reference: S708
branche: codex/note-s810-six-trained-axis-oof
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S810 — mesure du premier entraînement indépendant

## Exécution

Entraînement local réel sur 80 traces signées FIT80 de [S800 #39352880035](https://github.com/Andriamarosoa/note/actions/runs/39352880035). Toutes les compositions disjointes, cinq modèles par axe (30 checkpoints) sauvegardés et chaque fichier accompagné de sa signature SHA256 dans le rapport JSON. Code Python, 30 modèles joblib, log et rapport disponibles dans `S810_six_trained_independent_models_FIT80.zip` (SHA256 `42477cc806caed608944c3733aca76bc9d4a26a82a3b06eb5894dd2a59f9aedc`). Le dépôt GitHub ne dispose PAS encore d'une exécution scientifique CI équivalente : ne pas affirmer un 6/6 PASS de ce seul entraînement.

## Métriques

Ces valeurs sont des évaluations SHADOW spécifiques aux cibles de formation à **1 024 échantillons**. Elles ne sont PAS directement les six métriques S704 sur une transcription MIDI complète.

| Axe entraîné indépendamment | Évaluation FIT80 shadow | Interprétation |
| --- | ---: | --- |
| Exact-K | **76,8042 %** | K0–K6, 58 832/76 600 fenêtres |
| Active Exact-K | **76,5656 %** | K exact sur trames effectivement actives, 47 708/62 310 |
| Onset | **11,3986 % F1** | Marquage binaire par hauteur/trame 1024, pas appariement temporel S704 |
| Offset | **12,7457 % F1** | Marquage binaire de fermeture par hauteur/trame, pas Offset Accuracy conditionnelle S704 |
| Frame Pitch | **78,7230 % F1** | Présence binaire de hauteur à la décision 1024 |
| Full Note | **95,7386 % accuracy d'état** | Quatre classes trame, dominées par silence ; F1 de début de note seulement **14,8887 %**, PAS Full Note F1 |

### Active Exact-K par composition FIT

| Composition exclue de l'entraînement | Active-K shadow |
| --- | ---: |
| BN1 | 79,8911 % |
| Funk2 | 76,2151 % |
| Jazz1 | 76,7075 % |
| Rock3 | 76,5559 % |
| SS1 | 73,9993 % |

Attention : le décodeur SHADOW S809 initial fournissait 42 451/62 310, soit environ 68,13 %, contre 47 708/62 310 en S810. C'est une hausse **locale et spécifique au diagnostic shadow** de +8,44 points, NON un gain officiel CAL32. Inversement Exact-K S809 shadow était 64 525/76 600 (84,24 %), supérieur aux 76,80 % du nouveau modèle : une régression *locale de cet axe* qui empêche toute promotion prématurée.

### Référence officielle protégée

| Caractéristique | S708 officiel CAL32 |
| --- | ---: |
| Onset F1 | 74,2639 % |
| Exact-K | 62,0455 % |
| Active Exact-K | 56,2762 % |
| Offset accuracy | 77,3292 % |
| Full Note F1 | 57,1319 % |
| Frame Pitch F1 | 76,4626 % |

Ces scores ne changent pas. Les six métriques réelles S704 sur un MIDI S810 fusionné sont **NON MESURÉES**.

## Comparaison à S708

Seuls de nouveaux modèles SHADOW ont été entraînés. Aucun événement MIDI natif ne change, aucune latence ajoutée au moteur actuel, pas d'estimation du surcoût CPU/micro d'une intégration future. Les six scores officiels S708 restent les mêmes. Les sorties des six axes sont indépendantes, leur future combinaison MIDI **n'est pas encore autorisée**.

## Audits

| Audit | Statut | Preuve |
| --- | --- | --- |
| Données | PASS local | 80/80 NPZ/JSON signés SHA256, 30 checkpoints sérialisés |
| Acoustique | PASS local | 23 variables S795 causales disponibles sans futur |
| Architecture | PASS local | Six classifieurs/scalers différents par composition, 80/80 tests d'indépendance d'axe |
| Apprentissage | PASS local | Cinq folds par axe ; quatre autres compositions exclusivement |
| Métriques | PASS local | Six diagnostics SHADOW séparés des six ratios MIDI S704 |
| LIVE | UNKNOWN | Rien de changé en production ; coût CPU et latence iRig non mesurés |

Il ne s'agit PAS des six agents Work personnalisés, ni de six audits CI certifiés. Aucun modèle NVIDIA consulté.

## Verdict

**30 checkpoints réellement entraînés et conservés ; architecture à poids indépendants disponible en SHADOW.** Active Exact-K augmente par rapport à l'ancienne estimation shadow S809, mais **Exact-K et de nombreux indicateurs temporels sont insuffisants**, et une intégration MIDI finale n'a pas été évaluée. Zéro candidat promu. Ne pas confondre 95,74 % d'exactitude sur quatre états (principalement silence) avec 95,74 % de Full Note F1.

## Prochaine action

Exécuter les tests CI sur les six classifieurs et leurs signatures, puis intégrer **chacun séparément** à son sélecteur et décodeur privé S809, sans recopier les décisions de ses voisins. Rejouer en MIDI256 réel sur FIT80 avec veto global et par composition sur les six scores avant de songer à CAL32 et au test de latence matérielle LIVE.
