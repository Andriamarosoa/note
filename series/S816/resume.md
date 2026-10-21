---
serie: S816
statut: termine
type: experience
reference: S708
branche: codex/note-s816-activek-risk-selector
run_id: 39539306030
commit_sha: a86bb13b68c986672e58900021f4b825f36c0baa
promotion: false
---
# S816 — Sélection Active-K avec risque de surcomptage

## Objectif

Tester une correction de sélection privée d'Active Exact-K sans réentraîner S815 ni modifier les cinq autres axes, la référence S708/S707 ou le LIVE. Le véritable but reste >85 % sur six enseignants distincts validés sur une cohorte commune et une émission MIDI256 sans latence ajoutée.

## Hypothèse

Sur les mêmes 249411 trames actives GuitarSet FIT80 : H0 correct 142213 (57,019538 %), S814 136360 (54,672809 %), S815 141946 (56,912486 %). S815 réduit les sous-comptages de 80654 à 65530, mais augmente les surcomptages de 26544 à 41935. Une règle de sélection privée pénalisant les surcomptages sur les compositions de calibration pourrait équilibrer le compromis sans changer les têtes.

## Modifications

Cinq modèles S815 préalablement entraînés restent fixés. Pour chaque composition laissée dehors, le choix du sélecteur et du décalage reste limité aux 15 politiques S815 prédéclarées sur sa composition de calibration interne. Score de sélection = précision Active-K de calibration − 0,50 × fréquence de surcomptage sur ses trames actives, avec perte de précision interne permise d'au plus 2 points vs le maximum de cette calibration. Aucun réentraînement, aucune sélection sur les étiquettes du pli évalué.

## Protocole

Conserver FIT80 GuitarSet authentique, 80 pistes, cinq plis (48 entraînement passés / 16 calibration interne / 16 évaluation par composition). Ré-extraire le PCM signé au pas de 256 samples avec les mêmes 222 observations et comparer les SHA des WAV, annotations et maniféstes stables ; les durées de calcul ne sont pas des identifiants. Exiger parité exacte des 80 prédictions S815 gelées et du MIDI historique S708 avant toute publication. Enregistrer les décisions réévaluées et les résultats par vrai K1–K6. FIT80 ayant déjà été exploré, ce test reste exclusivement exploratoire.

## Audits obligatoires

Tests de validité des 15 décisions, règle anti-surcomptage fixée, absence d'oracle utilisant les étiquettes externes, contre-tests de provenance, causale PCM256 et référence S704 immuable. En cas de résultat inférieur à H0, rejet intégral. Aucune revendication de généralisation, de latence matérielle ni de consensus NVIDIA sans preuve.

## Statut

Protocole et script préparés, score S816 non encore connu. Aucun modèle musical promu ni unification permise. Comparaison rigoureuse S815/H0 requise.


## Clôture expérimentale signée

[GitHub Actions S816 #39539306030](https://github.com/Andriamarosoa/note/actions/runs/39539306030) SUCCESS : 80 pistes, métrique MIDI256, vrai flux H0 142213/249411 répliqué et S815 141946/249411 reconstruit à l'identique depuis les checkpoints immuables. La nouvelle règle anti-surcomptage obtient 140728/249411, soit **56,42413526 %** (contre S815 56,91248582 % et H0 57,01953803 %). Baisse de 0,48835055 point vs S815 et 0,59540277 point vs H0. Seul le pli Rock3 a changé de politique, confidence_direct_pitch→median ; écart Rock3 de 58,10040607 % à 55,88251361 %. Pas d'amélioration ni promotion, aucun changement du LIVE. Les données FIT80 restant exploratoires, aucune généralisation indépendante n'est revendiquée.
