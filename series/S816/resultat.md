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
# S816 — Résultat du sélecteur privé anti-surcomptage

## Exécution

Expérience de resélection sans réentraînement sur les poids immuables S815 et les WAV GuitarSet mono 44,1 kHz signés. [GitHub Actions #39539306030](https://github.com/Andriamarosoa/note/actions/runs/39539306030) terminé SUCCESS : checkpoints gelés, 80 sorties historiques et 80 sorties S815 reproduites, 80 décisions privées S816 sauvegardées, code de sélection audité. Performance inférieure à H0, donc rejet scientifique.

## Métriques

| Caractéristique | S708 officielle CAL32 | S816 shadow FIT80 |
|---|---:|---|
| Onset F1 | 74,2639 % | inchangée, non recalculée |
| Exact-K | 62,0455 % | inchangée, non recalculée |
| Active Exact-K | 56,2762 % | **56,42413526 % FIT80**, en dessous de H0 FIT80 57,01953803 % |
| Offset accuracy | 77,3292 % | inchangée, non recalculée |
| Full Note F1 | 57,1319 % | inchangée, non recalculée |
| Frame Pitch F1 | 76,4626 % | inchangée, non recalculée |

Référence comparable exclusivement sur FIT80 : H0 57,019538 %, S815 56,912486 %. Deux valeurs issues de l'exécution [S815 #39536863317](https://github.com/Andriamarosoa/note/actions/runs/39536863317). La précision d'une tête Active-K ne correspond pas au score du MIDI joint final. S816 correctes 140728/249411 ; sous-comptages 69540, surcomptages 39143 ; S815 correctes 141946/249411, sous-comptages 65530, surcomptages 41935. Delta vs S815 −0,48835055 point, vs H0 −0,59540277 point. Sur 5 plis, seul Rock3 a changé sa sélection de confidence_direct_pitch à median (58,1004 % → 55,8825 %).

## Comparaison à S708

Aucun code ni événement de la sortie MIDI S708/S707 modifié. S816 ne remplace aucun détecteur LIVE, aucune modification des cinq autres axes ni de leur modèle. La règle de sélection agit sur la seule sortie SHADOW Active-K des trois poids gelés.

## Audits

| Axe | Statut | Preuve/limite |
|---|---|---|
| Données | PASS | Authentification SHA original PCM, manifestes et sources 80/80 dans run scientifique |
| Acoustique | PASS | Extraction native MIDI256 inchangée et schéma S813 requis |
| Architecture | PASS | Trois modèles S815 gelés, sélecteur propre à Active-K seulement |
| Apprentissage | FAIL | Calibration interne seule, mais performance générale 56,42 % sous H0 57,02 % et objectif >85 % échoué |
| Métriques | PASS | Parité 80/80 historiques et S815, nouveau score S816 140728 / 249411 sur MIDI256 |
| LIVE | UNKNOWN | CPU complet/iRig et latence physique non mesurés |

## Verdict

**REJET S816** malgré CI SUCCESS : le correctif anti-surcomptage fait perdre 0,4883 point vs S815, 0,5954 point vs H0. Promotion=false, unification=false, LIVE inchangé. Le seuil de 85 % sur validation indépendante et la validation MIDI final/hardware restent requis.

## Prochaine action

Essayer une architecture indépendante exploitant la morphologie temporelle causalement disponible et/ou les décompositions polyphoniques K4–K6 plutôt qu'une correction globale de seuil, tout en conservant la référence et les six veto de score + preuve hardware.
