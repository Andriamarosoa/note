---
serie: S825
statut: termine
type: experience
reference: S708
branche: codex/note-s825-private-crossfit-pitch-midi256
run_id: 39587162104
commit_sha: d012d7e116789efa50ab88ea9b789685bfb5bc67
promotion: false
---
# S825 — Résultats des boucles de cardinalité apprises sur pseudo-postérieurs indépendants PCM256

## Exécution

En attente du run scientifique : 80 pistes FIT80, 5 folds composition-outer, chaque fold 3 sous-folds de pitch OOF, puis trois classifieurs Active-K distincts. Les métriques proviendront des 249411 trames actives MIDI256 originales.

## Métriques

| Expérience | Active Exact-K FIT80 | Statut |
|---|---:|---|
| H0 source émis MIDI256 | 57,019538 % | Référence historique |
| S815 3 têtes Active-K indépendantes MIDI256 | 56,912486 % | Comparateur |
| S823 S795 dépendant S708, snapshot 1024 | 76,877523 % | Diagnostic non admissible |
| S825 pitch_stats | 53,466367 % | Régression, non promu |
| S825 pitch_harmonic | 55,024037 % | Régression, non promu |
| S825 pitch_temporal | 55,282245 % | Régression, non promu |
| S825 calibration-only | 54,775852 % | Régression, non promu |

## Six caractéristiques officielles S708 CAL32

| Caractéristique | S708 CAL32 | S825 |
|---|---:|---|
| Onset F1 | 74,2639 % | Non modifié |
| Exact-K | 62,0455 % | Non modifié |
| Active Exact-K | 56,2762 % | Recherche SHADOW seule |
| Offset accuracy | 77,3292 % | Non modifié |
| Full Note F1 | 57,1319 % | Non modifié |
| Frame Pitch F1 | 76,4626 % | Non modifié |

## Comparaison à S708

S708 officiel et LIVE inchangés. S825 peut produire une décision à chaque hop de 256 échantillons, mais le temps d'extraction et d'inférence réel doit encore être mesuré; aucun échantillon futur ni hold supplémentaire.

## Audits

| Domaine | Statut | Justification |
|---|---|---|
| Données | PASS | 80 WAV originaux SHA, H0 142213/249411, 306536 trames native256 |
| Acoustique | PASS | Aucune entrée S708/MIDI; PCM et DSP réellement actualisées toutes les 256 samples |
| Architecture | PASS | 4 modèles privés par pli : pitch_activity et 3 classifieurs K indépendants |
| Apprentissage | PASS | 3 sous-plis OOF de composition puis 48 TRAIN/16 inner-CAL/16 OUTER |
| Métriques | PASS | 3 boucles × 5 compositions, 249411 GT actives, sous/sur et K1..6 |
| LIVE | UNKNOWN | No lookahead/hold, mais pas de temps CPU+hardware microphone mesuré |

## Verdict

Aucune promotion ni unification : métriques réelles S825 toutes inférieures à H0 et à l'objectif de 85 %.

## Prochaine action

Les 3 ablations ont été comparées : rejeter toutes les régressions; auditer les mauvaises distinctions K4/K5/K6 et la généralisation par composition. Une nouvelle architecture privée et un test externe sont nécessaires avant promotion.



## Clôture scientifique vérifiée — 21 octobre 2026

**Le vrai run [#39587162104](https://github.com/Andriamarosoa/note/actions/runs/39587162104) a terminé SUCCESS**, commit `d012d7e116789efa50ab88ea9b789685bfb5bc67`, [artefact modèles, 80 prédictions et rapport #12339688454](https://github.com/Andriamarosoa/note/actions/runs/39587162104/artifacts/12339688454). Le premier run #39586398675 a été rejeté: l'ancien appel au module S822 subsistait dans la commande CI, bien que les tests S825 aient passé; cette erreur a été corrigée avant le deuxième run, **qui exécute réellement S825**. Il est interdit de confondre ses anciens chiffres S822 avec les résultats S825.

Le protocole réel utilise 80 pistes WAV GuitarSet originales SHA, 306536 trames 256 samples à 44100Hz, 249411 trames GT actives, 5 outer compositions de 16 pistes; chacune est exclue de 48 TRAIN et de 16 CAL disjoints. Les 48 TRAIN sont divisées en 3 groupes de 16, chaque groupe reçoit les scores d'une tête pitch apprise uniquement sur les deux autres groupes TRAIN. **Aucune prédiction pitch in-sample n'alimente un entraînement de cardinalité**. Un modèle final privé pitch est entraîné sur 48 TRAIN pour CAL/OUTER. Les trois classifieurs de cardinalité ont leurs propres poids et sont sélectionnés uniquement par inner-CAL. Aucun postérieur S708 ni MIDI émis, sortie/poids d'un autre axe, futur audio ou hold post-attaque. Aucun des scores ne provient d'une reconstruction MIDI final.

### Scores S825 GuitarSet FIT80 MIDI256

| Modèle | Bonnes décisions / 249411 | Active Exact-K | Undercount | Overcount |
|---|---:|---:|---:|---:|
| H0 MIDI historique, témoin | 142213 | **57,019538 %** | 80654 | 26544 |
| S815 indépendante, historique | 141946 | 56,912486 % | 65530 | 41935 |
| S823 SHADOW S708/1024 dépendant | 191741 | 76,877523 % | 34611 | 23059 |
| S825 pitch_stats | **133351** | **53,466367 %** | 82968 | 33092 |
| S825 pitch_harmonic | **137236** | **55,024037 %** | 80231 | 31944 |
| **S825 pitch_temporal (meilleur modèle fixe)** | **137880** | **55,282245 %** | **79097** | **32434** |
| S825 sélection inner-CAL | **136617** | **54,775852 %** | **78818** | **33976** |

Le meilleur mode fixe est **−4333 décisions** / **−1,7373 point** sous H0, malgré **1557 undercounts évités**, en contrepartie de **5890 overcounts supplémentaires**. La sélection CAL baisse encore de 1263 décisions par rapport au mode fixe temporal. Aucune promotion n'est justifiée.

### Lacunes par vrai K du meilleur modèle pitch_temporal

| Vrai K | Correct / frames | Précision conditionnelle | Sous-comptages | Surcomptages |
|---|---:|---:|---:|---:|
| K1 | 92024/112014 | 82,15 % | 11236 | 8754 |
| K2 | 16894/46311 | 36,48 % | 20711 | 8706 |
| K3 | 12258/34182 | 35,86 % | 13469 | 8455 |
| K4 | 13730/35190 | 39,02 % | 16709 | 4751 |
| **K5** | **2121/16323** | **12,99 %** | **12434** | 1768 |
| **K6** | **853/5391** | **15,82 %** | **4538** | 0 |

**Falsification** : les probabilités de hauteur natives256 issues d'un modèle privé n'ont PAS retrouvé la capacité de S708, notamment K5/K6; une simple somme ou un classifieur logistique à partir de la carte OOF de hauteurs ne suffit pas. La compétition harmonique augmente certains plis mais l'erreur dominante reste le **sous-comptage des fortes polyphonies**, et son correction aveugle crée des fausses K5/K6. Par composition, la variante temporale donne BN1 +617 et Funk2 +20 décisions vs H0, mais Jazz1 **−1217**, Rock3 **−1893**, SS1 **−1860**. Une amélioration moyenne pourrait cacher des régressions : ne pas l'accepter.

### Verdict et prochain diagnostic à mener

**S825 TERMINÉ — trois boucles de modèles indépendants vérifiées; toutes régressives sur H0 ; promotion FALSE.** Six axes CAL32 S708 inchangés, aucune unification, LIVE strict inchangé. Résolution de décision MIDI256 = 5,805ms, pas une mesure de latence matérielle. FIT80 déjà exploré, ne pas transformer ses chiffres en validation externe.

Pour la suite, rechercher une architecture **vérificateur ordinal conditionnel K3→K4→K5→K6** entraînée sur des sorties OOF de hauteur mais avec preuves acoustiques privées de séparation fondamentale/harmoniques. En parallèle évaluer **les erreurs du modèle pitch privé** par K et par registre, et tester la calibration de probabilité sur CAL indépendante, sans choisir les seuils en consultant les métriques OUTER. Conserver une variante BASE et des veto par K, composition, under/over avant toute promotion; effectuer une validation externe indépendante avant d'envisager un système unifié.
