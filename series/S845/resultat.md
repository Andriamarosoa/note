---
serie: S845
statut: en_cours
type: experience
reference: S708
branche: codex/note-s845-raw-wave-acoustic-decoder256
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S845 — Reconstruction de six voies réelles après conservation de la forme d'onde originale

## Exécution
Code du décodeur `local_string_filters` et `string_condition` ajouté sur une **branche expérimentale nouvelle**, distincte de S844. Entrée strictement 256 nouveaux échantillons, état GRU du passé, six formes d'onde estimées (pas les véritables voies garanties) et ablation latente ±12.

## Métriques
### S844 — données vraies et constat déclencheur

[GitHub Actions #40087751049](https://github.com/Andriamarosoa/note/actions/runs/40087751049) SUCCESS, 16 TRAIN / 16 CAL / 64 OUTER, 49152 blocs OUTER, 8 époques, 1424 paires d'octave vraies.

| Diagnostic présence dans les blocs | Sans transposition S844 | Avec IA latente S844 |
|---|---:|---:|
| E2 (3889 positifs) | 17,5664 % F1 | 17,3446 % F1 |
| E3 (4953 positifs) | 22,2019 % F1 | 22,2095 % F1 |
| E2+E3 (1921 positifs) | 11,0171 % F1 | 10,9956 % F1 |

C'est une comparaison **intra-run** équitable avec et sans vues latentes ; le proxy S843 avait une autre sélection de fenêtres et ne peut servir à annoncer un gain quantitatif de S844. Le transport IA n'a pas de gain convaincant sur E2/E3. Les nombreuses fausses attaques potentielles rendent toute promotion impossible.

| Mesure S844 vraie hex (mae moyen, le plus bas est le meilleur) | Valeur |
|---|---:|
| Prédicteur silence | **0,029205378** |
| Réseau S844 sur vraie cible hex | **0,029785044** |
| Différence silence − réseau | **−0,000579666** |
| Moyenne amplitude de la sortie réseau | 0,001868151 |
| Moyenne amplitude cible hex | 0,029205378 |

**Échec scientifique de reconstruction S844 : moins bon que silence**, même si l'entraînement sur les vrais signaux a bien été exécuté.

### S845 — résultats en attente

| Axe | Score S845 |
|---|---|
| MAE hex / silence | **NON MESURÉ** |
| E2 par bloc, TP/FP/FN | **NON MESURÉ** |
| E3 par bloc, TP/FP/FN | **NON MESURÉ** |
| E2+E3 par bloc, TP/FP/FN | **NON MESURÉ** |
| Onset F1 de référence de travail | **69,2584331 %** (7 607 TP / 3 093 FP / 3 660 FN), INTACT |
| Exact-K | INTACT, non réévalué |
| Active Exact-K | INTACT, non réévalué |
| Offset | INTACT, non réévalué |
| Frame Pitch F1 | INTACT, non réévalué |
| Full Note F1 | INTACT, non réévalué |

## Comparaison à S708
S708/H0 historique 73,252019 % Onset entraîné à 1024, exclu. La tête S845 n'utilise que des blocs PCM256 et des cibles hex originales. Aucun chiffre de F1 de présence ne doit être déclaré comme F1 Onset ou un gain du MIDI.

## Audits
| Rôle | État | Contrôle |
|---|---|---|
| Données | UNKNOWN | nouvelle exécution des trois archives GuitarSet et découpe inchangée |
| Acoustique | PASS code | voie directe CNN sur vrai microphone, six sorties apprises, sans futur |
| Architecture | PASS code | un décodeur neuronal distinct, GRU conservé, autres axes et MIDI gelés |
| Apprentissage | UNKNOWN | réapprentissage réel à évaluer |
| Métriques | UNKNOWN | MAE vs silence et proxys E2/E3/joint sur OUTER à publier |
| LIVE | UNKNOWN | 5,805ms cadre causal, p99 matériel à prouver |

## Verdict
**NON PROMU** avant validation de mesures acoustiques vraies. Si S845 n'améliore pas MAE vs silence, l'approche doit être revue (par ex. audit physique de l'alignement mic/hex) plutôt que de prétendre reconstruire de « vrais sons ».

## Prochaine action
Comparer les vrais résultats du nouveau décodeur à silence et à S844, auditer si les six canaux hex sont physiquement alignés et quels modes de perte dominent, puis seulement envisager une tête Onset indépendante pouvant exploiter les composantes validées.
