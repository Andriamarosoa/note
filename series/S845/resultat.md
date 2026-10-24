---
serie: S845
statut: termine
type: experience
reference: S708
branche: codex/note-s845-raw-wave-acoustic-decoder256
run_id: 40089022537
commit_sha: 0a1653d50289c9f8b67c88d6e3067300959f0cd2
promotion: false
---

# S845 — Reconstruction de six voies réelles après conservation de la forme d'onde originale

## Exécution
Code du décodeur `local_string_filters` et `string_condition` ajouté sur une **branche expérimentale nouvelle**, distincte de S844. Entrée strictement 256 nouveaux échantillons, état GRU du passé, six formes d'onde estimées (pas les véritables voies garanties) et ablation latente ±12.

## Mesures définitives S845 — contrôle GuitarSet réel

[GitHub Actions #40089022537](https://github.com/Andriamarosoa/note/actions/runs/40089022537) **SUCCESS** le 24 octobre 2026 à 11:29:56 UTC. Sources micro, hex 6 canaux et JAMS originales vérifiées par SHA/MD5, 16 pistes TRAIN (joueur 04), 16 CAL (joueur 05), 64 OUTER (joueurs 00..03) ; 49 152 blocs 256 sur OUTER. Réentraînement de huit époques et meilleur checkpoint selon CAL à l'époque **6** ; 1 424 vraies paires sémantiques d'octaves enregistrées. Test de code exact PCM256 et de non modification du MIDI PASS. Artefact GitHub `S845_GENUINE_HEX_RAW_WAVEFORM_NEURAL_PCM256` ID **12558990965**.

### Résultats de notes actives sur OUTER (pas Onset événement)

| Vérité GT | Témoin même encodeur sans vues (TP/FP/FN) | F1 témoin | Avec IA transposition ±12 (TP/FP/FN) | F1 IA |
|---|---:|---:|---:|---:|
| E2, 3 889 positifs | 916/8335/2973 | **13,9422 %** | 974/8752/2915 | **14,3077 %** |
| E3, 4 953 positifs | 1409/10014/3544 | 17,2081 % | 1422/10139/3531 | **17,2218 %** |
| E2+E3, 1 921 positifs | 547/8589/1374 | 9,8942 % | 518/7867/1403 | **10,0524 %** |

Le transport neural augmente très légèrement le score agrégé dans chaque catégorie de cette expérience, mais **n'établit pas une amélioration convaincante** et les FP restent massifs. La baisse S845 vs S844 est sensible, malgré l'entrée directe micro dans le décodeur. Ces chiffres sont des proxy F1 **présence active de notes sur des blocs sélectionnés**, pas des notes MIDI/Onset.

### Test de vérité physique du son estimé

| Comparaison à la vraie prise hex du même morceau | MAE OUTER, plus faible = meilleur |
|---|---:|
| Silence (six voies sorties zéro) | **0,029205378** |
| Ancien S844 reconstruit | 0,029785044 |
| Nouveau S845 reconstruit | **0,029422376** |

Différence silence − S845 : **−0,000216998** (encore moins bon que silence), malgré le progrès par rapport à S844. **Moyenne amplitude absolue de S845 estimé : 0,000855748**, à comparer à la **vraie cible hex 0,029205378**. Cette très faible énergie est l'indice d'un décodeur quasi silencieux, pas une preuve de reconstruction de cordes fidèles. L'alignement exact des six prises micro/hex et des indices de corde JAMS n'a pas été audité physiquement. F1 présence corde (classification GT) **54,6021 %**, qui ne prouve pas la fidélité sonore. Mesure Linux d'un unique flux sur huit pas seulement : p99 **3,3386 ms**, pas un certificat Windows/iRig temps réel ou Note Live complet.

**Décision : S845 REJETÉ au critère de reconstruction réelle vs silence.** Le modèle et son outil d'export WAV restent isolés pour auditer les estimations, aucun MIDI ni tête Onset promus. La référence de travail Onset reste 69,2584331 % F1 (7607 TP/3093 FP/3660 FN).

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

### S845 — mesures réellement obtenues

Les résultats complets et TP/FP/FN sont dans la section « Mesures définitives S845 » ci-dessus.

| Axe | Score S845 |
|---|---|
| MAE hex / silence | **0,029422376** modèle vs **0,029205378** silence : échec |
| E2 par bloc, TP/FP/FN | **14,3077 %** avec transposition vs **13,9422 %** sans, fortement insuffisant |
| E3 par bloc, TP/FP/FN | **17,2218 %** avec transposition vs **17,2081 %** sans |
| E2+E3 par bloc, TP/FP/FN | **10,0524 %** avec transposition vs **9,8942 %** sans |
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
| Données | PASS | 3 archives originales vérifiées ; partitions par guitariste conservées |
| Acoustique | PASS code | voie directe CNN sur vrai microphone, six sorties apprises, sans futur |
| Architecture | PASS code | un décodeur neuronal distinct, GRU conservé, autres axes et MIDI gelés |
| Apprentissage | PASS technique | vraies cibles micro/hex, checkpoint époque 6 choisi sur CAL |
| Métriques | FAIL scientifique | meilleur que S844 en MAE mais toujours pire que le silence ; F1 très faible |
| LIVE | UNKNOWN | 5,805ms cadre causal, p99 matériel à prouver |

## Verdict
**TERMINÉ, NON PROMU**. Malgré l'entraînement sur de vrais sons de guitare, la tête S845 est encore moins fidèle à la prise hex originale que la sortie zéro. Aucun progrès pertinent ne permet une promotion. L'audit physique de l'alignement micro/hex et des six canaux est prioritaire ; un décodeur à forme d'onde entraîné sur une cible décalée temporellement peut favoriser les sorties peu énergétiques. Cette cause est une hypothèse à démontrer, pas un diagnostic confirmé.

## Prochaine action
Comparer les vrais résultats du nouveau décodeur à silence et à S844, auditer si les six canaux hex sont physiquement alignés et quels modes de perte dominent, puis seulement envisager une tête Onset indépendante pouvant exploiter les composantes validées.
