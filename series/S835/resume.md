---
serie: S835
statut: en_cours
type: experience
reference: S708
branche: codex/note-best-of-six-axes-preserved
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S835 — Branche de conservation des six meilleures caractéristiques

## Objectif
Constituer un point de départ GitHub **contenant simultanément les implémentations des meilleures références documentées pour chaque caractéristique**, sans les mettre à tort dans un seul décodeur MIDI ni sacrifier la meilleure référence d'Onset ou d'Offset. C'est une **branche de conservation**, non une promotion. Les meilleures références ont des grilles, cohortes et unités parfois différentes ; leur métrique doit rester attachée à sa source.

## Hypothèse
Réutiliser les implémentations conservées de S834 et de ses ancêtres S708/S790/S809/S823/S828 permet de versionner les architectures favorables sans modifier des fichiers de production. Une déclaration structurée, vérifiée automatiquement, empêche qu'un score à 1024 ou une F1 d'événements soit déclaré score MIDI256 ou accuracy Offset.

## Modifications
- Branche `codex/note-best-of-six-axes-preserved` créée depuis `codex/note-s834-s809-six-axes-native256` : toutes les sources existantes restent présentes (S828 pour Active-K, S834 pour Exact-K/Frame Pitch/Full Note, S812/S708 pour les axes historiques Onset/Offset).
- `configs/best_six_axes_preserved.json` : six références choisies, pistes FIT80, preuves GitHub, bonnes décisions et dénominateurs, variantes historiques à cadence différente, distinction SHADOW / official / PCM256.
- `src/causal_note/s835_best_six_registry.py` : verrouille les six choix, refuse toute régression d'exactitude sur protocole apparié, interdit de mélanger les cohortes/métriques, exige identité de toutes les cinq autres sorties en cas de changement isolé, empêche toute promotion par défaut.
- `test/test_s835_best_six_registry.py` : protections anti-régression, anti-ambiguïté des scores Offset et des performances héritées à 1024, blocage d'une unification non testée.
- Aucun modèle original ni code MIDI de production modifié et aucun poids nouvellement entraîné ou prêt à déployer prétendu.

## Sélection — scores vérifiés sur FIT80
| Axe | Meilleure option conservée pour le travail courant | Score | Cadence |
|---|---|---:|---|
| Exact-K | S834 six axes SHADOW | 75,780659 % | PCM256 |
| Active Exact-K | S828 readout privé | 73,831146 % | PCM256 |
| Onset F1 | S708/H0 historique PROTÉGÉ | 73,252019 % | Source historique, **pas une tête nouvelle PCM256 certifiée** |
| Offset accuracy | S708/H0 historique PROTÉGÉ | 69,886229 % | Source historique, **pas le F1 d'offset S834** |
| Frame Pitch F1 | S834 six axes SHADOW | 84,998850 % | PCM256 |
| Full Note F1 | S834 six axes SHADOW | 71,108037 % | PCM256 |

Records d'autres grilles **conservés en référence** mais pas interchangés avec les résultats ci-dessus : Exact-K S809 84,24 % SHADOW1024 ; Active-K S823 76,8775 % dépendant de snapshots 1024 ; Frame Pitch S790 86,7532 % FIT80 SHADOW1024 (85,1177 % CAL32). Scores S708 officiels CAL32 : Onset 74,2639 %, Exact-K 62,0455 %, Active-K 56,2762 %, Offset accuracy 77,3292 %, Frame Pitch 76,4626 %, Full Note 57,1319 %.

## Protocole
Cette branche **n'évalue pas un nouveau flux MIDI** : aucune concaténation de ces six performances, aucune revendication de score d'unification ni de six têtes pleinement indépendantes depuis l'échantillon PCM. Les meilleurs modules SHADOW S834 partagent un backbone S708 gelé, dont le modèle source a été entraîné à 1024 puis recalculé à 256. S828 dépend également du backbone mais possède un readout K privé entraîné à 256, dont les poids de run n'ont pas été empaquetés comme production. Onset/Offset conservent S708/H0 parce qu'aucun remplaçant natif256 n'a encore été prouvé non régressif sur une même métrique et une même cohorte. Les cinq autres axes sont protégés contre toute modification croisée.

## Audits obligatoires
- Données : PASS si les liens vers les six rapports sources sont présents et les scores portent les bonnes métriques.
- Acoustique : audit de cohérence de la cadence ; **6/6 PCM256 non établi**, Onset/Offset historiques marqués comme tels.
- Architecture : contrôle de fichiers gelés plus interdiction d'injecter la sortie d'un axe vers un autre.
- Apprentissage : aucune tête nouvellement entraînée ; modèles historiques doivent rester sourcés.
- Métriques : aucune comparaison CAL32/FIT80 ou Offset F1/accuracy invalide ; deux protections Onset/Offset explicitement bloquantes.
- LIVE : NON VALIDÉ tant que p95/p99 matériel et budget complet strict 5,805 ms ne sont pas démontrés.
- Agents externes NVIDIA/Work : non impliqués.

## Statut
Branche de référence S835 en cours de test. Aucune promotion ni fusion MIDI autorisée ; la sélection indépendante est consignée, pas assemblée. Prochaine boucle indispensable : valider un Onset PCM256 **au moins aussi bon** que 73,2520 % FIT80 et un Offset accuracy PCM256 apparié au H0 69,8862 %, puis vérifier les six métriques sans régression et le budget temps réel.
