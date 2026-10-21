---
serie: S830
statut: termine
type: experience
reference: S708
branche: codex/note-s830-private-musical-evidence256
run_id: 39632626042
commit_sha: 088e28e122bfc1f04c981716bf8db73faa615ca7
promotion: false
---

# S830 — Résultats scientifiques sur 80 pistes FIT80 natif256

## Exécution
**COMPLETED SUCCESS** — [GitHub Actions #39632626042](https://github.com/Andriamarosoa/note/actions/runs/39632626042), commit scientifique `088e28e122bfc1f04c981716bf8db73faa615ca7`. Deux jobs `smoke` et `signed_fit80` ont réussi. L'artefact `S830_FIT80_INDEPENDENT_MUSICAL_HEADS_PCM256` (id `12360813096`) contient `report.json` complet avec 80 pistes, cinq folds et 306 536 trames. Source : S800 FIT80 avec métadonnées et WAV originaux vérifiés. La première exécution a échoué sur la fin PCM partielle, bug corrigé avant la validation finale.

## Métriques
Le dénominateur Exact-K est constitué de **toutes les trames K0–K6**, y compris K0 et non uniquement des trames actives.

| Source | Correctes / 306 536 | Exact-K | Sous-comptages | Surcomptages |
|---|---:|---:|---:|---:|
| H0 original (même FIT80) | 189 124 | **61,6972 %** | 80 654 | 36 758 |
| S829 réentraîné (même FIT80) | 176 334 | **57,5247 %** | 83 523 | 46 679 |
| **S830 (5 têtes privées)** | **176 334** | **57,5247 %** | **83 523** | **46 679** |

**Différence S830 – S829 : 0,0000 point**. S830 est à −4,1724 points du témoin H0. Parité parfaite et régression nulle du chemin S829 dans l'expérience. Aucune amélioration des classes polyphoniques démontrée.

| Composition OUTER | Exact-K S830 | Mode retenu inner-CAL | Seuil | Shift |
|---|---:|---|---:|---:|
| BN1 | 61,31 % | `s829` | 0,30 | 0 |
| Funk2 | 61,11 % | `s829` | 0,30 | 0 |
| Jazz1 | 57,48 % | `s829` | 0,30 | 0 |
| Rock3 | 54,83 % | `s829` | 0,30 | 0 |
| SS1 | 53,37 % | `s829` | 0,30 | 0 |

Le choix `s829` aux cinq folds signifie que les sorties du modèle musical sont **ignorées pour le K final** ; elles ont été entraînées et évaluées en CAL mais le sélecteur a retenu le témoin. Le rapport courant enregistre seulement le score CAL du gagnant, **pas** toutes les métriques des modes alternatifs, ce qui empêche de quantifier leur déficit par rapport à S829 sans nouvel audit ciblé.

| Vrai K | Trames | Exact-K S830 | Sous-comptages |
|---|---:|---:|---:|
| K0 | 57 125 | 80,30 % | 0 |
| K1 | 112 014 | 83,37 % | 9 160 |
| K2 | 46 311 | 22,02 % | 24 957 |
| K3 | 34 182 | 26,29 % | 14 612 |
| K4 | 35 190 | 43,61 % | 16 132 |
| K5 | 16 323 | 12,85 % | 13 722 |
| K6 | 5 391 | 8,37 % | 4 940 |

## Six métriques officielles du MIDI unifié
| Caractéristique | Résultat S830 (MIDI final) |
|---|---|
| Onset F1 | NON MESURÉ — axe intact |
| Exact-K | 57,5247 % SHADOW K uniquement ; le MIDI final n'a pas été reconstitué |
| Active Exact-K | NON MESURÉ — S828 reste la référence de développement séparée |
| Offset | NON MESURÉ — axe intact |
| Full Note F1 | NON MESURÉ — pas d'intégration |
| Frame Pitch F1 | NON MESURÉ — axe intact |

La référence officielle S708 CAL32 est protégée. S809 84,24 % SHADOW à 1 024 échantillons et S828 73,83 % Active Exact-K à 256 ont des protocoles différents, donc ne peuvent pas être comparés directement à cette métrique.

## Audits
| Catégorie | Statut | Preuve |
|---|---|---|
| Données | PASS | 80 WAV GuitarSet d'origine, SHA provenance, composition-disjoint |
| Acoustique | UNKNOWN | Tests synthétiques causalité/PCM256 PASS, mais généralisation acoustique et matériel iRig non validés |
| Architecture | UNKNOWN | Isolation de code PASS et cinq autres axes inchangés ; audit adversarial de toutes les interactions non réalisé |
| Apprentissage | PASS | Deux experts K + trois preuves musicales appris sur TRAIN, selector inner-CAL |
| Métriques | PASS | 306 536 trames K0–K6 et comparaison S829/H0 appariée |
| LIVE | UNKNOWN | p95/p99/latence matérielle iRig non mesurée |

Les six agents Work et trois avis NVIDIA authentiques n'ont pas été invoqués. Aucun certificat de généralisation HELD, de latence physique ou de non-régression du MIDI unifié.

## Comparaison à S708
Aucun changement de référence S708, S828, MIDI officiel ou des cinq autres axes. S830 s'exécute en SHADOW, indépendant. Son sélecteur conservateur a systématiquement retenu S829 ; il n'y a ni gain ni perte.

## Verdict
**Expérience TERMINÉE — NON PROMUE.** L'indépendance est possible mais les preuves musicales nouvelles, dans leur forme actuelle, n'apportent pas de gain Exact-K en sélection CAL. Avant d'étudier une influence entre axes, instrumenter les performances par candidat et par classe K, puis comparer les corrections conditionnelles sur CAL et OUTER sans modifier les producteurs de caractéristiques. La contrainte zéro lookahead/hold reste obligatoire.

## Prochaine action
Ne pas promouvoir S830. Instrumenter le score des sorties de chaque expert sur CAL et par vrai K, puis étudier une communication en lecture seule et sens unique entre architectures avec ablations et veto de régression, sans fusion des états et sans ajouter de latence.

### Audit du code S809 → S829 → S830 (21 octobre 2026)

L'audit suivant est **statique, basé sur les lignes du dépôt et sur le rapport FIT80 réussi #39632626042**. Les deux contre-exemples sont calculés avec NumPy ; ils n'attribuent pas, à eux seuls, une causalité quantitative aux erreurs de GuitarSet. Aucun modèle ni sortie MIDI n'est modifié par cet audit.

| Priorité | Observation et localisation | Implication | Action requise |
|---|---|---|---|
| **P0 – diagnostic** | `scripts/s830_train_private_musical_evidence256.py:138-145` sauvegarde uniquement `selected_CAL_stats`, et `:148-166` n'enregistre en OUTER que le gagnant. Le mode `s829` a été retenu sur les cinq folds. | Le score final 57,52 % ne démontre **pas** que les nouvelles têtes sont intrinsèquement mauvaises ; leurs scores individuels, corrections et régressions sont absents du rapport. | Enregistrer sur CAL et OUTER les métriques de chacun des cinq modes, par vrai K, sans utiliser OUTER pour choisir les modèles. Calculer corrections nouvelles, fuites et cas exclusivement résolus. |
| **P0 – LIVE** | `src/causal_note/s830_private_musical_evidence256.py:249-267` évalue toujours les **trois nouvelles têtes** même quand `mode == 's829'`, puis jette leur décision. `:252` commence le chronomètre **après** le frontal FFT. | Travail supplémentaire certain sans gain pour le mode retenu ; `last_inference_runtime_ms` ne mesure pas la chaîne PCM→FFT complète ni l'I/O. | Court-circuiter les experts inactifs ; chronométrer tous les composants et les p50/p95/p99 sur matériel réel, sans masquer les délais. |
| **P1 – relations harmoniques** | `src/causal_note/s830_private_musical_evidence256.py:107-130` forme la ligne d'entraînement avec seulement `features[frame_idx,pitch_idx]` et un scalaire de hauteur. | L'expert E3 ne voit pas directement qu'un E2 voisin peut expliquer les mêmes partiels. Cette information manquante est une cause plausible de confusion polyphonique. | Ajouter **au sein de la seule architecture Exact-K** les résidus et preuves de fondamentales, comparateurs d'octaves, voisins et harmoniques. |
| **P1 – routeur trompeur** | `src/causal_note/s830_private_musical_evidence256.py:218` et `:264` calculent `mean(max(p,1-p))` sur 49 hauteurs ; `:180-182` choisit `sustain` dès `confidence >= .75`. | Pour 49 valeurs `p(sustain)=0.01`, la confiance vaut **0.99** et le routeur choisit pourtant le compteur sustain, même sans preuve positive de note. | Remplacer par une confiance de cardinalité, des marges entre hypothèses de K ou un modèle de risque calibré sur CAL. Ajouter un test anti-surconfiance du silence. |
| **P1 – déséquilibre** | `src/causal_note/s830_private_musical_evidence256.py:98-136` limite à 160 positifs/240 négatifs par piste et restaure le déséquilibre réel par les poids ; `:149-165` applique des seuils fixes. | Le vrai K moyen FIT80 est **1,833** sur 49 classes possibles, soit environ **3,74 %** de positions actives. Onset et Release sont encore plus rares. Rien ne mesure la calibration, le rappel ou le taux de faux positifs de chaque tête. | Tester calibration par tête, PR/ROC, distribution probabiliste, rappel par K et seuils TRAIN/CAL seuls. Ne pas rebaptiser une hypothèse en cause établie. |
| **P1 – compression** | `src/causal_note/s829_exactk_s809_native256.py:110-131` calcule des agrégats `mean/std/max/top3/top8` sur les 49 hauteurs. | Une permutation de lignes de hauteurs conserve presque exactement le résumé (contre-exemple NumPy : variation maximale `2.4e-7`). Les arrangements d'intervalles utiles à l'explication harmonique sont perdus pour les deux experts K initiaux. | Ajouter des statistiques **relatives entre hauteurs** et des spectres résiduels, sans emprunter les poids d'autres axes. |
| **P2 – sélection non dynamique** | `src/causal_note/s830_private_musical_evidence256.py:187-209` retient un seul `mode` pour toute une composition. Seul `router` contient une commutation par fragment ; `s829` ignore entièrement le seuil exploré. | Les cinq valeurs `threshold=0.3` accompagnant `mode=s829` ne prouvent aucune calibration utile de seuil. Un expert supérieur sur K3 seulement peut être globalement rejeté. | Ajouter sélection locale causalement fondée sur les sorties privées, et prouver le gain par ablations + veto de régression. |
| **P2 – labels d'événements** | `src/causal_note/s830_private_musical_evidence256.py:42-49` dérive Onset/Release uniquement d'une transition d'occupation binaire. | Des notes réattaquées à la même hauteur **sans trame vide** sont confondues avec une tenue continue ; le label Onset perd l'événement. | Dériver attaques et fins depuis les vrais événements TRAIN horodatés, compatibles avec l'ABI 256, sans visibilité du futur à l'inférence. |
| **P2 – tests incomplets** | `test/test_s830_private_musical_evidence256.py:58-105` teste la mutation du futur sur 7 hops et la parité entraînée sur seulement 5 hops, tous inférieurs à la plus grande fenêtre FFT **8 192 échantillons = 32 hops**. | Aucune preuve de parité/causalité après saturation et rotation du tampon historique sur une vraie piste, ni d'identité `float16` (stockage FIT80) vs `float32` (LIVE). | Tester 64–128 hops, des découpages irréguliers, silence→accord→silence, réattaques, E2/E3 et les deux dtypes ; comparer les décisions exactes FIT/live. |
| **P2 – déploiement** | Le workflow fournit seulement `report.json`, sans sérialiser les 5 modèles entraînés ; aucune liaison au MIDI public ni mesure p99. | S830 ne fournit pas de version binaire déployable du modèle évalué et le délai Live n'est pas certifié. | Sauvegarder les cinq poids/checkpoints + SHA, fabriquer le runner causal strict, mesurer le délai total et tester la non-régression des six sorties. |

**Clarification S809 :** `s809_isolated_axis_pipeline.py:56-92` reçoit déjà le `Fragment.post` issu de prédictions existantes ; `scripts/s809_fit80_independence_replay.py:86-92` fabrique ce fragment à partir de traces archivées à 1 024 échantillons. Cela prouve la séparation de ses têtes/sélecteurs/décodeurs, **pas** six frontaux PCM indépendants. Les 84,24 % SHADOW S809 ne sont donc pas un contrôle apparié pour S830 natif256.

**Décision de l'audit :** aucune promotion et aucune influence entre caractéristiques avant instrumentation des propositions et tests de régression. La prochaine série devra démontrer séparément : (1) les corrections propres de chaque candidat K ; (2) le bénéfice des relations harmoniques ; (3) la sélection locale ; (4) les performances temporelles physiques.
