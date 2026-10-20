---
serie: S805
statut: termine
type: experience
reference: S708
branche: codex/note-s805-fit-crossfold-onset-risk
run_id: 39371919146
commit_sha: ee76229413424c8a1d902740077aa492162060ff
promotion: false
---

# S805 — rapport final scientifique GuitarSet FIT80

## Exécution

[Run GitHub vérifié #39371919146](https://github.com/Andriamarosoa/note/actions/runs/39371919146) **COMPLETED SUCCESS**, code scientifique commit `ee76229413424c8a1d902740077aa492162060ff`. S800 signed FIT80 80/80 pistes, 5 compositions, S802 forensics et référence S803 intégralement rejoués, source 306 536 frames et 189 124 Exact-K corrects. Tous les 12 compteurs natifs originaux reproduits, sans utilisation de CAL32, HELD ou TEST. 6 rôles d'audit indépendants PASS et consensus six PASS. Les trois avis NVIDIA restent SKIPPED car la consultation est volontairement réservée à un déclenchement manuel afin d'éviter les doublons et échecs S804. En conséquence, **9/9 est UNKNOWN, pas PASS**.

## Métriques

Scores du même corpus GuitarSet FIT80, pourcentage (ces candidats ne sont **pas combinables** par ligne). Les six sorties MIDI256 sont mesurées réellement sur les 80 pistes, pas extrapolées.

| Politique | Onset F1 | Exact-K | Active Exact-K | Offset accuracy | Full Note F1 | Frame Pitch F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S800/S803 témoin FIT | **73,2520 %** | 61,6972 % | 57,0195 % | **69,8862 %** | **50,6528 %** | 78,7278 % |
| spectral_balance | 72,9808 % | 61,6877 % | 57,0111 % | 69,5024 % | 50,1882 % | 78,7566 % |
| risk_acoustic | 72,8810 % | **61,7122 %** | **57,0412 %** | 69,3984 % | 50,0448 % | **78,8133 %** |
| risk_joint | 72,9091 % | 61,7092 % | 57,0376 % | 69,4787 % | 50,1218 % | 78,7991 % |

Trois variantes : zéro éligible. Le filtrage `spectral_balance` a bloqué 44 cellules proposées et produit 92 attaques synthétiques ; `risk_acoustic` a bloqué 1 cellule pour 135 attaques ; `risk_joint` en a bloqué 3 pour 115 attaques. Sur chaque variante le F1 Onset, le Full Note F1 et la précision Offset régressent. Des micro-gains Exact-K ou Frame Pitch ne compensent pas cette dégradation, et des variations par composition subissent aussi des régressions.

| Métrique | Référence officielle S708 CAL32 |
| --- | ---: |
| Onset F1 | 74,2639 % |
| Exact-K | 62,0455 % |
| Active Exact-K | 56,2762 % |
| Offset accuracy | 77,3292 % |
| Full Note F1 | 57,1319 % |
| Frame Pitch F1 | 76,4626 % |

Ces valeurs **CAL32 ne doivent pas être comparées directement** aux scores FIT80 du tableau précédent.

## Comparaison à S708

Aucune politique promue. S708 reste référence officielle inchangée. Aucun fichier LIVE de production ni émission native S708 n'a été modifié. Les deux têtes de risque sont des expériences SHADOW, elles n'utilisent ni lookahead ni NoteOff retenu, mais la latence CPU ou microphone de leur utilisation éventuelle est **non mesurée**. Toutes les expériences ont un veto strict de régression sur les six métriques globales ET chacune des cinq compositions.

## Audits

| Audit | Statut | Preuve |
| --- | --- | --- |
| Données | PASS | 39371919146, 80/80 SHA et 12 compteurs originaux |
| Acoustique | PASS | 39371919146, features exclusivement trame courante |
| Architecture | PASS | 39371919146, 5 têtes gelées + 2 têtes de risque croisées |
| Apprentissage | PASS | 39371919146, les quatre autres compositions uniquement pour chaque fold |
| Métriques | PASS | 39371919146, 6 scores et 5 compositions, veto correct |
| LIVE | PASS | 39371919146, 0 hold/lookahead additionnel ; latency physique NON MESURÉE |

Consensus local 6/6 **PASS**. Avis NVIDIA architecture (Gemma 4), données (Nemotron 3.5 Lightning) et contre-audit (Poolside Laguna XS 2.1) : **UNKNOWN / SKIPPED** sur ce run, non pas PASS. Consensus 9/9 : **UNKNOWN**, réservé à workflow_dispatch. Le succès des audits représente le respect du protocole de mesure, PAS un gain musical.

## Verdict

**S805 terminé scientifiquement, run PASS, aucun progrès promouvable sur les six métriques.** Échec des trois nouvelles politiques face au veto. S708 reste officielle et LIVE strict sans latence supplémentaire. Les nouveaux apprentissages OOF à partir des seules 136 observations de fausses/vraies attaques ne généralisent pas assez aux compositions. Ne pas réutiliser les mêmes probabilités en boucles de seuils FIT, car cela entraînerait une sélection biaisée.

## Prochaine action

Prioriser une nouvelle famille **de signaux acoustiques indépendants** : séparation de fondamentales et partiels, signatures temporelles d'attaque, chroma/énergie dans différents canaux filtrés et discrimination harmonique indépendante, sans fenêtre future. Avant nouvelle promotion, ablations FIT indépendantes, essais CAL32 réservés au dernier gel du modèle, scores six-métriques, audits 9/9 réels et test matériel iRig de latence. Les six agents Work personnalisés n'ont pas été exécutés comme six sessions : ce sont six audits CI déterministes.
