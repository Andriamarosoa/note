# Onset PCM256 — audit approfondi des erreurs et de la provenance

**Deux défaillances établies, non confondues :**

1. Défaillance du classifieur/sélecteur : S701 ne fait que *rescuer* des Onset sous le seuil natif. Aucun veto ne rejette les FP des attaques natives. La politique S701 originale a choisi **native**, pas S701, et S703 a échoué à ses quatre protections. Les poids sont bien entraînés à 256 et SHA-pinnés; la cadence native est réelle, ce n'est PAS un crash du code.
2. Défaillance de validité de benchmark : **FIT80 n'est pas un outer test indépendant**. Comparaison des noms originaux signés: S701 TRAIN ∩ FIT80 = **48/80 enregistrements identiques** (BN1, Funk2, Jazz1), S701 HELD ∩ FIT80 = **32/80** (Rock3, SS1), S613 TRAIN ∩ FIT80 = **20/80** (4 dans chacune des cinq compositions). Tous les 80 morceaux ont déjà été utilisés au moins comme TRAIN ou HELD par S701. Le fait que GT ne soit jamais consulté *pendant l'inférence* est vrai mais ne résout PAS le chevauchement avec l'apprentissage.

## Résultat observé du run #39716211840

| Variante | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|
| S613 natif 256 | 8099 | 4600 | 3168 | 67.5874% |
| S613+S701 (shadow) 256 | 8113 | 4603 | 3154 | 67.6563% |
| H0 S708 (poids d'origine entraînés à 1024) | 8528 | 3489 | 2739 | 73.2520% |

S701 apporte 14 TP et 3 FP, soit +0.06884 point F1. Seules 15/80 pistes changent en TP/FP/FN. L'essentiel des FP natifs vient des accompagnements **3,685 / 4,600 (80.11%)**. F1 des accompagnements 64.2002%, F1 des solos 76.9983%. SS1 + Rock3 portent **2716 / 4600 = 59.04%** des FP.

## Conséquence mathématique et code

S701 fait seulement `candidate[selected,0] = min(.999, onset_threshold+.025)` après un masque `p_onset < onset_threshold` : il **augmente** les probabilités de certaines attaques au lieu d'apprendre un veto anti-FP. Si tous les FN actuels étaient récupérés sans aucun FP supplémentaire, le F1 maximal de la variante actuelle avec 4603 FP serait **83.038%** — inférieur à l'objectif 85%.

Il faut une **tête dédiée de risque d'attaque pour rejeter des FP** avec un classifieur indépendant, recalibrée hors de l'OUTER; le gain de rétablissement S701 ne suffit pas. Le mécanisme de seuil, le réarmement après release et le rôle des harmoniques doivent être audités à l'échelle événement/note, car l'artefact FIT80 actuel ne conserve que des comptes agrégés et **ne permet pas de prouver quelle fréquence/quel harmonique a causé chaque FP**.

## Prochain protocole scientifiquement valable

- Geler modèles, configurations et listes TRAIN/CAL **avant** l'OUTER. Utiliser une composition et idéalement des joueurs absents des données d'apprentissage et de sélection, ou un corpus réellement externe tel qu'IDMT, puis les scorer avec les tolérances exactes.
- Appliquer un OOF au niveau des compositions/prises, *pas* un apprentissage du veto FP sur les 80 FIT80 déjà vus puis un score déclaré généralisation sur FIT80.
- Continuer de mesurer l'Onset uniquement : 256 vrais échantillons par inférence, pas de 1024, pas de lookahead, pas de retard artificiel, autres cinq axes et MIDI officiel inchangés.
- Conserver le seuil S708 de 73.2520% uniquement comme **référence historique**, pas comme étalon prouvant la généralisation des nouveaux modèles.

**Preuves** : S701 [entraînement #39085408073](https://github.com/Andriamarosoa/note/actions/runs/39085408073) et [artefact #12121062763](https://github.com/Andriamarosoa/note/actions/runs/39085408073/artifacts/12121062763), S613 `analysis/evidence/v273-s613-strict-live/protocol.json` et `scripts/s613_train_strict_live.py`, S701 + S613 [évaluation #39716211840](https://github.com/Andriamarosoa/note/actions/runs/39716211840) et [artefact #12399607233](https://github.com/Andriamarosoa/note/actions/runs/39716211840/artifacts/12399607233). Toutes les listes exactes sont figées dans `training_ledger.json`.
