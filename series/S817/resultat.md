---
serie: S817
statut: en_cours
type: experience
reference: S708
branche: codex/note-s817-activek-temporal-highpoly
run_id: en_attente
commit_sha: en_attente
promotion: false
---
# S817 — Résultats spécialistes hautes polyphonies

## Exécution

Tests et pipeline authentique GuitarSet FIT80 à exécuter. La nouvelle tête utilise de la mémoire de caractéristiques audio déjà reçues mais n'ajoute ni lookahead ni hold. Aucun résultat officiel nouveau n'est déclaré avant le run scientifique complet sur 80/80 pistes.

## Métriques

| Caractéristique | Référence S708 officielle CAL32 | S817 nouveau FIT80 |
|---|---:|---|
| Onset F1 | 74,2639 % | non modifiée |
| Exact-K | 62,0455 % | non modifiée |
| Active Exact-K | 56,2762 % | non mesurée |
| Offset accuracy | 77,3292 % | non modifiée |
| Full Note F1 | 57,1319 % | non modifiée |
| Frame Pitch F1 | 76,4626 % | non modifiée |

Sur la même cohorte FIT80 déjà explorée, H0=57,01953803 %, S815=56,91248582 %, S816=56,42413526 %. S817 doit reconstruire H0/S815 avant toute comparaison, puis détailler vrais K1–K6, sous/surcomptages et métriques positives.

## Comparaison à S708

Référence historique S708 et S707 inchangées, aucune émission MIDI de production, aucune unification. Les nouvelles têtes utilisent uniquement des signaux acoustiques courants/passés, leur propre mémoire et la prédiction privée d'Active-K S815. Il n'y a pas de partage de poids avec Frame Pitch, Exact-K, Onset, Offset ni Full Note.

## Audits

| Axe | Statut | Preuve/limite |
|---|---|---|
| Données | UNKNOWN | Exiger les 80 WAV/annotations originaux SHA vérifiés |
| Acoustique | PASS | Contrat causal des échantillons 256 et fenêtres antérieures seules |
| Architecture | PASS | Deux spécialistes propres et S815 gelé sur branche de recherche |
| Apprentissage | UNKNOWN | Entraînement réel sur 5 partitions 48/16/16 à réaliser |
| Métriques | UNKNOWN | Vérification complète S704 et vrai score MIDI256 non réalisés |
| LIVE | UNKNOWN | Aucune latence physique iRig ou CPU total validée |

## Verdict

Aucun score >85 % ni nouvelle performance revendiquée. Les protections strictes bloquent promotion et unification jusqu'aux preuves indépendantes.

## Prochaine action

Exécuter la CI, valider l'architecture sur les données réelles, documenter toute régression et tester sur cohorte indépendante seulement si un progrès FIT80 stable est démontré.
