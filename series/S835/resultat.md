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

# S835 — Résultat de conservation des meilleures références par caractéristique

## Exécution
Branche de conservation créée sur GitHub `codex/note-best-of-six-axes-preserved`, depuis S834. Vérification automatisée du manifeste et des six sources de code en cours. Aucune validation musicale nouvelle dans cette série.

## Métriques
Les six valeurs sont une **sélection de résultats expérimentaux déjà mesurés, et non le score d'un modèle unique** :

| Axe | Implémentation ou référence protégée | Score FIT80 connu | Autorisé à remplacer une autre tête |
|---|---|---:|---|
| Exact-K | S834 SHADOW PCM256 | 75,780659 % | NON, sans nouvelle évaluation conjointe |
| Active Exact-K | S828 SHADOW PCM256 | 73,831146 % | NON, poids requis et parité à refaire |
| Onset F1 | S708/H0 historique | 73,252019 % | NON, référence gelée |
| Offset accuracy | S708/H0 historique | 69,886229 % | NON, référence gelée |
| Frame Pitch F1 | S834 SHADOW PCM256 | 84,998850 % | NON, sans nouvelle évaluation conjointe |
| Full Note F1 | S834 SHADOW PCM256 | 71,108037 % | NON, sans nouvelle évaluation conjointe |

**S835 n'a pas de nouveau score Exact-K ou de nouveau MIDI fusionné.** Quatre des six choix sont des chemins de détection expérimentaux PCM256 ; deux sont des références historiques protégées. Par conséquent, une inférence LIVE 256 produisant six meilleures performances simultanées n'est PAS démontrée. Comparer aussi les records historiques S809 84,24 % Exact-K SHADOW1024, S823 Active-K 76,8775 % exploité depuis les snapshots 1024, S790 Frame Pitch 86,7532 % FIT80 SHADOW1024 : ces scores ne sont pas interchangeables avec les candidats PCM256.

## Comparaison à S708
S708 officiel CAL32 reste inchangé. S834 ne modifie aucun code de référence et l'ancienne sortie MIDI n'est pas réassemblée. Les scores FIT80 protégés Onset et Offset évitent de sélectionner sciemment le Onset 71,38 % S834 inférieur au H0 73,25 %, et de comparer frauduleusement Offset F1 38,68 % S834 à Offset accuracy 69,89 % H0.

## Audits
| Rôle | Statut | Preuve |
|---|---|---|
| Données | EN COURS | Vérification des références et du manifeste |
| Acoustique | PASS PARTIEL | PCM256 natif attesté pour 4 axes, deux références historiques conservées |
| Architecture | EN COURS | Gel des sources et tests anti-interférence obligatoires |
| Apprentissage | PAS D'ENTRAÎNEMENT | Registre sélectionne des sources existantes sans produire de checkpoints |
| Métriques | PASS DOCUMENTAIRE | Cohortes et unités distinguées ; aucune nouvelle mesure musicale |
| LIVE | UNKNOWN | Modèles/poids et iRig CPU p99 non validés |

## Verdict
**BRANCHE CRÉÉE, NON PROMUE**. Une seule branche pour conserver six meilleures références, sans dissimuler les deux obstacles Onset/Offset, ni s'engager dans une unification régressive.

## Prochaine action
Construire Onset et Offset strict PCM256 qui battent ou égalent leurs scores FIT80 H0 sur la même définition, sérialiser les poids de S828 et mesurer le coût CPU global. Maintenir strictement les cinq autres axes inchangés pendant chaque boucle.
