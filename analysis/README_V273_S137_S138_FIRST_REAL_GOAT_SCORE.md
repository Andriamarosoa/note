# S137–S138 : premier mini-test Exact-K externe — verdict négatif, reproductible

**Branche** : `codex/v273-scope-rescue`.

## Ce qui a réellement été testé

- **Nouvelle alternative S135** : modèle étudiant 111D qui imite les
  décisions du S5 original mais ne possède pas les caractéristiques
  `owned527`, ni le reste du pipeline S80 original.
- **Référence** : checkpoint officiel `freeze_local_combo`, sans
  changement.
- **Deux extraits GOAT seulement**, échantillons indépendants des
  entraînements GuitarSet ; uniquement leurs premières **6 secondes**.
- **57 groupes natifs identiques**, détectés uniquement à partir de
  l'audio original V8.4/V8.6/V9.0.
- Entrées et horodatages des deux prédicteurs joints strictement par
  identifiant natif `row_id=member:start_sample`, K dans [0,6].
- Deux jobs CI successifs : **sceller et hasher les prédictions sans MIDI**,
  puis **ouvrir les annotations** et comparer sur les mêmes groupes.
- **SHA-256 du fichier scellé** :
  `620ef28fd2fb4cd7ca6190432c460361e4da7fb1ffda064770390953360a8f02`.
- Correspondance annotée fixée avant labels : chaque `note_on` MIDI
  assigné au groupe [start, end] le plus proche à **20 ms maximum**
  (882 échantillons à 44,1 kHz), un seul groupe par note, plafond K6.

## Résultat réel S137

[Workflow S137, réussi — run #38191739817](https://github.com/Andriamarosoa/note/actions/runs/38191739817).

| Mesure | Référence | S135 étudiant |
|---|---:|---:|
| Correct Exact-K global | **44/57 (77,19 %)** | **44/57 (77,19 %)** |
| Correct sur les vrais K≥2 | **2/3** | **1/3** |
| Corrections / régressions vs référence | — | 1 / 1 |
| Bilan net | — | **0** |
| Vecteurs K différents | — | **3** |

Distribution de vérité issue du MIDI sur les **57 groupes** :
**K0 = 49, K1 = 5, K2 = 3, K3–K6 = 0**.
Les 57 groupes couvrent la même observation audio ; ce n'est pas
l'évaluation de chaque onset de l'enregistrement complet.

**Verdict** : la nouvelle alternative S135 ne bat PAS la référence
et perd un cas polyphonique sur trois.
Aucune promotion autorisée ; l'ancien S80 historique n'a pas été
exécuté de bout en bout sur GOAT.

## Pourquoi ce score est peu informatif : audit S138

[Workflow S138, réussi — run #38191885587](https://github.com/Andriamarosoa/note/actions/runs/38191885587).

| Sous-ensemble | Groupes audio | Note-ons MIDI dans les 6 s | Note-ons capturés à 20 ms |
|---|---:|---:|---:|
| GOAT example1 | 3 | 2 | 2 |
| GOAT example2 | 54 | 12 | 9 |
| **Total** | **57** | **14** | **11 (78,57 %)** |

**3 débuts de note** échappent au détecteur de groupes audio ;
ils ne peuvent pas être comptés correctement par une tête Exact-K
qui ne reçoit que les groupes détectés. La majorité des groupes
(49/57) est de vrais K0, ce qui gonfle les taux globaux. Les notes
MIDI au-delà de 6 s ne sont **pas** incluses dans cette mesure
de couverture ; il est faux d'interpréter les 998 notes du MIDI
complet comme des ratés de cette fenêtre de six secondes.

## Critères de non-promotion

1. **Pas de gain global : 0.**
2. **Baisse polyphonique : 2/3 → 1/3.**
3. Seulement 3 groupes réellement polyphoniques et 2 morceaux ;
   données insuffisantes pour une décision de généralisation.
4. S135 est un **nouveau modèle étudiant** et non l'original S80.
   Les producteurs S45/S46/S69/S72 de S80 restent incomplets sur
   un enregistrement extérieur.
5. Toute modification future guidée par ces labels rendra GOAT
   **exposé**, et ne pourra plus servir de test indépendant final.
   Pour l'étape suivante utiliser le corpus
   [IDMT-SMT-GUITAR_V2](https://zenodo.org/records/7544110)
   préenregistré, avec nombre élevé de groupes polyphoniques
   et prédictions scellées avant l'accès aux annotations.

### Décision de recherche

**REJET explicite de S135 comme remplacement de la référence sur
la preuve S137.** Conserver les modèles S133–S135 et les résultats
négatifs pour audit, mais ne plus déclarer la fidélité au professeur
une amélioration d'Exact-K réel.

**Le chantier principal à venir** : une nouvelle version réellement
exécutable de S80 sur des morceaux inédits, avec les 16 producteurs
S43/S44/S45, leurs risques S46 et les règles de déploiement sans
fold historique pour S69/S72/S80. Cette nouvelle version devra être
gelée puis testée sur une autre population audio annotée.

L'original `freeze_local_combo` reste **inchangé**.
