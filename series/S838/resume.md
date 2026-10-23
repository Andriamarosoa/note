---
serie: S838
statut: termine
type: experience
reference: S708
branche: codex/note-s838-fragment-risk-gates256
run_id: 40000089112
commit_sha: 4448f460a656a3c524114f18a6b7385e96fb74f4
promotion: false
---

# S838 — Sélection apprise PAR FRAGMENT : boucles K terminées, non promues

## Objectif
Continuer vers >80 % sur **les six caractéristiques NOTE** en traitant d'abord les régressions K de S837, sans perdre les plus petites corrections, sans audio futur ni attente ajoutée. Les six modèles ne sont pas fusionnés ; l'objectif musical ne peut être prétendu atteint avant six scores supérieurs à 80 sur la même métrique/cohorte.

## Hypothèse
Le choix d'une politique uniforme par composition peut réussir sur un type de fragment mais créer plus de pertes sur d'autres. Un **arbre privé de sélection du meilleur expert à chaque fragment 256**, entraîné sur trois compositions, calibré sur la quatrième et évalué uniquement sur OUTER, devrait récupérer de petites corrections sans sacrifier tous les K0–K6.

## Modifications
- `scripts/s838_fragment_risk_gates256.py` : entrée acoustique S708 vraie à chaque PCM256, caractéristiques causales privées 30D, banques K privées, décision arbre de risque pour chaque fragment, seuil fixé uniquement sur CAL, sans sortie d'une autre caractéristique.
- `test/test_s838_fragment_risk_gates256.py` et workflow GitHub CI : **36 tests PASS** sur [run #40000089112](https://github.com/Andriamarosoa/note/actions/runs/40000089112).
- `series/S838/audit_loop1.md` : explication de chaque régression du gate Exact-K SS1 et de chaque petite correction Active-K, catégories K0–K6.
- Source S708, six caractéristiques, moteur microphone et MIDI officiel demeurent **inchangés**. Uniquement des sorties SHADOW pour deux têtes de nombre.

## Protocole
- 44 100 Hz, décisions sans attente toutes les **256 valeurs** = 5,805 ms ; zéro hold, zéro future PCM et aucune entrée GT au runtime.
- Original GuitarSet FIT80 déjà exploré, cinq vrais WAV `_comp`, un par composition, **20 062** blocs ; ne pas traiter ces scores comme preuve externe HELD ou résultats 80 pistes.
- Pour chaque OUTER, entraîner sur 3 compositions disjointes, calibrer le seuil sur une quatrième, évaluer OUTER après choix verrouillé. Aucun événement des autres têtes ne sert d'entrée au sélecteur K.
- Le modèle-source S708 n'a pas été réentraîné à 256, il a été appris sur une cadence historique 1024. Non-régression obligatoire des six métriques à protocole apparié, puis budget CPU sur Windows/iRig avant toute promotion.
- Après toute petite progression ou régression, audit par échantillon et vrai K0–K6 ; conserver les exemples qui contredisent l'hypothèse, distinguer cause prouvée (mauvaise transférabilité) d'une origine harmonique encore à démontrer.
- Les quatre autres axes restent gelés. Seul Frame Pitch dépassait déjà 80 % sur S837 et cette nouvelle boucle n'autorise pas de toucher à son résultat.

## Audits obligatoires
| Rôle | État |
|---|---|
| Données | PASS vraie musique 5 compos et SHA |
| Acoustique | PASS causal PCM256 |
| Architecture | PASS têtes K séparées, quatre autres inchangées |
| Apprentissage | PASS TRAIN/CAL/OUTER séparés ; source préentraînée 1024 |
| Métriques | FAIL un K régresse, autre +0,47 point seulement |
| LIVE | UNKNOWN p99 Windows/iRig et six têtes unifiées |

## Statut
Série expérimentale terminée, non promue. **Exact-K 56,599541→55,926627 % (−135 blocs), Active Exact-K 46,772514→47,241020 % (+90 blocs)** sur cinq WAV. Un petit gain Active-K constitue une piste de nouveau sélecteur, pas une performance de production. [Run #40000089112](https://github.com/Andriamarosoa/note/actions/runs/40000089112) SUCCESS, artifact `S838_REAL5_PRIVATE_K_FRAGMENT_GATES_TRAIN_CAL_OUTER` ID `12521936945`. La prochaine correction doit améliorer l'identification des risques K1/K4 qui régressent sur SS1, sans sélectionner a posteriori sur cette composition OUTER.
