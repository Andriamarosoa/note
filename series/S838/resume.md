---
serie: S838
statut: en_cours
type: experience
reference: S708
branche: codex/note-s838-fragment-risk-gates256
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S838 — Sélection apprise PAR FRAGMENT des corrections de K

## Objectif
Continuer les boucles d'amélioration des six caractéristiques de NOTE vers **plus de 80 %** avec la priorité immédiate Exact-K et Active Exact-K, actuellement sous 80. Préserver S837 et les autres axes, et corriger **l'échec de généralisation des politiques fixes** en entraînant un sélecteur indépendant à chaque décision PCM256, sans retard ni futur.

## Hypothèse
Le S837 #39997037972 démontre une régression Exact-K de 66 blocs corrects nets (Jazz1 −22, Rock3 −44) et une régression Active-K de 254 blocs corrects nets (BN1 −112, SS1 −194 malgré Jazz1 +31 et Rock3 +21). Une politique constante par composition ne choisit pas le bon expert pour chaque fragment. Un petit arbre de décision appris à partir de la confiance K0–K6 et des observations causales Onset/Sustain/Release **de la propre banque acoustique source en lecture seule** peut accepter les corrections là où le témoin est faible et rejeter les contre-exemples, sans jamais regarder la vérité du futur ni les décisions d'un autre axe.

## Modifications
- Branche de recherche S838 distincte de S837 ; aucun pipeline officiel Live, aucun modèle natif S708, aucun des six décodeurs de production ni aucune sortie MIDI modifiés.
- `scripts/s838_fragment_risk_gates256.py` : pour chaque vraie piste, une inférence S836/S708 gelée par frontière 256 ; extraction privée de 30 caractéristiques de confiance/sustain/attaque/release **du fragment courant**, pas de futur ni de notes GT en entrée.
- Exact-K : banque d'experts du posterior K original et de quatre seuils de soutiens causaux.
- Active Exact-K : banque d'experts privés avec hystérésis indépendante par variation `on/off`, état propre à chaque expert, jamais l'état d'Exact-K ou d'une autre tête.
- Selector K : `DecisionTreeClassifier(max_depth=4, min_samples_leaf=90)` pour identifier quelle alternative corrige un K faux du témoin ; seuil de risque calibré sur composition CAL et recours au témoin par défaut. Aucun `sleep`, hold, calcul sur prochain bloc ni relecture d'une attaque.
- Split fixe par OUTER : **trois compositions TRAIN, une CAL, une OUTER**, sans lecture de la composition OUTER pour entraîner l'arbre ni choisir sa politique.
- `test/test_s838_fragment_risk_gates256.py` : causalité de l'entrée, exactitude du compte par référence, veto candidat incertain, récupération/perte par vrai K0–K6.
- Rapports `series/S838/resume.md` et `resultat.md`, tests et GitHub CI isolés.

## Protocole
- 44 100 Hz, exactement **256 nouveaux échantillons par décision**, 5,805 ms de cadence, zéro audio futur, zéro hold et aucun mécanisme de prédiction différée.
- Chaque tête possède sa banque d'experts, son entraînement et son sélecteur privés. La seule entrée partagée est le snapshot acoustique causal S708/NOTE en lecture seule ; **aucune sortie d'une tête alimente une autre tête**.
- Les comptes issus du S834 original doivent être reproduits strictement sur chaque piste avant de comparer ; ne jamais déplacer le témoin pour simuler un gain.
- Première preuve : les cinq vraies pistes GuitarSet FIT80 signées, chacune d'une composition distincte, 20 062 blocs PCM256. Les 75 autres pistes et les solos restent non vérifiés sur cette série.
- La sélection de la meilleure confiance `THRESHOLDS` se fait exclusivement sur CAL. OUTER n'intervient qu'après le choix gelé de la politique.
- Audits par composition et vrai K0–K6 : blocs récupérés/perdus, surcomptage/souscomptage, cas où les deux experts se trompent, liste des échantillons. Chaque perte entraîne une nouvelle hypothèse testable, pas un oubli.
- Aucune métrique CAL32 mélangée avec FIT80. Exact-K / Active-K en accuracy de trames. Ne jamais confondre ces scores avec Onset/Offset F1 ni Full Note F1 ou prétendre un MIDI unifié.
- Le modèle S708 a historiquement été entraîné à pas 1024. Cette série mesure **une inférence causale 256**, et non un vrai nouvel entraînement de la source S708 à 256.
- Les propres champs de `FIT80` ont déjà été explorés historiquement ; sans nouveau test externe indépendant, pas de prétention de généralisation ni de promotion.
- Si gain isolé, conserver le témoin et ses scores ; ne pas fusionner les têtes sans tests des six scores + budget p95/p99 physique Windows/iRig.

## Audits obligatoires
| Rôle | Protocole |
|---|---|
| Données | SHA source WAV original, composition disjointe TRAIN/CAL/OUTER |
| Acoustique | cadence 256 exacte, aucune caractéristique du futur |
| Architecture | deux sélecteurs K privés, quatre autres axes inchangés |
| Apprentissage | arbre entraîné sur TRAIN uniquement, seuil CAL seulement |
| Métriques | par vrai K0–K6, recouvrement et régressions, OUTER gelé |
| LIVE | absence d'attente logique et budget CPU 5,805 ms à mesurer sur matériel |

## Statut
En cours et non promu. Cette boucle répond directement aux régressions S837 mais ne promet pas un passage automatique au-dessus de 80 %. Les quatre axes non expérimentés (Onset, Offset, Frame Pitch, Full Note) restent gelés.
