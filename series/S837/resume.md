---
serie: S837
statut: en_cours
type: experience
reference: S708
branche: codex/note-s837-six-axis-80pct-independent-loops256
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S837 — Boucles indépendantes des six caractéristiques vers plus de 80 %

## Objectif
Obtenir **plus de 80 %** pour chacune des caractéristiques Exact-K, Active Exact-K, Onset, Offset, Frame Pitch et Full Note, sur la **même cohorte**, tout en maintenant une architecture Live MIDI256 réellement exécutable sans attente. Ne pas faire croire que six scores SHADOW constituent déjà une sortie MIDI unifiée, et ne jamais déclarer de réussite avant résultat complet.

## Hypothèse
Les seuils/cadences de réarmement gelés dans les six décodeurs S834 ne sont pas optimisés pour les nouveaux postérieurs S708 recalculés toutes les 256 valeurs. Une sélection indépendante de politiques causales par axe sur d'autres compositions GuitarSet peut corriger des erreurs, mais peut aussi produire des régressions dues au changement de distribution, à la polyphonie et aux attaques harmoniques. Toute correction faible ou négative exige une autopsie et n'est jamais promue automatiquement.

## Modifications
- Branche S837 indépendante depuis S836 : moteur microphone historique S613/S614, source S708, modèle strict LIVE, six autres axes et MIDI officiel non modifiés.
- `scripts/s837_six_axis_policy_loops256.py` : une seule inférence acoustique S836 cachée PCM256 sur chaque WAV réel signé ; cache de fragments [49,3] + K0–K6. Chaque politique utilise ensuite uniquement son **décodeur privé**. Aucune sortie d'un axe n'est injectée dans un autre.
- Grilles bornées de seuils, de réarmement et de support : Exact-K (distribution K source + option alternative sur nombre de sustains), Active Exact-K, Onset, Offset, Frame Pitch et Full Note.
- Politique pour chaque composition OUTER sélectionnée uniquement avec les **quatre autres compositions**; aucune étiquette OUTER dans sa sélection. Reporting de toutes les politiques et de toutes les régressions ; contrôle que `reference` reproduit exactement S834 sur chaque piste avant de sélectionner quoi que ce soit.
- Pour Onset/Offset : audits d'identité par appariement hongrois des vraies attaques et fausses attaques, exemples récupérés/perdus et FP nouvelles/supprimées, grave/médium/aigu, association possible avec fondamentales graves présentes ; association harmonique ≠ preuve causale.
- `test/test_s837_six_axis_policy_loops256.py` : référence par axe, conservation des têtes, absence de fuite OUTER, preuves des changements événementiels.

## Protocole
- 44,1 kHz, un bloc de décision tous les **256 nouveaux échantillons** (5,805 ms), zéro future audio, zéro hold, aucune file temporisant l'attaque, pas de chemin GPU nécessaire.
- Données : exactement les cinq compositions GuitarSet FIT80 déjà explorées, d'abord cinq WAV réels (une piste par composition), puis les **80 WAV complets** si le test d'intégrité et les contrôles le permettent. TRAIN/CAL/OUTER protégés d'une influence directe OUTER pendant le choix des règles. Les cinq sources tests ne forment pas une preuve externe.
- Métriques : Exact-K et Active Exact-K en accuracy, Onset F1 et Offset événement F1 comme mesures **différentes de l'Offset accuracy S708**, Frame Pitch F1 et Full Note F1. Les ratios ne doivent pas être mélangés entre CAL32/FIT80, SHADOW/final MIDI, PCM1024/256.
- Algorithme de sélection : pour chaque composition, choisir le meilleur score agrégé des autres compositions parmi la grille **préfixée dans le code** ; pas d'adaptation au score OUTER. Les modèles gelés n'ont jamais reçu les annotations du flux courant.
- Tous les états de la tête et du décodeur sont privés. Comparer à S834 sur chaque piste avant toute mesure de changement ; veto si une des six sorties de référence diverge.
- À chaque tentative régressive : trouver TP récupérés/perdus, FP ajoutés/retirés, sous/surcomptages K, registre, solo/accompagnement ; comprendre *pourquoi* avant de lancer une autre piste.
- Les modèles S708 d'origine ont été entraînés à une cadence effective 1024 ; rejouer les postérieurs toutes les 256 valeurs **n'établit pas** qu'ils sont entraînés natifs 256.
- Critère 80 % : chaque score individuel >0,80 sur le même protocole, comparé de façon appariée ; ne pas promouvoir un candidat qui régresse une autre tête. Unités et contrôles CPU matériel p95/p99 impératifs pour l'intégration Live.
- Aucune modification du MIDI officiel avant que les six performances soient non régressives et le Live physique validé.

## Audits obligatoires
| Rôle | Attendu |
|---|---|
| Données | sha256 originaux et 5/5 compositions puis 80 WAV |
| Acoustique | PCM256, échantillons présents exclusivement |
| Architecture | six décodeurs isolés, zéro influence inter-axes |
| Apprentissage | source gelée ; politiques sélectionnées hors composition cible |
| Métriques | F1/accuracy avec dénominateurs et événement identique |
| LIVE | zéro temps ajouté ; p95/p99 matériel et source de latence |

## Statut
Série en cours et non promue. L'objectif >80 % sur les six caractéristiques est un objectif de recherche, **pas une promesse de gain**. Si les politiques restent sous 80 %, conserver le diagnostic détaillé et cibler les causes structurelles (absence de propositions, calibration ou apprentissage) plutôt que d'afficher un faux record.
