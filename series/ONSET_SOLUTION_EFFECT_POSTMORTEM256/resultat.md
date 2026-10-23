# Audit constaté — HGB arbitre PCM256, run 39957354552

## Verdict signé

**SOLUTION REJETÉE : 69,2584331 % -> 69,2489759 % F1, TP 7607 -> 7607, FP 3093 -> 3096, FN 3660 -> 3660.** Le replay exact de 80 WAV / 306536 blocs PCM256 est passé et le garde-fou scientifique a arrêté la promotion.

Source : https://github.com/Andriamarosoa/note/actions/runs/39957354552
Archive immuable : artifact ID 12506429382, ONSET_LEARNED_HGB_ARBITER_80_WAV_PCM256 (fichiers prepared.json, trained/*/calibration.json, frozen_gate.json, outer_result.json, replayed/report.json).

## Pourquoi la nouvelle solution régresse : diagnostic chiffré

L'arbitre n'a effectivement ajouté que **3 événements** ; les trois sont des faux positifs. Aucun vrai positif supplémentaire. Les événements supplémentaires se concentrent sur :
- BN1 : piste 03_BN1-129-Eb_comp, +1 FP.
- Rock3 : piste 01_Rock3-148-C_comp, +2 FP.
- Funk2/Jazz1/SS1 : aucune modification de sortie.

**La cause structurelle identifiée est le déséquilibre des occasions de correction selon les compositions**, non un problème de latence ou de parité Live. Pour un total de 12 281 propositions HGB, seuls 802 événements ont été étiquetés « HGB correct pouvant ajouter un TP que l'acoustique a raté ». Ils sont répartis comme suit :

| Composition | HGB candidats | Propositions positives récupérables | Prévalence |
|---|---:|---:|---:|
| BN1 | 1 819 | 0 | 0,000 % |
| Funk2 | 2 364 | 4 | 0,169 % |
| Jazz1 | 1 723 | 0 | 0,000 % |
| Rock3 | 3 833 | 631 | 16,462 % |
| SS1 | 2 542 | 167 | 6,570 % |
| Total | **12 281** | **802** | **6,530 %** |

Les 798/802 = **99,5 %** des occasions récupérables se trouvent sur Rock3 et SS1.

## Contre-exemples aux décisions de CAL

Tableau tiré des checkpoints figés et des rapports de calibration/OUTER. TRAIN, CAL et OUTER sont séparés par composition.

| OUTER | Positifs disponibles TRAIN/CAL/OUTER | Gain CAL TP / FP | Gain OUTER TP / FP | Statut du gate |
|---|---:|---:|---:|---|
| BN1 | 798 / 4 / 0 | +3 / +2 | 0 / +1 | activé : faux positif |
| Funk2 | 798 / 0 / 4 | 0 / 0 | 0 / 0 | désactivé |
| Jazz1 | 171 / 631 / 0 | +266 / +483 | 0 / 0 | activé mais aucun événement OUTER |
| Rock3 | **4 / 167 / 631** | +1 / 0 | **0 / +2** | activé : faux positifs |
| SS1 | **635 / 0 / 167** | 0 / 0 | 0 / 0 | désactivé |

Interprétations étayées :
1. **Rock3** est le plus grand réservoir de TP potentiellement récupérables (631), mais son modèle n'a été entraîné qu'avec **4 exemples positifs** d'autres compositions ! Le gain minime sur SS1 en CAL (+1 TP) ne se transfère pas à Rock3 (+2 FP, 0 TP). C'est une couverture positive TRAIN insuffisante pour cet OUTER, pas une preuve physique sur les harmoniques.
2. **SS1** dispose de 167 occasions, mais son CAL BN1 possède **0 exemple positif**. Le protocole choisit donc de désactiver la tête entière et ne récupère aucune attaque SS1.
3. **Jazz1** gagne +266 TP / +483 FP sur le CAL Rock3, mais son OUTER Jazz1 ne possède **aucune occasion positive**. La calibration positive n'indique donc pas d'amélioration transférable à Jazz1.
4. **BN1** voit +3 TP / +2 FP sur CAL Funk2 mais zéro occasion positive OUTER, puis +1 FP. La correction ne peut mathématiquement apporter de TP sur son OUTER avec ce réservoir de propositions.
5. Les choix CAL globaux sur une seule composition sélectionnent un régime ne représentant pas nécessairement l'OUTER, en particulier lorsque la prévalence d'attaques récupérables passe de 0 à 16,5 %.

**Limite du mécanisme HGB :** même si l'arbitre récupérait idéalement les **802** occasions positives sans FP supplémentaire, le F1 passerait seulement à **73,864 %** sur le témoin réentraîné (borne conditionnelle, pas mesure d'un vrai modèle). Pour viser 80 %, il faut chercher des attaques absentes du réservoir HGB, pas seulement mieux filtrer ses propositions.

## Causes non prouvées et prochaine expérience

Le détail acoustique de chacun des trois FP n'est pas suffisamment présent dans ces rapports pour conclure à une fausse fondamentale harmonique ou une réattaque erronée. Les identifiants de piste sont prouvés, pas encore les causes physiques des FP.

Hypothèses testables :
- Refaire un apprentissage avec des exemples récupérables représentatifs du type d'accompagnement polyphonique, sans contaminer OUTER.
- Ne plus choisir une politique de rétablissement sur un CAL à zéro positif puis l'appliquer à un OUTER riche en positifs. Exiger une validation stratifiée couvrant les erreurs ciblées, **tout en préservant un vrai jeu d'évaluation non utilisé pour choisir les politiques**.
- Rendre au moins deux sources de propositions accessibles : HGB filtré ET nouvelles attaques acoustiques hors candidat HGB, sinon le plafond conditionnel ~73,9 % subsiste.
- Mesurer pour chaque hypothèse les paires TP gagnés / TP perdus, FP ajoutés / FP enlevés et les contre-exemples par piste, sur la même source PCM256. Isoler les autres têtes.

**Ne pas modifier le MIDI officiel, ne pas contourner le garde-fou, zéro attente supplémentaire et aucun score fictif.**

