# S703 — apprentissage séparé des attaques natives et des notes manquées

## Pourquoi cette seconde boucle

S702 a terminé mais sa tête commune a régressé : 63,1046 % → 62,1711 % de F1
Onset DEV. Elle a récupéré 10 références et en a perdu 63. Le correctif de
réarmement fonctionne, mais S701 ne perdait aucune référence vraie sur ce
HELD/DEV : ce défaut logiciel ne suffit pas à expliquer le plateau.

Une autre difficulté se trouve dans la cible de la tête commune. Si la vraie
attaque est à 200 ms, une attaque native à 240 ms est correcte à ±80 ms. Un
candidat ajouté à 201 ms est aussi admissible. Apparier tous les candidats
ensemble choisit 201 ms et étiquette 240 ms comme négatif. Ce label peut
ensuite apprendre au filtre à rejeter une attaque native pourtant correcte.
Ce mécanisme a été reproduit dans un test ; son poids dans les régressions
réelles n'est pas quantifié séparément.

## Modifications

- Une tête native apprend les TP/FP de la sélection native seule. Une
  proposition concurrente ne change pas ces étiquettes.
- Une tête résiduelle apprend à récupérer les références manquées par cette
  sélection native. Les annotations restent limitées à l'apprentissage.
- Les états natifs et proposés restent indépendants, comme dans S702.
- S675 reçoit l'horloge rythmique issue des événements natifs causaux pour
  chaque variante. Une suppression ou un ajout expérimental ne change donc
  pas cette entrée commune du modèle Offset.
- La calibration choisit le F1 sous contraintes TP et Exact-K. La précision
  et le rappel restent rapportés séparément ; le seul ratio de précision
  ne remplace pas l'objectif F1.

## Validation

Même protocole que S702 : 64 prises FIT, 16 CAL, 64 HELD et 12 DEV. Quatre
familles TRAIN servent à l'apprentissage, une cinquième à la calibration,
quatre autres à HELD. Les composants natifs et S701 ont leur exposition
historique propre. HELD et DEV restent des données de recherche déjà vues ;
TEST n'est pas ouvert et aucune promotion automatique n'est effectuée.

Quatorze tests couvrent les cibles, le réarmement, les six places de notes,
la parité native et l'invariance des préfixes causaux. Le benchmark contrôle
le F1 Onset avec bonne hauteur à ±80 ms, Exact-K décodé avec release native,
S675 à ±50 ms, le F1 note complète et les références longues perdues.

Le résultat S675 est un replay shadow distinct. Aucune certification de
latence ou de précision sur microphone réel n'est déduite de ces mesures.

Run : https://github.com/Andriamarosoa/note/actions/runs/39090842020

## Résultats vérifiés — 18 octobre 2026

Le run a terminé avec succès en 890,39 s de calcul. Les 14 tests ont réussi.
Les TP/FP/FN des 64 prises HELD et des 12 prises DEV ont été recomptés depuis
les événements JSONL, puis comparés au rapport du workflow. Le SHA-256 du ZIP
est `da9e3273043c3208f836413358da1fe0cfd888150e55baf7f6b2c6548da05abb`.

Apprentissage : 12 127 événements candidats, dont 8 019 natifs et 4 108
résiduels. La tête native a 5 643 exemples positifs ; la tête résiduelle 521.
La politique `event_native_0.2_rescue_0.35` est choisie sur CAL (69,82 % →
71,89 % de F1) avant HELD et DEV ; elle n'est pas retouchée ensuite.

| GuitarSet DEV — même protocole causal | Natif | S703 | Écart |
|---|---:|---:|---:|
| F1 Onset + hauteur, ±80 ms | 63,1046 % | 63,6709 % | +0,5662 point |
| TP / FP / FN | 998 / 585 / 582 | 1 006 / 574 / 574 | +8 TP / −11 FP |
| F1 Onset, morceaux avec accords | 58,6134 % | 59,0696 % | +0,4562 point |
| F1 Onset, morceaux solo | 76,1084 % | 76,8666 % | +0,7582 point |
| Exact-K décodé, release native | 54,4721 % | 55,4074 % | +0,9353 point |
| Offset conditionnel S675, ±50 ms | 78,0808 % (773/990) | 77,8557 % (777/998) | −0,2251 point |
| F1 note complète | 48,8776 % | 49,1772 % | +0,2996 point |

DEV : 24 références récupérées et 16 perdues, soit +8 TP net. Les pertes
concernent 14 notes médiums et 2 graves. Aucune note longue de DEV n'est perdue.
Sur les 973 événements appariés strictement identiques, aucune fin S675
correcte ne devient incorrecte à la tolérance de 50 ms. Cela ne rend pas
équivalentes l'ancienne population et la nouvelle : le ratio Offset global
conditionnel baisse malgré quatre fins correctes supplémentaires.

| GuitarSet HELD — 64 prises | Natif | S703 |
|---|---:|---:|
| F1 Onset | 67,5894 % | 68,0707 % |
| TP / FP / FN | 7 081 / 4 408 / 2 383 | 7 046 / 4 192 / 2 418 |
| Exact-K décodé | 49,9955 % | 50,6535 % |
| Offset conditionnel S675 | 67,6504 % (4 722/6 980) | 67,5886 % (4 692/6 942) |
| F1 note complète | 45,0723 % | 45,3290 % |

HELD : 141 références récupérées et 176 perdues, soit −35 TP net. Le F1 monte
par réduction des faux positifs (−216), mais le rappel régresse. Dix références
longues sont perdues. Sur les 6 793 événements appariés identiques, aucune fin
S675 correcte ne devient incorrecte à ±50 ms.

## Décision

**Gain Onset mesuré, protections complètes échouées, aucune promotion LIVE.**
L'objectif de 80 % n'est pas atteint. Les blocages sont le rappel HELD, les
10 notes longues perdues et les ratios Offset conditionnels en baisse.
Le F1 Onset progresse sur CAL, HELD et DEV, mais les nombres ne justifient pas
le remplacement de la référence sous les contraintes actuelles.

La séparation des têtes et le contexte Offset figé constituent un candidat
à poursuivre ; la suppression des attaques natives doit encore être rendue
plus fiable, surtout dans le médium. Ce constat ne prouve pas à lui seul la
cause acoustique des erreurs.

Code et bilan : branche `codex/note-s703-separate-event-heads`.
Synthèse recomptée : `analysis/evidence/s703-separate-event-heads/verified_summary.json`.
Artefact : `S703_SEPARATE_NATIVE_RESIDUAL_ONSET_HEADS`, avec modèle, événements
et audits par note. La référence LIVE officielle reste inchangée.
