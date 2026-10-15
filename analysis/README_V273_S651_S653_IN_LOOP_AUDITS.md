# S651–S653 — Audit obligatoire À L'INTÉRIEUR de chaque boucle NOTE

Branche : `codex/s651-audit-inside-loops`.

## Problème corrigé

Les cycles S642–S650 ont parfois signalé des vraies notes
« supplémentaires » avant de détailler les régressions. Un
simple `delta TP = +8` ne dit pas si ce sont huit notes
récupérées et zéro perdue, ou cinquante récupérées contre
quarante-deux notes historiques perdues.

Le but de cette branche n'est **pas de générer un nouveau
score en changeant les poids** mais de faire de l'audit
une partie obligatoire de toute boucle de proposition.

## Garde-fou 1 — S651, dès la fin de CHAQUE variante

`src/causal_note/s651_loop_guard.py`

Chaque essai conserve en mémoire et dans un rapport JSON
immuable : identification du corpus et de la source,
état de référence / état candidat, TP/FP/FN/F1,
gain F1 par source, sources en régression, faux positifs
nets par vrai positif net, ventilation solo/accompagnement,
motifs exacts de refus. Les source-level TP+FN doivent
rester invariants ; F1 est recalculé depuis les nombres
bruts. Si aucune amélioration sur les **deux** corpus,
si le budget FP est dépassé ou si trop de sources
régressent, l'option est rejetée, mais conservée
dans le registre des variantes.

Pour une promotion : résultats TRAIN/CALIBRATION avant DEV,
pas de modification des poids originaux, test réellement
indépendant, vraie preuve événementielle et latence
LIVE physique. Le garde-fou de recherche n'autorise
jamais à lui seul une promotion.

### Validation S651

[Run GitHub Actions S651 corrigé, réussi](https://github.com/Andriamarosoa/note/actions/runs/38823396602),
[artefact de toutes les variantes refusées](https://github.com/Andriamarosoa/note/actions/runs/38823396602/artifacts).

Les quatre familles d'essais S642, S644, S645 et S648 ont
été rejouées depuis leurs **rapports figés d'origine**.
Le registre contient 3 variantes S642, 10 essais S644,
10 essais S645, 10 essais S648, soit **33 évaluations
individuelles** réparties entre TRAIN/CALIBRATION et DEV.

Les verrous originaux ont tous été conservés sur
S614. La preuve est une analyse des mêmes données
que précédemment ; pas une validation musicale
indépendante ou un nouvel F1.

Exemples S648 sur GuitarSet DEV (source de comparaison
S614 998 TP, 585 FP, 582 FN, F1 63,10 %) :

| Boucle | Δ TP nets | Δ FP nets | FP nets par TP net | Régressions parmi 12 sources |
|---|---:|---:|---:|---:|
| q≥0,70 | +72 | +1 042 | 14,47 | 12 |
| q≥0,80 | +20 | +310 | 15,50 | 11 |
| q≥0,85 | +8 | +114 | 14,25 | 11 |
| q≥0,90 | +2 | +15 | 7,50 | 5 |

Les 11/12 régressions au q≥0,85 démontrent que
« corriger +8 TP » ne suffit pas. Il faut savoir
**quelles notes corrigées ou perdues** sont impliquées.

## Garde-fou 2 — S652, à chaque événement vrai

`src/causal_note/s652_event_round_audit.py`

La métrique NOTE officielle `event_score` (appariement
un-à-un, hauteur exacte, tolérance ±80ms) est rejouée
sur les prédictions NoteOn et les vraies annotations,
avec vérification des NoteOff et timestamps causaux.

Pour chaque référence on conserve :
- `newly_correct` — ancienne erreur maintenant corrigée ;
- `regressed_true_note` — correcte dans S614, perdue dans
  la nouvelle sélection ;
- `retained_correct` ;
- `still_missed`.

Les NoteOn faux détectés restent enregistrés
individuellement. L'audit fournit aussi les proximités
de hauteurs (±12/19/24 demi-tons), mais cela **ne
prouve PAS** que l'erreur est causée par l'harmonique.

Ventilation K1/K2/K3/K4+ selon le nombre de hauteurs
ayant une naissance dans ±20ms : **ce K local ne remplace
PAS l'Exact-K S80**. Ventilation par registre grave,
médium et aigu.

La règle `require_event_audit_before_next_round`
refuse de continuer s'il manque une source, ou si les
corrections individuelles n'ont pas été démontrées.
Les tests prouvent notamment qu'un delta TP = 0
peut cacher simultanément une correction et une
régression.

[CI combinée S651/S652 validée](https://github.com/Andriamarosoa/note/actions/runs/38823396602).

## S653 — Audit réel événement par événement

[Exécution GitHub Actions S653](https://github.com/Andriamarosoa/note/actions/runs/38823590757).

Ce workflow reprend le **checkpoint S648 original SHA256
`7613dca4aa9bc5992c96a75ab55f714d8720c55b6880ee9e223e39ea880eb4e9`**,
le vrai S614 gelé et les **17 audios DEV annotés**.
Les variantes de diagnostic 0,85 / 0,15 et 0,90 / 0,15
sont prédéfinies : aucun paramètre n'est réentraîné
ou resélectionné.

Après chaque variante et avant la suivante :
1. refaire le matching individuel ;
2. émettre les vraies corrections et les régressions
   par référence, registre et K local ;
3. enregistrer tous les FP avec leur temps et hauteur ;
4. contrôler les totaux contre le benchmark historique
   S648. Une divergence échoue la CI.

**Aucun résultat S653 chiffré ne peut être considéré
vérifié avant succès complet du workflow et du contrôle
des totaux d'origine.**

## Règle pour les boucles futures

Proposition → inférence causale inchangée → audit
individuel S652 et agrégé S651 → preuve des
correctifs ET des régressions → verdict TRAIN →
validation DEV après verrouillage → archive immuable
de l'essai accepté ou rejeté.

Toujours préserver S80/S614/S632. La référence S614
reste **69,03 % F1 historique TEST** ; les expériences
actuelles ne constituent pas un nouveau TEST indépendant.
