# V27.3 — Boucles S24–S26 : l'ordre inverse de la sélection faible vers la forte

## Bilan empirique
L'ordre **modifie les résultats** : ce n'est ni commutatif ni une simple permutation de masques. En revanche, une première tête faible n'améliore pas automatiquement les autres, et les trois expériences n'ont produit **aucun** garde-fou qui sauve des régressions sans perdre de bonnes prédictions précédentes.

Contexte : 59 309 événements natifs, 7 385 vrais K2–K6, 19 morceaux/folds 0/1/2/4. Parent S18 inchangé : **82,9183 % global, 40,5958 % poly**, 2 934 corrections et 2 210 régressions vs `freeze_local_combo`. YourMT3+ conservé comme benchmark de comparaison, pas entrées des réseaux. **Tous ces choix sont évalués sur la même cohorte déjà exposée** ; rejouer des modèles ne constitue pas une validation sur de nouveaux morceaux.

### S24 — inversion à poids identiques (expérience causale sur l'ordre)
Exécution : https://github.com/Andriamarosoa/note/actions/runs/37862241264
Archive permanente : https://github.com/Andriamarosoa/note/releases/tag/v273-series24-research-37862486503

Les 19 modèles S20 A-first ont été rechargés ; reproductibilité initiale `max |A-archive| = 1,192×10^-6`. B émet son message sur le prior S18 **avant** le premier A ; exactement le même réseau est ensuite exécuté A-first ou B-first. **A1 change d'argmax dans 1 152 cas**, A2 dans 148, A3 dans 21, A4 dans 1. Le changement moyen de P(A1) est 0,00642, preuve causale locale de l'influence de B et non d'un apprentissage inversé.

| Étape d'A | A-first correct global | B-first correct global | A-first correct poly | B-first correct poly |
|---|---:|---:|---:|---:|
| 1 | 48 880 | 48 795 | 2 635 | 2 631 |
| 2 | 48 930 | **48 935** | 2 720 | 2 707 |
| 3 | 48 924 | 48 924 | 2 699 | 2 698 |
| 4 | 48 928 | 48 928 | 2 703 | 2 703 |

La tête B seule, dont la sortie BCE est utilisée comme diagnostic d'argmax et **pas comme une distribution Exact-K calibrée**, donne 49 144 prédictions globales correctes mais seulement 26,6486 % poly. B erronée → premier A corrigé dans 1 848 cas, mais B correcte → premier A faux dans 2 197 cas. Meilleure politique protégée S24 (passage2 marge0,6) : 82,9419 % global / 40,4062 % poly, +61 corrections vs S18, **−47 régressions nouvelles**. Aucun candidat de S24 ne préserve toutes les corrections.

### S25 — apprentissage explicitement inversé B→A→B→A
Entraînement : https://github.com/Andriamarosoa/note/actions/runs/37862417522
Rejeu indépendant des poids : https://github.com/Andriamarosoa/note/actions/runs/37862590340
Archive permanente : https://github.com/Andriamarosoa/note/releases/tag/v273-series25-research-37862590340

Même capacité A/B, mêmes seeds, quatre pertes de chaque tête, 6 epochs, 19 morceaux exclus un à un, prior S18, signatures d'origine. B parle avant A pendant **l'apprentissage**, et les poids sont réentraînés à cet ordre. Rejeu des 19 poids réussi : erreur A 4,47×10^-7, B 2,98×10^-7, **28 politiques exactement reproduites**.

| Passage | Corrections B-first vs S20 A-first | Régressions B-first vs S20 A-first | Solde |
|---|---:|---:|---:|
| 1 | 323 | 384 | −61 |
| 2 | 97 | 81 | **+16** |
| 3 | 53 | 43 | **+10** |
| 4 | 61 | 50 | **+11** |

La tête B seule obtient 49 122 corrects globaux / **25,4435 % poly**. Après B, A1 corrige 1 956 cas où B se trompait mais détruit 2 259 bonnes décisions de B. Cela ne prouve pas que la tête B est la pire **au global** par rapport à tous les bras, seulement qu'elle est **faible pour la polyphonie** et qu'un message faible peut aider A à certains passages.

Le meilleur choix au global, S25 `pass3 margin0.6`, obtient **82,9520 % global, 40,4604 % poly** : 67 corrections et **47 régressions** vs S18. Aucun choix ne passe le critère zéro perte ; pas de promotion.

### S26 — « promouvoir » une proposition B-first confirmée par A-first
Exécution : https://github.com/Andriamarosoa/note/actions/runs/37862754589

**90 règles prédéclarées sans nouveaux labels ou entraînement** : six structures d'accord entre décisions A après B dans les deux ordres ; trois conditions (confiance commune A, B0 soutient le candidat, B0 conteste le parent), cinq marges. Tous les candidats et vrais K audités par fold. Meilleur global `p2p4 B0_supports margin>0.4` : **82,9469 % global / 39,8646 % poly**, 135 nouvelles corrections, **118 nouvelles régressions** vs S18 ; gain net +17 mais baisse poly de 0,7312 point. **Zéro des 90 politiques** atteint le critère strict sans correction perdue.

## Conclusion et décision

L'ordre inversé est utile pour produire **d'autres propositions**, mais ne garantit pas une meilleure réponse : il peut faire apparaître des corrections que l'ordre ordinaire n'obtient pas, avec des régressions qui les compensent ou les dépassent. Plusieurs passages rapprochent les prédictions, car les messages B finissent par influencer A dans les deux séquences ; à poids figés, les résultats sont presque identiques au passage4. Dans le cas appris, l'ordre inversé procure de petits gains globaux au passage2–4, mais sans gain poly ni protection des bonnes décisions initiales.

Le blocage n'est pas l'absence de boucles ; c'est l'absence d'un **signal fiable distinguant un message B qui permet une vraie correction d'un message B qui détruit une note**. Il faut maintenant définir des contraintes musicales/acoustiques indépendantes (nombre d'attaques plausibles, cohérence harmonique, résidus et morphologie du son) pour que l'arbitre apprenne la **qualité de la modification proposée**, pas seulement sa probabilité brute. Conserver S18 et les 2 934 anciennes corrections ; pas de promotion ni revendication de dépassement de YourMT3+.
