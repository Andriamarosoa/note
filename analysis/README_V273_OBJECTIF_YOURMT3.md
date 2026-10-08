# Objectif : dépasser YourMT3+ en Exact-K

**Objectif non atteint. Les seules corrections de sélection ne peuvent pas
suffire avec les propositions et décisions actuellement conservées.**

La demande est de battre YourMT3+, et pas seulement `freeze_local_combo`.
Le critère devient : dépasser le checkpoint YourMT3+ fixé **en Exact-K global
et en polyphonie K2–K6**, puis confirmer ce résultat sur des données inédites,
avec les entrées, le délai et les exclusions documentés. Les scores K0 à K6,
les corrections et les régressions restent tous visibles.

Cette comparaison porte sur le nombre de nouvelles attaques attribuées aux
groupes natifs. Elle ne mesure pas la qualité d'une transcription MIDI complète.

## Comparaison exécutée sur les mêmes lignes

Les **59 309 identifiants**, vérités, prédictions de référence et folds ont
été alignés et vérifiés entre les deux expériences. Les **211 politiques**
conservées ont été comparées. Folds 0, 1, 2 et 4 ; fold 3 et player05 exclus.

| Système | Corrects globaux | Exact-K global | Corrects poly | Exact-K poly |
|---|---:|---:|---:|---:|
| `freeze_local_combo` | 48 454 / 59 309 | 81,6976 % | 2 530 / 7 385 | 34,2586 % |
| Meilleure politique actuelle : calibration + garde | 48 617 / 59 309 | 81,9724 % | 2 693 / 7 385 | 36,4658 % |
| YourMT3+ figé | 51 328 / 59 309 | 86,5434 % | 4 028 / 7 385 | 54,5430 % |

Il manque **2 711 réponses globales et 1 335 réponses polyphoniques** pour
égaler ce résultat, soit respectivement **4,5710 et 18,0772 points**.
Pour le dépasser strictement sur ce même échantillon : au moins 51 329
réponses globales et 4 029 réponses polyphoniques correctes.

La meilleure politique est
`regression_loops_posthoc__group_audit__cost_1.15__no_favorable_group`.
Son choix comme meilleure des 211 politiques est descriptif, après exposition
aux résultats ; il ne constitue pas une nouvelle validation indépendante.

| Vrai K | Lignes | Système actuel | YourMT3+ | Écart de réponses correctes, actuel − YourMT3+ |
|---|---:|---:|---:|---:|
| 0 | 39 652 | 95,9220 % | 95,4202 % | +199 |
| 1 | 12 272 | 64,2846 % | 77,1186 % | −1 575 |
| 2 | 3 445 | 39,5646 % | 52,7721 % | −455 |
| 3 | 2 289 | 37,9642 % | 54,6964 % | −383 |
| 4 | 1 207 | 35,7912 % | 60,0663 % | −293 |
| 5 | 355 | 8,1690 % | 53,2394 % | −160 |
| 6 | 89 | 0,0000 % | 49,4382 % | −44 |

## Preuve d'un plafond du catalogue figé

Pour chaque ligne, on autorise un **oracle utilisant la vérité** à garder
la référence correcte ou à choisir n'importe quelle bonne réponse disponible.
Il ne commet aucune régression. Ce n'est ni un modèle, ni un résultat réalisable
revendiqué ; c'est le maximum exact d'un sélecteur limité à ces sorties figées.

| Choix disponibles à cet oracle | Corrects globaux | Global maximal | Poly maximal | Dépasse YourMT3+ global ? |
|---|---:|---:|---:|---|
| Les 255 groupes actuels + KEEP | 50 211 | 84,6600 % | 58,0501 % | Non |
| Les 211 politiques historiques + KEEP | 50 181 | 84,6094 % | 57,6439 % | Non |
| Union des deux catalogues + KEEP | 50 436 | 85,0394 % | 61,0968 % | Non |

Même la dernière union manque de **892 réponses correctes** pour égaler
YourMT3+. Il faut donc au minimum **893 bonnes réponses supplémentaires hors
de cette couverture**, en supposant déjà un choix parfait partout ailleurs.
Ce minimum n'est pas une prévision de gain.

La mémoire reste utile : elle couvre **225 erreurs de référence supplémentaires**
par rapport aux 255 propositions actuelles. Ces succès ne sont pas perdus.
La couverture totale des erreurs corrigibles monte ainsi de 1 757 à **1 982**.
Réentraîner les producteurs, ajouter une nouvelle sélection ou construire de
nouvelles propositions change cette couverture : le plafond calculé ne borne
pas ces systèmes futurs ni tous les réseaux possibles.

## Où se trouve l'écart actuel ?

Les groupes entiers restent les unités de choix. Aucun veto fondé sur un
membre isolé n'est ajouté. Le système intervient sur les 7 493 lignes où le
K initial vaut 2, 3 ou 4, et propose seulement des K de 2 à 6.

| Région descriptive | Lignes | Actuel correct | YourMT3+ correct | Avantage net YourMT3+ |
|---|---:|---:|---:|---:|
| Hors des lignes où le correcteur peut agir | 51 816 | 45 939 | 46 927 | +988 |
| Lignes éligibles, bonne réponse absente des choix actuels | 3 221 | 0 | 1 789 | +1 789 |
| Lignes éligibles, bonne réponse disponible, KEEP compris | 4 272 | 2 678 | 2 612 | −66 |

Ces régions utilisent la vérité pour le diagnostic ; elles ne sont pas des
catégories directement utilisables à l'inférence. Elles montrent que la
capacité à **produire la bonne réponse et à intervenir sur la ligne** est
indispensable pour combler l'écart. Elles n'établissent pas une cause acoustique.

Exemples de limites vérifiées :

- Le score K1 est inchangé par les 211 politiques : aucune correction vers K1 n'est
  disponible, alors que YourMT3+ obtient 1 575 réponses K1 correctes de plus.
- Avec les 255 groupes actuels, les maxima K5 et K6 sont seulement 94/355
  et 3/89, contre 189/355 et 44/89 pour YourMT3+.
- Même en ajoutant toute la mémoire, les maxima K5 et K6 atteignent seulement
  143/355 et 18/89. Une meilleure sélection seule ne comble pas ces déficits.
- Les 1 889 lignes polyphoniques prédites initialement K0 ou K1 sont hors du
  champ d'action actuel. Leur accès doit être évalué, en protégeant les
  nombreuses bonnes réponses K0/K1.

## Conséquence pour les prochaines boucles

1. **Élargir les réponses et les lignes accessibles.** Tester des producteurs
   capables de proposer K0 à K6, y compris quand la référence prédit K0/K1,
   avec leurs exclusions d'entraînement propres. Mesurer d'abord la nouvelle
   couverture de bonnes réponses, puis les corrections/régressions réellement
   choisies. Ajouter des seuils au catalogue figé ne résout pas son plafond.
2. **Conserver la mémoire et les interactions.** Une correction même minime
   reste un candidat documenté avec ses succès, ses régressions et ses poids.
   Les choix portent sur des combinaisons complètes. Les sorties historiques
   ne deviennent pas automatiquement des données d'entraînement sûres pour un
   nouveau consommateur : il faut régénérer leurs sorties avec les exclusions
   de celui-ci. Une fuite résiduelle ne prouve pas à elle seule une catégorie
   nouvelle.
3. **Mesurer chaque résultat contre YourMT3+ aussi.** Un gain interne est une
   étape conservée ; il ne satisfait pas l'objectif tant que les deux scores
   comparables ne dépassent pas la référence externe. Garder les déficits par
   K et les cas gagnés/perdus, sans masquer les classes faibles par K0.
4. **Confirmer hors des données déjà explorées.** Figer le système et ses
   réglages avant la mesure sur des morceaux réellement inédits. Pour une
   affirmation à délai égal, contrôler le contexte audio des deux systèmes.

Les nouveaux producteurs et cette validation indépendante **ne sont pas
encore exécutés**. Cet audit établit la nécessité d'élargir le système ; il
ne prétend pas avoir obtenu ces gains.

## Portée et preuves conservées

La référence est le checkpoint officiel `YPTF.MoE+Multi (noPS)`, fixé dans le
[protocole initial](yourmt3-exactk-protocol.md), du
[run réussi 37605163312](https://github.com/Andriamarosoa/note/actions/runs/37605163312).
Le ZIP original est conservé octet pour octet, SHA256
`d47cbe59f66139dbbbb0c38b1dc2a1eb050d7ea25062f1dbba096971cb56eb89`.
Tous ses scores ont été recalculés depuis les prédictions.

YourMT3+ utilise du contexte hors ligne et le recoupement de son préentraînement
avec GuitarSet n'est pas exclu. Notre résultat vient aussi de données déjà
exposées pendant le développement ; les calibrateurs utilisent d'autres
morceaux du fold, tout en excluant le morceau testé des ajustements. Cette
comparaison alignée reste **descriptive**, pas une preuve indépendante ni à
latence égale. Aucun modèle n'est promu.

- [Rapport numérique](evidence/v273-yourmt3-target/comparison/report.json)
- [Comparaison des 211 politiques](evidence/v273-yourmt3-target/comparison/all-211-versus-yourmt3.csv)
- [Preuves par ligne et ensembles de réponses](evidence/v273-yourmt3-target/comparison/row-evidence.npz)
- [Archive originale YourMT3+](evidence/v273-yourmt3-target/upstream/yourmt3-exactk-summary.zip)
- [Provenance](evidence/v273-yourmt3-target/upstream/source-manifest.json)

Reproduction, depuis la racine du dépôt, vers un dossier inexistant :

```sh
PYTHONPATH=. python -B scripts/audit_v273_yourmt3_target.py \
  --output /tmp/v273-yourmt3-target-replay
```

Le script refuse les différences d'identifiants, annotations, référence,
folds, empreinte du benchmark et inventaire des politiques. Il n'entraîne
aucun modèle et ne modifie aucun des 211 candidats existants.
