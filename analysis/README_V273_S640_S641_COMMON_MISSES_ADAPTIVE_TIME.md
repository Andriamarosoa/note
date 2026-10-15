# NOTE S640–S641 — Causes des 613 échecs communs et coût du retard LIVE

Date : 16 octobre 2026. Branche : `codex/s640-common-miss-audit`.
**Statut : analyses reproductibles terminées, AUCUNE promotion.**

## Provenance gelée

- [S638 : 12 véritables WAV GuitarSet et les deux moteurs originaux](https://github.com/Andriamarosoa/note/actions/runs/38811643000).
- [S639 : appariements causaux exacts et 613 annotations manquées par les deux](https://github.com/Andriamarosoa/note/actions/runs/38812452658).
- [S640 : audit du signal acoustique S614 pour ces mêmes annotations](https://github.com/Andriamarosoa/note/actions/runs/38813870389), [rapport S640 original](https://github.com/Andriamarosoa/note/actions/runs/38813870389/artifacts/12010191294).
- [S641 : rejeu des temps de sortie sans modifier le modèle](https://github.com/Andriamarosoa/note/actions/runs/38814405812), [rapport S641 original](https://github.com/Andriamarosoa/note/actions/runs/38814405812/artifacts/12009627597).

S614 officiel SHA256
`7ccf046e3a1870eaca3a108e6f9565298aeeadf43732a2f83559542c03f5f17f`.
Modèles S80 / S614 / S632 / MIDI V2.2.1 strictement inchangés.
Les 12 extraits sont DEV précédemment exposés ; ni nouveau holdout,
ni résultat TEST indépendant, ni preuve de latence hardware.

## S640 : audit de la vraie attaque après la référence

**Workflow réussi**, 14 tests, 12 SHA256 WAV vérifiés, événements
NoteOn/NoteOff de S614 reproduits **à l'identique** à partir du
signal PCM original. Parmi 1 580 annotations MIDI 40–76,
**613 notes manquées par les deux** dans l'évaluation S635 causale
à 80 ms et **967 annotations contrôles** reconnues par au moins
un système.

Les catégories sont diagnostiques **exclusives**, choisies dans
cet ordre : sortie de même hauteur prématurée, sortie tardive,
attaque sous seuil, attaque sans maintien simultané, sélection
disarmée/réfractaire, capacité et autre. La présence d'une
sortie voisine de même hauteur peut être due à une note
différente ou une réattaque : ce ne sont pas 177 corrections
garanties sans analyse de la chronologie.

| Catégorie | Quantité | Fraction des 613 |
|---|---:|---:|
| Pas de pic d'attaque S614 ≥0,60 dans les 80 ms suivant la vérité | **379** | 61,8 % |
| Pic d'attaque sans maintien S614 simultanément ≥0,50 | **53** | 8,6 % |
| NoteOn même hauteur trop tôt (dans les 80 ms avant vérité) | **93** | 15,2 % |
| NoteOn même hauteur trop tard (80–240 ms après vérité) | **84** | 13,7 % |
| Proposition prête mais non appariée | **4** | 0,7 % |
| **TOTAL** | **613** | **100 %** |

Autres constats, **non exclusifs** :
- **468/613 = 76,35 %** ne dépassent **jamais** le seuil
  d'attaque 0,60 après l'annotation, même si un événement
  proche a pu se déclencher *avant* l'annotation ;
- 101/613 disposent d'une émission tardive (drapeau indépendant
  du classement principal), 93 d'une émission précoce ;
- 49/613 = 7,99 % ont un compétiteur harmonique S614
  éligible (+12,+19,+24 demi-tons) dans la fenêtre, **sans preuve
  que l'harmonique provoque la perte** ;
- **0** panne de sélection classée sous le plafond de polyphonie
  ou une fenêtre de réarmement (sur ce protocole et cette
  classification). Cela ne prouve pas qu'aucune erreur de
  sélection n'existe sur d'autres audios ;
- S614 seuil officiel attaque 0,60 / maintien 0,50.

Distribution comparée (extraits annotés positifs) :

| Mesure | 613 notes manquées par les deux | 967 autres notes |
|---|---:|---:|
| Pic attaque médian dans les 80 ms après annotation | **0,366** | 0,935 |
| Pic maintien médian | **0,392** | 0,979 |
| Pic attaque sous 0,60 | **76,35 %** | 4,03 % |
| Compétiteur harmonique éligible | **7,99 %** | 11,69 % |
| Proposition prête | **0,65 %** | 93,17 % |
| Même hauteur sortie précocement | 15,17 % | 2,79 % |

Il n'y a pas d'enrichissement apparent en cas avec
compétiteur harmonique selon cet indicateur : **ne pas
attribuer automatiquement la majeure partie des FN aux
harmoniques.** L'échec majeur observable est le faible
signal de naissance de note et le désalignement du temps.

Sur 613, **526** sont en accompagnement, **87** en solo.
Les cas graves (MIDI 40–52) comprennent 133 attaques
sous seuil, 19 sans maintien simultané, 30 tardives,
20 prématurées.

## S641 : les délais uniformes sont utiles mais régressifs

Workflow réussi, sur **les mêmes 12 sources DEV**
et les **1 583 NoteOn NOTE** inchangés. On décale le temps
de sortie, mais **pas** la hauteur ni la classification ;
aucun futur audio ni vérité MIDI n'intervient dans une
prédiction, aucune tête n'est entraînée.

Les émissions retardées après la fin du clip restent
comptées dans le score (elles ne sont pas supprimées).
Les cas de score sont recomptés au même appariement
causal S635 « même hauteur, NoteOn ≥ attaque réelle,
≤ attaque réelle +80 ms ».

| Retard hypothétique de toute émission | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|
| 0 ms, original | 901 | 682 | 679 | **56,97 %** |
| 5,80 ms | 917 | 666 | 663 | **57,98 %** |
| 11,61 ms | 932 | 651 | 648 | **58,93 %** |
| **17,41 ms** | **937** | **646** | **643** | **59,25 %** |
| 23,22 ms | 930 | 653 | 650 | 58,80 % |
| 34,83 ms | 863 | 720 | 717 | 54,57 % |
| 46,44 ms | 707 | 876 | 873 | 44,70 % |
| 69,66 ms | 205 | 1 378 | 1 375 | 12,96 % |

**À 17,41 ms :** 59 vérités nouvellement reconnues et
**23 vérités auparavant reconnues devenues manquées** ;
gain net +36 TP, +2,2763 points F1, sur DEV.
8/12 extraits améliorés et 3 régressés (1 inchangé).

**Ne pas promouvoir** le « meilleur 17,41 ms » : il a été
sélectionné *a posteriori* sur les annotations DEV ;
c'est une simulation de délai uniforme, pas une
latence adaptative réellement entraînée et exécutée
sur microphone ; aucun score TEST indépendant.
Les NoteOff doivent aussi être décalés en production
pour préserver les paires MIDI et les durées.

Le véritable enjeu reste d'apprendre **quelle attaque**
mérite d'attendre, et pour combien de trames, tout
en préservant celles déjà correctement émises.

## Décision pour l'architecture suivante

À étudier **en TRAIN d'abord, jamais sur ce DEV** :

1. Améliorer la **création de candidats acoustiques faibles**
   à hauteur exacte, avec hard négatifs correspondants
   (vrais harmoniques, résonances, notes voisines).
2. Entraîner une sortie temporelle causale **0 / 1 / 2 / 3 hops**
   conditionnée par les caractéristiques de naissance, plutôt
   que de choisir le meilleur retard par recherche sur DEV.
3. Préserver la cohérence MIDI NoteOn/NoteOff et mesurer
   le délai microphone→MIDI, pas seulement le F1.
4. Geler la stratégie sur TRAIN et DEV puis valider les sources
   inédites et par rôle solo/accompagnement ainsi que registre.
5. Si la stratégie améliore un corpus mais régresse sur les
   autres, garder l'assistant désactivé par défaut.

**S614 reste la référence : 69,03 % F1 sur son historique
TEST selon la métrique symétrique ±80 ms.**
La valeur 59,25 % de S641 est un contre-factuel DEV selon
la métrique causale [0,+80 ms], pas un remplacement.
