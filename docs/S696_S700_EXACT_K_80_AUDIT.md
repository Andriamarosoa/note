# S696–S700 — Exact-K causal et objectif 80 %

## Protocole conservé

* Modèle de base : NOTE strict causal S630 (256 samples/hop à 44,1 kHz).
* Enregistrements authentiques MIC GuitarSet : 36 TRAIN, 12 DEV.
* Les compositions TRAIN et DEV sont disjointes, comme les guitaristes.
* Vérité K : nombre de notes annotées simultanément, limité aux classes 0..6.
* **Deux mesures distinctes** : toutes les trames et trames avec vérité K>0.
* **NE PAS CONFONDRE** K natif (tête 7 classes), K décodé (événements MIDI
  effectivement actifs) et F1 Onset/pitch ou F1 des notes complètes.
* S675, les sorties NoteOn, NoteOff, les pitchs et la tête K originale
  **n'ont jamais été modifiés** lors des benchmarks S696–S700.
* Les morceaux DEV ont déjà été exposés historiquement ; ces expériences
  ne constituent pas une validation prospective finale.

## S696 — Mesure du déficit d'Exact-K

Sur les 12 DEV : 48 970 trames dont 37 080 avec notes actives.

| K | Exact total | Exact sur K>0 |
|---|---:|---:|
| Original MIDI décodé | 54,4721 % | 46,7125 % |
| Tête K native directement | 58,2009 % | 58,1823 % |

La politique de lissage sélectionnée sur 5 compositions TRAIN ne passe
pas les verrous de généralisation sur 4 autres compositions TRAIN.

Run : https://github.com/Andriamarosoa/note/actions/runs/39069279263

## S697 — Tête K apprise sur 20 sources TRAIN

63 entrées causales : distributions natives K, K passés et lissés,
statistiques de sustain/onset/release et spectre fondamental agrégé,
compte des notes MIDI déjà détectées.

| S697 | Exact total | Exact sur K>0 |
|---|---:|---:|
| Tête K native, 16 TRAIN tenus hors apprentissage | 84,1134 % | 85,1048 % |
| Tête K apprise, mêmes 16 TRAIN | 79,85 % | 79,27 % |
| Tête K native, DEV | 58,2009 % | 58,1823 % |
| Tête K apprise, DEV | 61,51 % | 59,45 % |

**80 % sur TRAIN ne valent PAS 80 % sur DEV**, en particulier sur les
compositions/guitaristes différents. Choix TRAIN : `frozen_direct_K`,
donc pas de promotion de la tête K apprise.

Run : https://github.com/Andriamarosoa/note/actions/runs/39069560833

## S698 — 80 prises originales multi-guitaristes, 5 compositions TRAIN

Même tête de famille K, 100 200 exemples supervisés. Aucun joueur
04 (DEV) ou 05 (TEST) à l'entraînement.

| S698 | Exact total DEV | Exact sur K>0 DEV |
|---|---:|---:|
| K native | 58,2009 % | 58,1823 % |
| Nouvelle K apprise sur 80 prises | **66,96 %** | **65,65 %** |

La nouvelle tête gagne **8,76 points** sur DEV par rapport au K natif,
mais perd face au K natif sur les compositions TRAIN tenues à l'écart
(74,75 % au lieu de 84,11 %), ce qui la disqualifie pour promotion.

Run : https://github.com/Andriamarosoa/note/actions/runs/39069662013

## S699 — Toutes les 9 compositions TRAIN, validation joueur jamais vu

Architecture indépendante : 108 prises authentiques joueurs 00–02
entraînent une K SHADOW ; 36 prises joueur 03 servent exclusivement à
la sélection ; joueur DEV 04 reste entièrement inconnu. En cours de
vérification au moment de cette première rédaction.

Run : https://github.com/Andriamarosoa/note/actions/runs/39070073982

## S700 — Peut-on corriger l'Exact-K par un décalage ?

Sur les 48 MIC réels, test de 13 délais causaux de 0 à 64 hops,
d'une borne FUTURE oracle distincte et interdite en LIVE.

* Aucun délai causal n'a passé le choix TRAIN.
* Meilleur oracle NON CAUSAL (+4 hops) : **60,4574 %** K total DEV,
  **59,5011 %** K actif DEV.
* Donc l'erreur ne vient pas principalement d'un simple retard constant.

Run : https://github.com/Andriamarosoa/note/actions/runs/39070392402

## Bilan objectif 80 %

Le projet n'a **PAS 80 % de détection des notes**, ni 80 % de K
sur GuitarSet DEV : ces pourcentages ne doivent pas être inventés
ou extrapolés d'un corpus TRAIN facile. L'écart dépend manifestement
des timbres, des styles et de la présence de notes superposées.

Ordre de priorité : généralisation K des nouvelles compositions et
guitaristes, récupération des fondamentales faibles au niveau
NoteOn+pitch, stabilité de note complète (avec S675), puis benchmark
réel du microphone Windows sans `input_queue_full`. Aucune
nouvelle tête S696+ n'est intégrée au moteur LIVE.
