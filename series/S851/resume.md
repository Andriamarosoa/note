---
serie: S851
statut: en_cours
type: experience
reference: S708
branche: codex/note-s851-identical-guitar-standard-ab256
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S851 — Boucle 4 : comparaison honnête BCE/RANK au même transposeur standard

## Objectif
Tester si un véritable entraînement S850 sur plusieurs guitaristes et 32 blocs passés **corrige le blocage MIDI73** et suit réellement la phrase originale. Ne pas promouvoir par simple amélioration F1 de note. Comparer à la sortie pitch-shift standard +12 hors ligne issue du run S848 sur **exactement le même WAV réel**.

## Hypothèse
Le BCE sélectionné sur CAL en S850 peut être plus discriminant que S847 sur des guitaristes indépendants ; pourtant le F1 sur GuitarSet et la sortie sonore sont deux preuves différentes. S849 avait démontré que reset8/reset32 sans réentraînement ne corrigeait pas S847. Un réseau S850 entraîné sur 32-hops pourrait être différent, mais cela doit être démontré.

## Modifications
- `scripts/s851_compare_neural_rank_vs_standard256.py` : charge les **vrais poids** S850 BCE et RANK issus de #40097584744, les 39 vraies notes enregistrées issues de S847 #40094119273 et le même audio 3,5s / standard +12 signé de S848 #40095759550. Pour chaque modèle, évalue le flux GRU continu et le flux reset32, **quatre audios réels échantillonnés**, et mesure vrais choix de notes, silence, pourcentage note dominante, spectre, corrélation d'enveloppe et distance log-mel par rapport au standard.
- Choix du meilleur réseau provenant **exclusivement de CAL S850**, jamais du WAV externe (les quatre sont des audits et non une recherche opportuniste de meilleur son).
- Le seul post-traitement est la **copie de vrais blocs d'ondes de la banque** ; pas de synthétiseur, pas de DSP pitch shift pour l'IA. Le standard issu S848 est la référence externe offline, non causale, et ne doit pas être invoqué dans le modèle.
- La sortie **rejouée de la banque** n'est pas une transposition fidèle de la performance, même si la hauteur est juste. Ne pas prétendre améliorer le timbre/attaque/vibrato en réutilisant une vraie note.

## Critères et protocole
Aucun score F1 d'Onset dérivé de cette expérience. Pour accepter une future démonstration, il faut au minimum sélectionner différentes bonnes notes quand la musique change (absence de classe unique >95 %), conserver une enveloppe raisonnablement proche du standard (corrélation >0,5 dans cette étape), produire du son audible, et réduire la distance spectrale comparée au témoin S847 qui avait 18,377 dB de distance log-mel. **Ces seuils sont des garde-fous nécessaires, jamais une preuve suffisante de justesse des notes.** Le vrai terrain GT GuitarSet est analysé séparément en S850.

## Audits obligatoires
| Type | Contrôle |
|---|---|
| Données | trois artefacts réels S847/S848/S850, source SHA identique |
| Architecture | 256 nouveaux PCM par hop et GRU passée ; méthode standard hors réseau |
| Apprentissage | aucun training sur ce benchmark, checkpoints figés |
| Métriques | sortie physique enregistrée mais non transposition exacte, refus note figée/silence |
| LIVE | pas de futur ni hold, CPU p99 Windows/iRig non mesuré |

## Statut
En cours, non promu. Référence de travail Onset PCM256 **69,2584331 % F1**, six têtes et MIDI intacts.
