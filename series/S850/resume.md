---
serie: S850
statut: en_cours
type: experience
reference: S708
branche: codex/note-s850-causal-multinote-rank256
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S850 — Boucle 2 et 3 : apprendre à distinguer les vraies notes concurrentes en transposition

## Objectif et origine
S848 a démontré par une référence pitch shift standard sur le **même WAV** que S847 répétait **MIDI 73 sur 603/603 blocs**, avec 11,64 % de corrélation d'enveloppe vs 90,38 % pour le standard. S849 a ensuite montré qu'une réinitialisation GRU tous les 8 ou 32 pas ne corrige **pas** le blocage : toujours MIDI 73 sur 603/603. Nous ne maquillons plus les erreurs avec un seuil ou une normalisation de volume : deux **réseaux réellement entraînés** sont maintenant comparés à données égales.

## Hypothèse
L'ancien réseau apprenait sur **8 blocs consécutifs de 256** uniquement et sa perte BCE pondérée pouvait privilégier des classes récurrentes au lieu d'apprendre les différences physiques entre plusieurs hauteurs. Une tête avec convolution brute + mémoire GRU causale sur **32 blocs d'entraînement**, vraie supervision **multi-note**, *silence explicite* et perte **hard-negative ranking** peut mieux séparer un E3 physiquement joué d'une harmonique E2 ; l'hypothèse reste à démontrer, notamment sur des guitaristes jamais vus.

## Modifications
- `scripts/s850_prepare_real_mic_longcontext256.py` : lit uniquement **vrais WAV GuitarSet mono 44.1kHz/16-bit et JAMS signés SHA256**, sans hex/SF2 synthétisé, ni DSP de transposition. TRAIN guitaristes 00–03 / CAL 04 / OUTER 05 (aucune piste ni guitariste partagé), 16 pistes par guitariste équilibrées BN1/Funk2/Jazz1/Rock3/SS1, 48 séquences de 32 décisions 256 par piste, priorité hors ligne aux annotations de vraies notes graves E2/E3 et chevauchements. Les GT ne sont **jamais** des entrées à l'inférence Live.
- `src/onset256/s850_causal_polyphonic_transposer256.py` : nouvelle tête 256 brute CNN+GRU, embedding de décalage ±12 appris, 49 scores de notes source + cibles transposées, deux scores explicites de présence de notes, GRU état passé et aucune fenêtre de micro futur. Multi-label : plusieurs vraies notes peuvent être actives simultanément, pas seulement une note argmax. Les GT déplacées en hauteur sont générées **OFFLINE** pour apprendre, jamais au moment de détecter des sons.
- `scripts/s850_train_real_causal_poly_note256.py` : compare deux **variantes IA indépendantes et entraînées de zéro** sur le même vrai corpus : **BCE** (contrôle) et **RANK** (même architecture + rang positif vs concurrents négatifs à coefficient 0,75). Deux modèles et deux traces TRAIN/CAL/OUTER sont archivés ; poids positifs calculés sur TRAIN seulement, checkpoint le plus bas en erreur CAL, variante globale sélectionnée sur **CAL uniquement**, puis score OUTER consulté. Jamais choisir sur OUTER pour créer un gain artificiel.
- `tests/test_s850_causal_polyphonic_transpose256.py` : teste la causalité exacte à 256, l'équivalence d'unroll 32 et streaming, polyphonie E2+E3, gradient réel du rang, seuil CAL, partitions sans fuite.
- Aucune modification du MIDI officiel, ni de Onset/Offset/Exact-K/Active Exact-K/Frame Pitch/Full Note. Les modèles restent shadow research.

## Contrat
**Seule contrainte temporelle** de la tête : nouveau PCM256 à chaque décision de 256/44100 ×1000 = **5,805 ms**, sans audio futur ni attente additionnelle. Les 32 blocs ne sont **pas attendus** à l'inférence : le réseau apprend à exploiter sa propre mémoire causale accumulée ; il émet une décision dès le premier 256. Il reste à mesurer les durées de calcul CPU p99 sur Windows/iRig. Une GRU conservant le passé n'induit pas de latence algorithmique, mais un modèle coûteux peut dépasser la fenêtre physique.

## Mesures et conditions de promotion
- Avec vrai GT E2/E3 et vrais accords, publier TP/FP/FN par pitch pour shifts -12, 0, +12 sur CAL et OUTER ; publier top1 dont le GT contient la note, ainsi que la sortie « aucune note ».
- Le score est un **proxy de note active**, pas le vrai Onset F1. Un gain de ce proxy ne justifie pas de toucher au MIDI.
- Après les vrais résultats GT, refaire A/B/C sur le **même WAV** que S848, standard +12 vs sortie IA basée sur de vraies prises physiques, vérifier notamment si MIDI73 reste fixé, corrélation de dynamique et distance log-mel ; ensuite véritable replay de 80 WAV GuitarSet, plus set externe, et p99 machine Live, avant promotion.
- Rejeter les faux positifs, les notes imaginaires, les silences, ou une reconstruction audio faible. Un standard hors ligne n'a aucune obligation de latence 256, il n'est qu'une référence de précision sonore.
- Les notes d'une banque de vraies prises ne constituent pas une transposition fidèle de la performance ; la comparaison standard doit le révéler sans embellissement.

## Audits obligatoires
| Type | Critères |
|---|---|
| Données | SHA micro 237cdc... et SHA JAMS 8daa..., vrai audio et partitions guitaristes étanches |
| Acoustique | 256 nouveaux samples, GRU passé, aucun FFT à longue fenêtre futur, zéro pitch DSP dans l'IA |
| Architecture | BCE vs RANK appris, chaque note indépendante, activité/silence, plusieurs hauteurs actives |
| Apprentissage | 32 hops consécutifs pour entraîner la mémoire, CAL seuls pour choisir checkpoint/architecture |
| Métriques | F1 par classe et top1 GT, TP/FP/FN et vrai standard en A/B indépendant ; ne jamais appeler proxy Onset |
| LIVE | sans attente, p99 Windows/iRig à prouver, pas de MIDI |

## Statut
En cours, non promu. Référence Onset travail PCM256 **69,2584331 % F1** (7 607 TP / 3 093 FP / 3 660 FN), intacte.
