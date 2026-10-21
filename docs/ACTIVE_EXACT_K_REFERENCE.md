# Référence de développement — Active Exact-K : S828 natif256

**Décision utilisateur, 21 octobre 2026 : adopter S828 comme référence pour les prochaines recherches Active Exact-K sur la grille MIDI de 256 échantillons.**

## Périmètre exact

| Axe | Référence de développement retenue | Score vérifié | Cohorte |
|---|---|---:|---|
| Onset F1 | S708 gelé | 74,2639 % | CAL32 |
| Exact-K | S708 gelé | 62,0455 % | CAL32 |
| **Active Exact-K** | **S828** | **73,831146 %** | **FIT80** |
| Offset accuracy | S708/S707 gelé | 77,3292 % | CAL32 |
| Full Note F1 | S708 gelé | 57,1319 % | CAL32 |
| Frame Pitch F1 | S708 gelé | 76,4626 % | CAL32 |

**Ces scores sont des références individuelles, pas un score d'un système unifié évalué sur une même cohorte.**

S828 a été validé sur 80 vrais enregistrements GuitarSet, 249 411 trames de vérité GT actives et 184 143 comptages corrects, cinq compositions BN1/Funk2/Jazz1/Rock3/SS1. Série S828 : `series/S828/resume.md`, `series/S828/resultat.md`. Run GitHub **SUCCESS** : https://github.com/Andriamarosoa/note/actions/runs/39604850899 ; commit du calcul vérifié `bec403685379a581554f4d6d2d1e6629370e477f`. Artefact `S828_FULL_FIT80_PRIVATE_S708_256_COMPOSITION_CALIBRATED_ACTIVEK_ONLY`, ID `12349523181`.

La meilleure référence **historique / diagnostic** S823 reste **76,877523 % sur la même cohorte FIT80** (191 741/249 411). Le déficit S828 est **7 598 comptages et 3,046377 points**. Mais S823 dépend d'observations S795 recalculées toutes les 1 024 valeurs et d'états MIDI historiques ; S828 utilise les caractéristiques S708 réellement recalculées toutes les 256, ce qui justifie le choix de S828 comme base **natif256**.

## Nature du modèle de référence

- Grille audio 44,1 kHz, extrait 256 échantillons (≈5,805 ms). Une inférence S708 réelle par 256, entrée `[49,288]`; aucune caractéristique as-of1024 à l'inférence S828.
- **Les poids neuronaux S708 préexistants sont gelés** et ont été appris à une cadence historique de 1024 ; seul un readout privé Active Exact-K K0..K6 est entraîné sur l'entrée native256, avec sélecteur calibré sur un split de compositions disjoint.
- Onset, Exact-K, Offset, Full Note, Frame Pitch : poids/architectures/sélecteurs/décodeurs historiques intouchables ; les sorties **MIDI de production** S708/S707 restent inchangées.
- Il ne s'agit **ni** d'une intégration en LIVE production, **ni** d'une promotion de modèle unifié, **ni** d'une validation sur CAL32 ou externe de S828, **ni** d'une mesure matérielle p99 garantissant de tenir chaque 5,805 ms.
- La précision du readout n'est pas certifiée `out-of-fold` pour le pré-entraînement du checkpoint S708 gelé ; FIT80 a déjà été utilisé en recherche. Éviter d'annoncer une généralisation indépendante.

## Règles pour les expériences S829+

1. Partir du code et des métadonnées S828 conservés dans cette branche de référence, ne modifier **que** la chaîne Active Exact-K ; ne pas faire régresser les cinq autres axes, ne pas modifier le MIDI historique signé.
2. Comparer chaque nouveau candidat à **S828 = 73,831146 %** sur les **mêmes 80 pistes et 249 411 trames actives** et publier les comptes de sous/surcomptages et K1..K6 ; S823 = 76,877523 % reste un objectif expérimental historique distinct.
3. Chaque test pilote de 5 pistes peut servir aux contrôles techniques, **jamais** comme score de référence ou comme preuve de gain musical. Chaque série doit comporter `resume.md` et `resultat.md`.
4. Refuser promotion, fusion dans LIVE ou changement de politique globale avant preuves de métriques + absence de régression + coût réel CPU/latence matérielle à 256 + validation hors FIT80.

## État GitHub

Cette branche `reference/active-exact-k-s828-midi256` est un **repère de recherche figé**. La PR S828 #132 reste DRAFT et n'est pas fusionnée. Le champ `reference: S708` présent dans les métadonnées des séries reste **inchangé** car il désigne la référence officielle protégée (six axes / MIDI public), différente de notre **nouvelle référence de développement Active-K**.
