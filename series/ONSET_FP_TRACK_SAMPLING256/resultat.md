# Résultats — Onset FP H_FP_TRACK_SAMPLING_03 issue #144

**STATUT : entraînement et replay réel à vérifier. Aucun résultat nouveau confirmé au commit initial.**

Objectif 80 WAV GuitarSet / 306536 décisions PCM256 : F1 vrai **>67.9713963%** ET FP vrai **<4519** face au meilleur contrôle FP69 du run #39775357133 (TP8127 FP4519 FN3140). Contrôle quatre têtes TP8160 FP4605 FN3107 inchangé obligatoire.

Deux modes indépendants et préenregistrés :
- `track_uniform` : même N_train, probabilité d'échantillonnage inverse du nombre d'attaques par piste TRAIN.
- `track_class` : même N_train, TP/FP environ 50/50, pistes uniformes dans chacune de ces classes.

Cinq MLP69 réellement réentraînés par mode, architecture strictement identique, même 3 TRAIN / 1 CAL / 1 OUTER, poids contrôle70/74 gelés, CAL seuil avant OUTER et verdict après vrai replay sur les 80 WAV. Aucun score proxy de CAL présenté comme résultat Live.

Chaque échec expérimental exige audit `REFUTED_OR_INCOMPLETE` et archivage des contre-exemples, conformément à [#140](https://github.com/Andriamarosoa/note/issues/140).

Autres cinq axes, official MIDI, ancienne branche de référence intacts. Aucun entraînement 1024 ni future audio. FIT80 historiquement chevauché pour S613/S701 : pas validation externe.
