# S585–S591 — Attaque, maintien et réattaque : protocole fixé avant le sixième test

Cette expérience poursuit S584. S578 Global28 demeure le témoin (59,34 % sur la **cinquième** cohorte S577, pas le score attendu sur la sixième).

## Amendement sur les seules métadonnées — avant tout nouveau WAV/MIDI

Le premier workflow [38678912773](https://github.com/Andriamarosoa/note/actions/runs/38678912773) a arrêté S586 : les 33 sources exposées couvrent 37 composantes de compositeurs. **Seuls Enescu et Moussorgski restent dans l’intégralité de MAESTRO**, tous les splits confondus ; leurs sept enregistrements sont longs (1 364–2 009 s) et tous dans train. Aucune source ne satisfait donc la limite initiale de 1 000 s.

L’amendement conserve toutes les exclusions et le même ordre SHA. Le test porte sur **deux morceaux**, un par compositeur restant, durée maximale 2 300 s et plafond technique WAV 500 Mo. Les extraits restent +15 s / 12 s. Cette décision est fondée exclusivement sur les métadonnées, avant le moindre audio, MIDI ou score de ces morceaux. Le manifeste est [v273_s586_two_remaining_composers_metadata.json](v273_s586_two_remaining_composers_metadata.json).

- Moussorgski : *Pictures at an Exhibition*, enregistrement 2004.
- Enescu : *Suite Op. 10*, enregistrement 2006.

Les poids S585 sont récupérés dans le premier job d’entraînement, **sans relancer l’entraînement**. Les trois politiques, seuils et cibles ne changent pas. Deux extraits, soit 24 secondes, constituent seulement une **sonde indépendante limitée**, pas la confirmation à cinq compositeurs initialement prévue. Les intervalles à deux sources ne permettent pas de conclusion générale. Aucun remplacement de référence ne peut en découler.

Les paragraphes suivants conservent la trace du protocole initial ; le nombre de sources et les limites opérationnelles sont remplacés par cet amendement.

## Hypothèse et contrôle

La suppression systématique d'une même hauteur pendant 100 ms peut perdre de vraies réattaques. Une tête distingue quatre cibles : hauteur inactive/résonance, touche ou pédale maintenue, nouvelle attaque, réattaque. Les états MIDI sont uniquement des **cibles d'entraînement**. Ils ne sont jamais fournis aux modèles en inférence. La pédale donne une approximation du maintien acoustique, pas un NoteOff mesuré.

- Exactement les **28 mêmes sources** que S578 ; S577 n'entre pas dans l'entraînement.
- Les mêmes caractéristiques audio 52D, les mêmes cinq folds de composantes de compositeurs/co-compositeurs. Aucune affirmation nouvelle de causalité des fenêtres audio historiques.
- Le témoin OOF S578 doit retrouver exactement TP/FP/FN avant que la nouvelle expérience soit acceptée.
- Toutes les attaques positives historiques sont conservées. Les candidats négatifs voisins à ±80 ms d'une vraie attaque de même hauteur sont exclus des cibles ambiguës.
- Poids totaux : 1/2 attaques (nouvelles et réattaquées), 1/4 maintien, 1/4 absence. ExtraTrees : 135 arbres/fold, 210 arbres final, profondeur 19, feuille ≥4, max_features 0,8, seed 58501.

## Trois politiques fixées

1. **S578_global28** : poids originaux, p≥0,7, dédup100, veto harmonique0,8.
2. **S585_state_strict100** : somme P(nouvelle attaque)+P(réattaque)≥0,7 ; même dédup100 et veto.
3. **S585_state_restrike** : même tête ; dans les 100 ms après une émission, autoriser la répétition uniquement si P(réattaque)≥0,5 et P(attaque totale)≥0,7.

Les naissances Basic Pitch restent protégées avec leur comportement original. Aucun seuil n'est choisi sur la sixième vérité. La comparaison stricte/apprise sépare l'effet du décodeur de celui de la tête.

## Ordre des preuves

Un seul workflow, avec groupe de concurrence fixe. Deux travaux indépendants entraînent sur les anciens morceaux et préparent les nouveaux WAV. Le gel dépend de leur succès. Le score lit un **artefact de prédictions déjà publié par un autre job** avant d'ouvrir un MIDI cible.

- **S586** : métadonnées officielles uniquement ; cinq compositeurs distincts du split MAESTRO train, durée 28–1000 s, SHA ordre fixé ; exclusion des 33 fichiers déjà exposés et de leurs compositeurs/co-compositeurs. Si cinq noms éligibles manquent, arrêt explicite, sans assouplir les exclusions.
- **S587–S589** : vrai V88 original, vrai Basic Pitch, caractéristiques des 49 hauteurs sur les mêmes WAV, contrôles SHA ; aucune annotation cible.
- **S590** : poids d'origine vérifiés, trois sorties NoteOn complètes et probabilités figées avec SHA.
- **S591** : ouverture des MIDI ensuite ; hauteur exacte + onset ±30/50/80 ms, appariement 1:1, tous FP/FN, scores par morceau et registre, liste des vrais TP perdus/récupérés et des FP. Intervalle bootstrap par morceau descriptif, limité à cinq sources.

Ce test concerne du piano, cinq extraits de 12 s, et ne certifie pas l'absence d'utilisation des morceaux dans toutes les anciennes branches. Il ne constitue pas un benchmark officiel MAESTRO TEST.

## Protection de la référence

Seuls de nouveaux scripts, ce document et le workflow sont ajoutés. Le système Exact-K complet S80, ses poids, les fichiers Live et le pont MIDI ne sont pas modifiés. Le test NoteOn ne constitue **pas** un rejeu complet Exact-K/GuitarSet ni une validation multi-instruments ou de latence Live. Aucune promotion automatique, quel que soit le résultat.

Le workflow de continuation est terminé avec succès. Voir le [verdict complet S585–S591](README_V273_S585_S591_FOUR_STATE_RESULTS.md), les pertes individuelles et les limites du test à deux sources.
