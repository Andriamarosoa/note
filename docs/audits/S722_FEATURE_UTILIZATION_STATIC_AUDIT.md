# S722 — Audit statique de l'utilisation des caractéristiques NOTE (après S721)

**Date : 18 octobre 2026. Statut : audit du code, pas une ablation et pas de nouveau score.**

## Résultat S721 vérifié avant l'audit

[GitHub Actions 39114792725](https://github.com/Andriamarosoa/note/actions/runs/39114792725) : **succès technique**, S708 reste seul modèle éligible parmi 15 (1 témoin + 14 mélanges), aucune promotion. CAL `sustain_0.10` gagne marginalement sur **les six métriques agrégées** (Onset 74,26386%→74,29299%; Exact-K 62,04554%→62,07246%; Exact-K actif 56,27625%→56,29919%; Offset 77,32919%→77,35264%; full note 57,13193%→57,17197%; frame-pitch 76,46264%→76,48583%), **mais perd deux Offset individuels antérieurement corrects**, interdisant promotion par garde stricte. HELD et DEV ont rejoué S708 inchangé.

## Cartographie vérifiée des 288 entrées S708

1. `src/causal_note/strict_live.py:25-88` — **126 dimensions par hauteur** issues de 3 FFT Hann exclusivement passées (2048/4096/8192 échantillons), 8 harmoniques, amplitudes absolues/relatives, agrégats fondamentaux, différences dans le passé, comparaisons avec hauteurs voisines, historique d'énergie et flux spectraux moyens globaux.
2. `scripts/s701_train_causal_onset_shadow.py:49-100` — **39 dimensions par hauteur** : probabilités Onset/Sustain/Release et retards, fondamental/harmoniques 2/3, hausse d'énergie, relations avec les 12/19 demi-tons en dessous, distribution K, hauteur.
3. `src/causal_note/s704_joint_heads.py:15-22` — **27 dimensions**, état temporel passé à trois décalages 8/32/64 hops sur 9 variables d'énergie/postérieur.
4. `src/causal_note/s706_sharp_spectra.py:11-59` — **96 dimensions**, 24 log-amplitudes interpolées, 24 contrastes autour de chaque harmonique, 24 cos et 24 sin de cohérence de phase entre décisions successives.
5. `src/causal_note/s705_temporal_heads.py:39-61` — **288 dimensions totales** encodées vers `width=48`. Le graphe harmonique *latent* mélange uniquement ±12/±19 demi-tons. Quatre convolutions temporelles depthwise de longueur 3/dilatations 1,2,4,8 n'échangent pas directement d'informations entre *pitches*. La tête K exploite les 49 embeddings aplatis ; la tête pitch per-note reçoit les 7 logits K softmax, mais aucune représentation explicite des contributions de différentes cordes.
6. `src/causal_note/s704_joint_heads.py:53-89` et `src/causal_note/s708_cardinality_selection.py` — le sélecteur fort requiert Onset et Sustain simultanément, `armed`, délai de 65 ms, et au plus six notes actives. La branche faible utilise K, croissance temporelle et seuils de Sustain/Release. La politique S708 `count_confidence=1.1` n'applique pas de veto dur K ; **ne pas la rétablir sans ablation**.
7. `src/causal_note/s707_causal_release.py` et `src/causal_note/s716_s707_online.py` — 75D hazard NoteOff est séparé. Ne pas confondre contribution à l'identification de la *hauteur* avec durée MIDI.

**Attention** : les dimensions sont *transmises au modèle*, donc « non utilisées » n'est PAS démontré. Il faudrait les masquer par groupes, ou mesurer leur importance par ablation, à poids et splits contrôlés. « Sous-exploité » désigne ici une représentation/liaison/supervision explicite absente ou insuffisamment isolée dans le code, non un zéro de contribution mesuré.

## Priorités étayées par le code

### P0 — Corriger dans une nouvelle ABI le traitement des voisins inexistants pour les cordes graves
Dans `s701_train_causal_onset_shadow.py:67-75`, `sus12/ons12/sus19/ons19` sont correctement masqués quand la hauteur inférieure n'existe pas. **Mais**, aux lignes 90–91, `f_rise - f_rise[:, lower12]` et `f_rise - f_rise[:, lower19]` ne sont pas masqués. Les indices `lower12/lower19` sont bornés à 0 : pour MIDI40 (E2), le contraste devient artificiellement **toujours zéro** au lieu de comparer à « pas de fondamentale en dessous ». Pour MIDI41–51, la comparaison peut se faire contre MIDI40 sans lien harmonique. **Défaut de frontière certain d'après le code ; son impact musical reste à mesurer.** Ne pas modifier le calcul du checkpoint S708 sans réentraîner et changer explicitement l'ABI.

### P1 — Attaque indépendante par harmonique, et flux **local par pitch**
`CausalPitchFeatures._frame` calcule un `flux` de variation spectrale *moyenné sur tous les bins* à chaque résolution, diffusé ensuite identiquement à chaque hauteur. Il calcule également 4 différences passées des agrégats `score/fondamentale`, mais n'expose pas explicitement le vecteur des **8 montées individuelles d'harmoniques** par pitch et temps, ni un détecteur de synchronisme de plusieurs partiels. `sharp_track` contient des variations de phase mais elles sont traitées comme 96 entrées brutes sans expert spécifique de « nouvelle attaque indépendante ». Cela peut aider à distinguer une corde pincée d'une harmonique de corde déjà active ; preuve quantitative requise.

### P1 — Résidu de reconstruction harmonique / explication par les notes déjà actives
`strict_live.py:64-78` offre des scores harmonic-offset ±12/±19/±24 ; `s701:67-93` reprend des relations de posterior avec voisins bas. Cependant la nouvelle chaîne S708 **ne soustrait pas explicitement le spectre prédit par les hauteurs déjà actives** avant d'attribuer une émergence à une deuxième fondamentale. La tête harmonique latente est limitée à ±12/±19. Implémenter un indicateur de *nouvelle énergie spectrale non expliquée* par l'ensemble actif, entraîné uniquement FIT, avec compétition de vrais partiels.

### P1 — Interaction inter-hauteurs adaptée aux accords, pas seulement à l'octave/19 demi-tons
Dans `TemporalHeads.forward`, `pitch_shift(z,s)` n'est appelé que pour (-12,-19,+12,+19), puis les convolutions ont un noyau spatial 1. Les intervalles d'accords 3,4,5,7 demi-tons ne sont **pas explicitement mélangés au niveau latent** ; le réseau K voit globalement 49 hauteurs mais les logits de hauteur ne reçoivent que la synthèse K à 7 classes. Petit graphe local d'intervalles harmoniques/acoustiques + identité instrumentale pour limiter les faux positifs ; évaluer ablation + coût CPU.

### P2 — Mieux isoler Onset de Sustain, apprendre les négatifs « quasi-Onset »
Le réseau `TemporalHeads.pitch` produit les 3 logits Onset/Sustain/Release d'un même embedding, optimisé en moyenne BCE sur 49×3, `s706_train_sharp_temporal_loops.py:160-175` ; le décodeur fort requiert les deux probabilités hautes. Aucun **expert séparé de discrimination faux Onset vs vrai pincement** n'est branché dans la version S708. S718 : 427/525 FN DEV n'ont pas de preuves conjointes suffisantes, S720 : 159 TP supplémentaires / 472 FP supplémentaires avec pondération positive. Priorité aux négatifs difficiles FIT (harmoniques, sustain d'autres notes, glissés/transitoires), avec tête de vraisemblance d'attaque indépendante et suppression à score gelé.

### P2 — Robustesse tonalité/EQ/niveau avant de conclure que les graves sont absents
Les caractéristiques absolues/relatives et `max` par pitch/res/harmonique sont dépendantes de la balance spectrale ; `s706:83-96` n'expose pas d'augmentation EQ/Tone/RMS dans la préparation FIT. Les fondamentales faibles et le changement du potentiomètre Tone sont des cas de décalage de domaine à tester **en FIT uniquement**, avec contrôle sans augmentation. Pas d'affirmation de gain avant mesure.

### P3 — Qualité/confiance de la phase et compensation des notes proches
`s706_sharp_spectra.py:45-55` fournit `cos/sin` de cohérence de phase avec garde `norm>1e-12`, sans entrée dédiée de fiabilité du phase-diff à faible amplitude autre que les amplitudes déjà calculées ; ablation phase seul / magnitude seul / phase + fiabilité. Ne jamais désactiver sans test : cette famille peut déjà porter de l'information utile.

## Preuves externes et discriminantes

- S718 [run 39111634032](https://github.com/Andriamarosoa/note/actions/runs/39111634032): DEV 525 FN dont 321 preuves neuronales conjointes insuffisantes, 106 Onset trop bas, 3 cas uniquement `armed`.
- S720 [run 39113354252](https://github.com/Andriamarosoa/note/actions/runs/39113354252): apprentissage réellement refait sur FIT, renforcement polyphonique : **+159 TP et +472 FP** sur CAL ; Exact-K et frame-pitch progressent mais F1 Onset et Offset régressent.
- S721 [run 39114792725](https://github.com/Andriamarosoa/note/actions/runs/39114792725) : fusion par tête 10/25/50%, meilleure piste qualitative `sustain_0.10` ; régressions d'événements (2 corrects Offset perdus) la rendent non éligible.

## Prochaine expérience, S722, sans toucher aux références

FIT-only, trois contrôles comparables par entraînement et budget :
1. contrôle strict S708 ; 2. contraste local de nouveautés par partiel + masque correct de voisins hors registre ; 3. contrôle identique + résidu de spectre expliqué par notes actives. Une ablation séparée de l'extension inter-pitch ±3/4/7/24 et de la phase est préférable à tout combiner d'un coup. **CAL choisit**; HELD/DEV historique diagnostic; vraie source neuve ultérieure requise. Seuils et 6 métriques doivent être appariés sur une même liste d'événements, zéro régression de NoteOff auparavant corrects. Aucune promotion tant que cette preuve n'existe pas.
