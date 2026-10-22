# Onset — audit reproductible des 4 411 fausses attaques résiduelles (PCM256)

**Contexte vérifié :** après les trois têtes `MainAttack`, `RescueAttack`, `ErrorPrediction` MLP69, sur les 80 WAV GuitarSet, **TP8054 / FP4411 / FN3213 / F1=67,8746 %** (run scientifique [#39749340403](https://github.com/Andriamarosoa/note/actions/runs/39749340403)). La quatrième tête expérimentale `FNRecovery` indépendante augmente le score à **67,9095 %**, mais augmente les FP à **4 605** (plus 194 FP, plus 106 TP, moins 106 FN).

Le run « entraîner FNRecovery sur tous les négatifs TRAIN » [#39765038340](https://github.com/Andriamarosoa/note/actions/runs/39765038340) a un **résultat scientifique négatif, pas une panne** : TP8100 FP4500 FN3167 F1 **67,8761 %**, amélioration des FP face à l'ancienne quatrième tête, mais dégradation F1 par rapport à son record. H_FN_NEGCOVERAGE_01 [issue #141](https://github.com/Andriamarosoa/note/issues/141) est insuffisante/réfutée comme solution complète; NE PAS promouvoir les nouveaux poids.

## Où sont les FP ? Audit sur les checkpoints réellement entraînés à PCM256

Ce nouveau script reconstruit les 12 699 propositions S613 depuis le rapport d'événements original qui fut créé durant **un replay réel** ; charge les cinq réseaux MLP69 de la troisième tête avec SHA256 ; applique exactement les cinq seuils composition-OOF gelés ; confirme **8 054 TP et 4 411 FP conservés**. Il n'altère aucune sortie Live et n'utilise **pas** les labels pour décider. La GT est uniquement une annotation historique apposée **après** le replay audio précédent.

Sur le groupe conservé :

| Condition post-audit | TP | FP | Proportion FP |
|---|---:|---:|---:|
| Solo, 40 WAV | 2 435 | 863 | 26,2 % |
| Accompagnement, 40 WAV | 5 619 | **3 548** | 38,7 % |
| E2–B2 | 602 | 624 | **50,9 %** |
| C3–B3 | 3 561 | **2 145** | 37,6 % |
| C4–E6 | 3 891 | 1 642 | 29,7 % |
| Fondamentale inférieure plus forte | 1 929 | 1 239 | 39,1 % |
| Note inférieure annotée à intervalle harmonique ±80ms (**GT posthoc seulement**) | 972 | 1 050 | 51,9 % |

Ces catégories **se chevauchent** et ne sont pas des causes démontrées. Elles ne doivent **jamais** servir comme masques automatiques ni entrer dans les features avec annotation de vérité terrain.

## Nouvelle trace causalement exploitable

La tête 69D voit déjà une partie des spectres, des harmoniques et la probabilité des autres têtes. Elle ne voit pas nécessairement **l'historique récent des attaques effectivement proposées pour les autres hauteurs**. Le script reconstruit donc, avant chaque NoteOn Main, l'âge depuis les **précédentes** attaques d'une note de hauteur inférieure à +12, +19, +24 et +28 demi-tons. Il ne prend aucune note de l'instant en cours ou du futur, seulement l'état des anciennes décisions strictement avant ce hop 256.

- La condition parent-harmonique / vraie attaque **PASSÉE** à moins de 80ms est un facteur de risque descriptif. L'audit exploratoire indiquait environ **782 FP et 678 TP** dans cette condition parmi les propositions déjà retenues ; il faut vérifier ces nombres dans GitHub Actions.
- **Un veto fixe serait mauvais :** il rejetterait AUSSI 678 bonnes attaques, au détriment du F1. Sur les événements figés, sa F1 serait environ **66,24 %**, en baisse face au score 67,87 %. Ce n'est qu'un contre-factuel, pas un second replay WAV. En outre la majorité des 4 411 FP reste hors de ce seul groupe.
- Nous exportons, pour chaque événement reconnu, les **69 caractéristiques acoustiques causales** + **4 âges de parents harmoniques passés** + **P(correct) du MLP69 gelé** = **74 valeurs** utilisables dans une future tête indépendante, sans aucune GT dans les 74 caractéristiques. Les étiquettes TP/FP demeurent uniquement pour la supervision de TRAIN et pour l'évaluation différée.

## Nouvelle hypothèse scientifique, à NE PAS considérer prouvée

[Issue préenregistrée **H_FP_PARENT_TIMELINE_01 #142**](https://github.com/Andriamarosoa/note/issues/142).

Tester **un réseau neuronal réellement entraîné** pour distinguer les mauvaises attaques avec l'historique causal des parents harmoniques en plus du 69D, versus un contrôle sans ces quatre valeurs, avec EXACTEMENT mêmes compositions TRAIN/CAL/OUTER, mêmes graines/architecture, calibration sur CAL uniquement et vraie comparaison au décodeur causal WAV. L'hypothèse sera **réfutée ou incomplète** si la nouvelle tête n'abaisse pas les FP sans faire régresser le F1 par rapport au meilleur quatre-têtes **67,9095 %**. Le taux d'erreur est déjà très élevé dans le médium et l'accompagnement hors des relations harmoniques ; une tête spécialisée ne doit pas ignorer ces autres erreurs.

**Le présent audit est descriptif, PAS le test de cette intervention.** Il ne lance aucun entraînement neuronal, ne modifie pas les anciennes têtes et ne promet aucun nouveau F1.

## Fichiers

- `scripts/audit_onset_fp_residual256.py` : vérification des cinq checkpoints SHA256 MLP69, reconstruction TP/FP exacte, statistiques par texture/registre/composition, historique parent strictement passé, vecteurs 74D et contre-factuel non promu.
- `tests/test_fp_residual_audit256.py` : hop256, fenêtre exacte 3 528 échantillons (=80 ms), absence de fuite temporelle et ABI.
- `.github/workflows/onset-fp-residual-audit256.yml` : source originale SHA-pinnée, tests et audit, archive du résultat.
- `series/ONSET_FP_RESIDUAL256/resume.md`, `resultat.md` : preuves et statut de réfutation.
- [Règle générale d'audits réfutables #140](https://github.com/Andriamarosoa/note/issues/140).

**Contraintes :** source entraînée PCM256 à 44,1 kHz, aucune attente future ni ajout de latence, pas de poids entraînés à 1024, autres cinq axes gelés, aucune décision de production MIDI modifiée. Les poids S613/S701 ont déjà vu certaines pistes FIT80 durant leur historique d'entraînement, donc ces résultats ne prouvent pas la généralisation externe.
