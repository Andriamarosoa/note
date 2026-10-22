# Onset — la prédiction d'erreur devient une troisième TÊTE du modèle (PCM256)

**Branche indépendante** : `codex/note-onset-integrated-error-head-256`. **Trois têtes**, un seul axe `Onset`. Aucune branche Exact-K, Active-K, Frame Pitch, Offset, Full Note ni MIDI officiel n'est modifiée.

## Architecture réelle : tête → avis → sélection privée Onset

```text
PCM 44 100 Hz / 256 nouveaux samples
              |
     S613 Onset frontend (poids entraînés nativement à 256)
              |
  +-----------+-----------------------+
  |                                   |
  v                                   v
MainAttackHead                    RescueAttackHead
poids S613 256                    poids S701 256
p(onset/sustain/release)           P(rescue) + 39D causal
  |                                   |
  +-----------------+-----------------+
                    |
                    v
             ErrorPredictionHead
         VRAI MLP entraîné à 256 (63→64→32→2)
          P(erreur du candidat Onset)
                    |
                    v
          OnsetHeadSelector indépendant
       - référence main : inchangée
       - référence rescue : inchangée
       - avis tête erreur : DOUTE / NON-CONTESTÉ
       - sélection expérimentale challenge :
         peut proposer une autre décision, sans
         muter les références ni le MIDI officiel
```

La tête de risque n'est PAS un outil de post-traitement : elle fait partie de `src/onset256/heads.py` sous le nom **`ErrorPredictionHead`**, avec ses propres poids MLP, un compteur de décisions indépendant et une sortie explicite `HeadOpinions.error_probability`. L'architecture `OnsetHeadStack` comprend trois instances nommées : `main_attack`, `rescue_attack`, `error_prediction`. Les caractéristiques sont partagées en lecture à l'intérieur d'Onset uniquement ; aucun autre axe ne reçoit ni n'écrit l'état d'une tête.

**Ordre temporel de chaque hop256 :** 1) audio disponible ; 2) prédictions des trois têtes ; 3) sélection Onset main/récupération/challenge. La sélection ne voit jamais d'échantillon futur. Chaque décision se termine exactement au même `sample`. Le modèle principal ne dépend pas du verdict de risque.

## Différence entre « doute » et « rejet »

La sortie `Decision.head_doubts` attache à chaque attaque originale S613 `P(erreur)`, `P(correct)` et `contested` selon la calibration du MLP. **Doute** = opinion d'un modèle appris, **pas** vérité terrain et **pas** suppression automatique du résultat du modèle principal. Trois sorties sont mesurées séparément :

- `Decision.intents` : attaques principales S613 inchangées.
- `Decision.candidate_intents` : attaques de récupération S701 inchangées.
- `Decision.predecision_intents` : variante privée où `OnsetHeadSelector` utilise le doute MLP et peut contester des attaques. Aucun changement du MIDI officiel.

Le routeur n'a accès qu'aux probabilités et à la calibration apprise sur `CAL`. Un seuil ne constitue pas la tête : **ce sont les poids MLP** qui apprennent à reconnaître des configurations FP/TP, y compris la concurrence fondamentale–harmonique. Le doute est prédit avant `NoteOn`, mais à partir de l'audio courant/passé et non d'un futur son.

## Entraînement et expérience

Le workflow `.github/workflows/onset-predecision-neural256.yml` réentraîne cinq véritables réseaux MLP (63→64→32) sur l'audio GuitarSet : 3 compositions TRAIN / 1 CAL / 1 OUTER. Chaque modèle privé est sauvegardé avec ses poids `.joblib`, ses métadonnées et SHA256, séparément pour le fold OUTER, et ne doit pas avoir vu son propre OUTER. Une version groupée strictement expérimentale est également produite, sans promotion.

Le test scientifique **obligatoire** ne se contente pas du filtrage d'événements archivés : il réévalue les **80 WAV**, **306 536 décisions 256**, avec **vraies prédictions de la tête erreur à chaque hop avant le sélecteur** et des états indépendants. Il doit produire :

1. `ROC-AUC` de la tête risque, une mesure de prédiction d'erreur distincte du F1 musical ;
2. `challenged_tp`, `challenged_fp`, `retained_tp`, `retained_fp` : combien de bonnes attaques le MLP mettait en doute, combien de fausses ;
3. F1 de la variante vraiment causale `predecision_intents`, comparé sur le même audio à `intents` S613 intacte et à la référence historique S708 ;
4. Aucune projection des poids historiques S708 (entraînés à 1024), aucune annotation GT avant la fin de l'inférence, aucun retard ajouté, aucune influence sur les cinq autres axes ni sur le MIDI de production.

## Contrôles non négociables

- **Poids ET décisions à 256** (44 100 Hz : 5,805 ms entre décisions), aucun `lookahead`, aucun `hold`, aucune rétrodatation. Une cadence d'appel de 5,805 ms n'est pas une mesure de latence p95/p99 CPU ou de latence physique : cela reste à mesurer sur machine réelle.
- Baseline S613 : **TP 8099 / FP 4600 / FN 3168**, **67,5874 % F1** sur FIT80 ; S701 **67,6563 %** ; référence historique S708 entraînée à 1024 : **73,2520 %**. Toute régression mesurée doit conserver son verdict scientifique rouge ; aucune « correction » du test de non-régression.
- Attention : **20/80 pistes FIT80 ont servi à l'entraînement initial S613 et 48/80 à S701**, malgré le crossfit du nouveau MLP. Les résultats FIT80 permettent l'étude et l'ablation, mais **ne constituent pas une validation indépendante du pipeline complet**. La promotion finale exige des prises et/ou un corpus absents de tous les entraînements et calibrations.
- Tant que la tête n'a pas obtenu des poids entraînés et testés, elle n'est disponible que sous forme de code ; il est interdit d'affirmer qu'elle atteint un nouveau score.

## Fichiers

- `src/onset256/heads.py` : les **trois têtes indépendantes**, `HeadOpinions`, `Doubt` et `OnsetHeadSelector`.
- `src/onset256/runtime.py` : exécute `OnsetHeadStack.predict` **avant** les sélections, expose le doute explicite sans écraser l'original.
- `src/onset256/predecision_risk.py` : le MLP neuronal avec son propre checkpoint et ABI 63D vérifié.
- `scripts/train_onset_predecision256.py` : entraînement 5-fold OOF et seuil CAL seulement.
- `scripts/replay_predecision256.py` : audit du doute et vrai score Onset causal, par piste/registre/texture selon données disponibles.
- `tests/test_embedded_error_onset_head256.py` : ordre temporel, indépendance des sorties, vrai doute, latence causale au hop, présence des trois têtes.
- `.github/workflows/onset-predecision-neural256.yml` : tests + entraînement + replay signé; résultats et artefacts restent indépendants.

**État : expérimental ; ne pas promouvoir sans score scientifique et validation externe.**
