# Correctif Onset — tête ErrorPrediction sans rétroaction sur le réarmement PCM256

**Nouvelle branche :** `codex/note-onset-error-head-no-feedback-256`. Modification indépendante du sous-module Onset, aucun changement de poids ni d'architecture des cinq autres axes, aucun MIDI officiel modifié.

## Problème REPRODUIT par les mesures S69D

[Run de référence en échec scientifique #39741883968](https://github.com/Andriamarosoa/note/actions/runs/39741883968) : 15 **vrais réseaux neuronaux MLP 69D** ont été entraînés sur la résolution PCM256. Le nouveau modèle conteste sur les événements S613 enregistrés 189 FP et 45 TP (soit 234 doutes, dont **80,77 % FP**), et le proxy évalué sur les événements S613 donne TP8054 / FP4411 / FN3213, F1 **67,8746 %**, contre **67,5874 %** natif. Pourtant, le véritable replay causal donne TP8089 / FP4675 / FN3178, **67,3214 %**, inférieur à S613 et au meilleur MLP63 (67,6704 %).

**Cause technique précise dans le code :** la stratégie précédente remplaçait `p[predict_correct < cut, onset] = 0` **avant** `CausalSelection.step()`. Or, dans `src/onset256/selector.py`, la ligne `self.armed |= onset < c.onset_reset` re-arme précisément la note lorsqu'elle voit un onset nul. Si l'événement est ensuite de nouveau suggéré, cette réinitialisation artificielle peut autoriser un NoteOn supplémentaire après le temps réfractaire. Modifier la probabilité utilisée **par la machine d'état** pour contester une attaque introduit donc des *nouvelles erreurs* non présentes dans les 12699 attaques du modèle original.

Cela constitue une explication **directement vérifiable du mécanisme logiciel**, et un test de non-régression synthétique reproduit une deuxième attaque fantôme à `sample=2816` lorsqu'un seul frame antérieur est annulé artificiellement. Le rôle exact de ce mécanisme dans chacun des 264 FP supplémentaires entre le proxy et le replay S69D n'est pas déterminé par le seul test et exigera le replay réel.

## Correctif

L'architecture reste composée de trois **têtes internes Onset** :

- `MainAttackHead` S613, entraîné à 256 ;
- `RescueAttackHead` S701, entraîné à 256 ;
- `ErrorPredictionHead` MLP69, **poids inchangés, SHA256 inchangé**.

Dans chaque décision PCM256, la tête ErrorPrediction estime `P(erreur)` **avant tout `CausalSelection.step()`**. Le `OnsetHeadSelector` conserve son rôle d'arbitre final, mais applique maintenant le doute **uniquement sur les événements Onset que la machine d'état aurait normalement émis**, sans réécrire la valeur `p[:, onset]` qui gère l'armement, la réattaque et l'activité.

```text
PCM256, nouvelles valeurs actuelles
        |
   Têtes Onset indépendantes
      S613 / S701 / MLP69 -> P(erreur) avant décisions
        |
        +--> CausalSelection MAIN : posterior S613 inchangé, état natif inchangé
        |
        +--> CausalSelection SHADOW : posterior rescue S701 inchangé
        |
        +--> CausalSelection privée de contrôle :
             posterior S613 inchangé ; mêmes événements exacts que MAIN
                    |
               OnsetHeadSelector
               - prend le doute MLP calculé avant
               - conteste seulement une proposition NOTE-ON déjà créée
               - ne peut créer de nouvelle attaque ni réarmer artificiellement
               - n'allonge ni fenêtre ni délai
```

**Invariant exécutable :** `mirror == native` sur chaque 256 hop, et `{NoteOn contestés/acceptés}` est un sous-ensemble des NoteOn émis par le Main au même pitch et au même sample. Une différence provoque immédiatement `AssertionError`.

Le protocole n'attend aucune information acoustique future et ne modifie ni le `refractory_frames`, ni la confirmation, ni `NoteOff`, ni `Exact-K`, ni `Offset`, ni le MIDI public. `predecision_intents` est explicitement **une sortie expérimentale Onset-only** : on ne peut pas brancher directement cette liste sur le MIDI de production sans traiter la correspondance NoteOn/NoteOff.

## Validation imposée

- Même S613 natif TP8099, FP4600, FN3168, F1 **67,5874155 %** sur **80 WAV GuitarSet / 306536 décisions PCM256**.
- Même MLP69 avec EXACTEMENT les poids signés de l'artefact [12411673219](https://github.com/Andriamarosoa/note/actions/runs/39741883968/artifacts/12411673219). Cinq réseaux sélectionnés et seuils sont **gelés** avant cette expérience, pas de réentraînement ni d'optimisation sur le nouveau test.
- Comparer aussi à l'ancien MLP63 F1 **67,6704308 %** et au S708 historique 1024 F1 **73,2520 %** (aucun de ses poids dans les modèles 256).
- Exiger que la sélection ne produise **aucun nouveau faux positif par divergence de machine d'état**. Le test scientifique retrace les NoteOn originaux et la sélection challenge à chaque hop réel.
- GitHub Actions : `.github/workflows/onset-no-feedback-256.yml`, tests natifs et vrai replay audio, résultat signé `report.json`.
- La cohorte FIT80 a été partiellement vue à l'entraînement S613 et S701 (**20/80 et 48/80**). Même un gain positif ne constitue PAS une validation extérieure du système complet. Ni promotion, ni modification des autres caractéristiques sans vraie preuve externe.

## Scores nouveaux

**Non encore mesurés au moment du commit.** Le proxy de 67,8746 % est une hypothèse utile, mais il ne doit jamais être annoncé comme le score de la nouvelle sélection avant le vrai replay sur les WAV.
