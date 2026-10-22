# NOTE — quatrième tête Onset `FNRecovery` indépendante, réellement entraînée à PCM256

**Branche :** `codex/note-fn-recovery-trained256`. Uniquement Onset expérimental ; aucun commit sur les cinq autres caractéristiques ni sur le MIDI officiel.

## Raison justifiée par l'audit préenregistré

L'audit causal natif 256 [run #39754613388](https://github.com/Andriamarosoa/note/actions/runs/39754613388) a vérifié sur **80 pistes originales GuitarSet et 306 536 décisions PCM256** que les 3 168 FN d'origine S613 se décomposent en :

- 426 cas `P(Onset) < 0,12`, 1 861 `P(Onset) < seuil principal` : **2 287 / 3 168 = 72,19 %** avec Onset faible ;
- 175 Sustain trop faible ; 269 coïncidence Onset+Sustain absente ;
- 130 sélecteur non armé ; 307 sélecteur, budget, appariement ou timing.

**La GT n'entre jamais dans l'inférence.** Les trois détecteurs de *propositions* causaux préréglés ont identifié après coup 1 923 / 3 168 FN avec le plus sensible (60,7 %), 1 668 avec le modéré et 1 367 avec le prudent. Les 1 923 sont une **couverture ORACLE** sur des candidats, pas des notes reconnues sans connaître la GT. Rien ne prouve encore qu'une vraie tête supervisée puisse en sélectionner suffisamment sans augmenter FP.

Le précédent modèle triple tête **ErrorPrediction MLP69 sans feedback** a réalisé TP=8 054 / FP=4 411 / FN=3 213, soit **67,8746 % F1**, suite au replay scientifique [run #39749340403](https://github.com/Andriamarosoa/note/actions/runs/39749340403). Parmi les 3 213 FN, 3 168 sont originaux et 45 ajoutés par contestation des TP.

## Quatre têtes, un axe Onset

```text
Audio MONO PCM44.1 kHz : 256 échantillons nouveaux
              |
    frontal harmonique natif entraîné 256 (lecture partagée)
              |
      +-------+----------+-------------------+------------------+
      |                  |                   |                  |
 MainAttackHead    RescueAttackHead   ErrorPredictionHead   FNRecoveryHead
 S613 natif256      S701 natif256      MLP69 correct/FP      NOUVEAU MLP69
 attaques          récupération      valide/conteste       prédit P(FN correct)
 standard          ancienne          les attaques Main     depuis les candidats
      |                  |                   |                  |
      +-------+----------+-------------------+------------------+
                            |
                  sélections privées ONSET
                  main / rescue / risk / FN candidate
                            |
                    sélection finale ONSET
               - 3 anciennes sorties inchangées
               - + FNRecovery accepte de nouveaux NoteOn
               - pas de feedback sur le réarmement Main
               - pas d'attente / pas de future trame
               - MIDI officiel inchangé
```

Les quatre têtes disposent de **leurs poids et compteurs propres** ; la quatrième tête est située dans `src/onset256/heads.py`, avec une sortie propre `HeadOpinions.fn_recovery_probability`. Son véritable réseau supervisé est `StandardScaler + MLPClassifier(64,32)` à 69 caractéristiques exclusivement disponibles AVANT la décision : 63 caractéristiques S613/S701/harmoniques causales et 6 valeurs internes Onset courantes/passées. Un quatrième `CausalSelection` privé génère les candidats à seuil faible. La tête FNRecovery les accepte/rejette, à l'instant même du 256, avec un veto causal de doublons de notes récentes, **sans réécrire l'état des trois anciennes têtes**.

La sortie experimentale `Decision.fn_unified_intents` conserve tous les événements acceptés de la tête ErrorPrediction MLP69 et y ajoute les vrais candidats que `FNRecoveryHead` approuve. `Decision.intents`, `candidate_intents`, `predecision_intents` sont inchangés. La sortie fn_unified n'est pas envoyée au MIDI officiel, et cette expérimentation est strictement limitée aux NoteOn : l'association NoteOff doit rester un travail de validation ultérieur.

## Entraînement réel et choix CAL (aucun proxy présenté comme F1 réel)

La série utilise les **données audio natives réellement déduites de l'audit FN**, écrites après l'inférence en `candidate_proposals.jsonl`. Le modèle apprend les étiquettes `recoverable_fn` contre `not_fn`. Le même vecteur 69D est produit pour chacun des 49 pitches à chaque 256 dans le LIVE. Les trois sources de proposition sont figées : `sensitive`, `moderate`, `cautious`.

- **5 plis par composition** : 3 TRAIN / 1 CAL / 1 OUTER (16 pistes/composition). Les poids de la nouvelle tête, sa politique de proposition et son seuil sont déterminés exclusivement avec TRAIN et CAL. OUTER n'est jamais utilisé avant le choix figé.
- **15 vrais réseaux MLP** entraînés (3 politiques × 5 plis), chacun exclusivement sur les candidats de TRAIN. Les positives FN sont conservées, les négatives sous-échantillonnées uniquement sur TRAIN selon un maximum 3:1. Cinq checkpoints sélectionnés selon CAL, chacun avec SHA256, liste TRAIN/CAL/OUTER, score de CAL et provenance d'architecture.
- Le baseline frozen ErrorPrediction MLP69 est **reproduit à partir des checkpoints précédents**, sans nouveau tuning. Un simulateur candidat-proxy applique le **même arbitre temporel causal** que celui du flux, mais cela reste un proxy hors audio.
- **Deuxième passage obligatoire sur les 80 WAV d'origine, un vrai `Stream` audio** : les quatre modèles prédisent chaque trame PCM256, chaque sélecteur privé évolue causalement, seuls les événements de NOTE sont comparés à la GT APRES la fin du WAV. Chaque événement récupéré, nouvelle erreur et régression est compté par composition et piste. Ce passage mesure enfin TP/FP/FN/F1 vrais.
- Contrôles exacts du baseline S613 `8099/4600/3168`, et du troisième modèle `8054/4411/3213` : divergence = échec technique. Le critère scientifique de promotion est **F1 strictement supérieur à 67,8746 % sur le replay**, sans toucher les autres têtes. Le run échoue explicitement si la quatrième tête n'apporte pas de gain. Aucun paramètre ne peut être choisi après ouverture d'OUTER.

## Nouvelle règle d'audit (issue #140)

**Hypothèse préréglée H_FN_RESCO_01 :** au moins un spécialiste FNRecovery doit parvenir à sélectionner sur CAL des attaques véritablement manquées assez proprement pour améliorer la F1 causale OUTER agrégée face au MLP69 figé. Le succès ne sera reconnu que si le F1 du **flux real audio** dépasse 67,8746 %. Si ce n'est pas le cas, marquer l'idée « les candidats bas seuil suffisent pour une récupération utile » `REFUTED_OR_INCOMPLETE` ; auditer les faux positifs et les oublis de la nouvelle tête et ne PAS régler aveuglément les seuils après avoir vu OUTER.

**Limites :** GuitarSet FIT80 a des recouvrements historiques avec le training des poids S613 (20/80) et S701 (48/80). Même une amélioration OOF de la quatrième tête n'est **pas une validation indépendante du pipeline complet**. Les entraînements sont natifs 256 (pas de poids S708 entraînés1024) ; pas de délai volontaire, mais la latence matérielle p95/p99 doit rester à mesurer.

## Fichiers

- `src/onset256/fn_recovery.py` : modèle MLP FNRecovery, checkpoint SHA, proposition et arbitre causal.
- `src/onset256/heads.py` : quatre têtes indépendantes, quatrième prédiction avant la décision.
- `src/onset256/runtime.py` : nouveaux `fn_recovery_intents` et `fn_unified_intents` privés à 256 PCM.
- `scripts/train_onset_fn_recovery256.py` : 15 vrais MLP et CAL composition-disjointe ; proxy uniquement.
- `scripts/replay_onset_fn_recovery256.py` : véritable replay GuitarSet causal, audit des progrès/régressions.
- `tests/test_fn_recovery_4th_onset_head256.py` : invariance trois premières têtes, entraînement MLP réel, propositions, timing causal et anti-duplication.
- `.github/workflows/onset-fn-recovery-trained256.yml` : provenance audio/traces/poids SHA, tests, entraînement et replay.
- `series/ONSET_FN_RECOVERY256/resume.md` / `resultat.md` : bilan complet et statut sans promotion.

**État au commit : code expérimental et protocole ; aucun nouveau score FNRecovery mesuré avant workflow complet.**
