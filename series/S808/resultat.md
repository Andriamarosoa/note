---
serie: S808
statut: en_cours
type: experience
reference: S708
branche: codex/note-s808-independent-birth-continuity-fit80
run_id: en_attente
commit_sha: en_attente
promotion: false
---
# S808 — résultats réels du replay local FIT80

## Exécution

Test Python exécuté localement sur les **80 pistes GuitarSet** des traces OOF issues du [run source S800 #39352880035](https://github.com/Andriamarosoa/note/actions/runs/39352880035). Toutes les pistes ont vérifié la parité des 12 compteurs d'événements MIDI natifs S708 contre les JSON signés. Quatre tests unitaires négatifs PASS. **Pas de run scientifique S808 GitHub Actions, ni d'audit autonome externe**, à cette étape. Script et rapport complet inclus dans l'archive `S808_test_reel_FIT80.zip`, vérifiable par SHA256 `5be93836634c628aaf595c3adec7afb2acab0301dc44fccbf7a1da207b070feb`.

## Métriques

Toutes ces valeurs concernent le **même FIT80** et les vrais événements MIDI256, et non la validation officielle CAL32 :

| Politique | Onset F1 | Exact-K | Active Exact-K | Offset accuracy | Full Note F1 | Frame Pitch F1 |
|---|---:|---:|---:|---:|---:|---:|
| S708/S800 original FIT80 | 73,2520 % | 61,6972 % | 57,0195 % | 69,8862 % | 50,6528 % | 78,7278 % |
| birth_only | 68,8605 % | 62,6843 % | 58,2881 % | 60,9245 % | 41,5009 % | 79,5923 % |
| birth_plus_continuity | 68,8605 % | 62,6843 % | 58,2881 % | 60,9245 % | 41,5009 % | 79,5923 % |
| sustain_recovery_plus_continuity | 68,9150 % | 63,2608 % | **58,9998 %** | 60,9277 % | 41,5375 % | **80,3019 %** |

**Aucun candidat n'est admissible : toutes les politiques font nettement régresser Onset, Offset et Full Note F1.** La récupération ajoute 2 394 NoteOn synthétiques sur 80 pistes, dont 346 déclenchés via le canal sustain-recovery. Un changement de **+1,9803 point** Active Exact-K et **+1,5741 point** Frame Pitch F1 n'autorise **pas** la promotion, face à −4,3370 points Onset, −8,9585 points Offset et −9,1153 points Full Note F1.

Les politiques birth-only et birth-plus-continuity sont strictement identiques (comptes et scores) : à ce seuil figé, la tête de continuité n'a provoqué **aucune décision**. Le test n'établit donc pas qu'une architecture de continuité apporterait un bénéfice ; il montre seulement l'absence d'effet du mécanisme essayé.

## Comparaison à S708

Les scores officiels S708 sur CAL32 demeurent : Onset F1 74,2639 %, Exact-K 62,0455 %, Active Exact-K 56,2762 %, Offset accuracy 77,3292 %, Full Note F1 57,1319 %, Frame Pitch F1 76,4626 %. CAL32 n'a pas été ouvert pendant l'expérience ; il est méthodologiquement interdit de comparer ces deux corpus comme une évolution directe. Aucun changement de logiciel de production, NoteOff natif, inférence microphone ou référence signée S708.

## Audits

| Rôle | Statut | Preuve |
|---|---|---|
| Données | PASS local | 80/80 FIT80, douze comptes source par piste identiques aux JSON d'origine |
| Acoustique | PASS local | Aucune trame future, épreuve de préfixe causal incluse |
| Architecture | PASS local | Deux modèles distincts et protection des événements source |
| Apprentissage | PASS local | Cinq folds, composition cible jamais dans le train de la tête |
| Métriques | PASS local | Six scores MIDI256, veto global et sur les cinq compositions |
| LIVE | UNKNOWN | Zéro buffering/hold ajouté, mais latence CPU/iRig non mesurée |

Ces vérifications locales **ne constituent pas** des audits CI indépendants, pas les six agents Work personnalisés, pas les trois modèles NVIDIA ni un consensus 9/9. La performance LIVE physique reste inconnue.

## Verdict

**S808 FIT80 local : tous les candidats FAIL, aucune amélioration promouvable.** La récupération de voix augmente bien deux métriques, mais détruit la précision d'attaque et de fin des notes. L'hypothèse « simple séparation birth/continuity résout Active-K » n'est pas confirmée. Le canal continuation essayé n'a eu aucun effet sur cette sélection de données. Conserver strictement S708 et ne pas tourner en boucle sur les seuils FIT déjà observés.

## Prochaine action

Avant d'entraîner une variante nouvelle, instrumenter la disponibilité de la tête de continuité et ses exemples de décision, distinguer les fausses attaques des véritables notes récupérées par composition et démontrer sur un nouveau split que les améliorations d'Active-K préservent aussi Onset, Offset et Full Note F1. Interdire absolument une confirmation qui attend après l'attaque ou prolonge un offset pour augmenter artificiellement la F1.
