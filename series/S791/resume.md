---
serie: S791
statut: termine
type: experience
reference: S708
branche: codex/note-s791-framepitch-complementary-experts
run_id: 39325760225
commit_sha: 6361a01a846f257c6c2039f243b1fe6bbb4bf536
promotion: false
---

# S791 — Résumé

## Objectif

Déterminer expérimentalement combien d'experts complémentaires sont utiles à Frame Pitch, plutôt que fixer arbitrairement une ou six têtes. Comparer l'architecture S791 multi-experts avec la tête spécialisée S790, sans changer S708/S707 ni le flux MIDI final.

## Hypothèse

Les erreurs Frame Pitch varient selon les fondamentales graves, les voisinages harmoniques, les transitions Sustain et les contextes généraux. Des modèles distincts entraînés sur ces populations pourraient améliorer une décision **par fragment**. Le nombre retenu doit émerger de la validation FIT par composition, non d'une valeur décidée à l'avance.

## Modifications

- `src/causal_note/s791_framepitch_expert_bank.py` : quatre spécialisations candidates, plus la source S708 inchangée, seize combinaisons possibles.
- `scripts/s791_train_framepitch_complementary.py` : experts séparés, validation croisée sur cinq compositions FIT80, contribution de chaque tête, élimination des têtes sans apport et routeur causal par fragment.
- `scripts/s791_six_role_audits.py` : auditeurs originaux, initialement bloqués par l'absence de NumPy dans les jobs indépendants.
- `scripts/s791_six_role_audits_replay.py` et `.github/workflows/note-s791-audit-replay.yml` : correction de l'audit et réexécution **sans refaire l'entraînement** sur les artefacts immuables.
- `series/S791/resume.md` et `series/S791/resultat.md` : traçabilité du résultat et des six audits.

La validation par composition a conservé les experts **general, bass, transition** et exclu **harmonic**. Il reste 8 masques admissibles sur 16 ; le sélecteur choisit une combinaison par fragment.

## Protocole

Sources GuitarSet FIT80 (BN1, Funk2, Jazz1, Rock3, SS1) et CAL32 (BN2, Jazz2). Le choix des experts et du routeur dépend uniquement de FIT ; CAL sert à vérifier les décisions figées. HELD/TEST non ouverts. Modèles entraînés sur des régions acoustiques distinctes ; aucune vérité terrain au moment de l'inférence.

Premier run de calcul [#39280680273](https://github.com/Andriamarosoa/note/actions/runs/39280680273), source `ff96ffec8aa34b5dd55d92142413a4a8690c745c` : **entraînement SUCCESS mais workflow final FAILURE**, car les jobs d'audit ne disposaient pas de NumPy. Les preuves scientifiques `S791_FRAMEPITCH_MULTIEXPERT_FIT80_CAL32` ont été figées. Replay des six audits sur ces artefacts, sans réentraînement : [#39325760225](https://github.com/Andriamarosoa/note/actions/runs/39325760225), vérificateur `6361a01a846f257c6c2039f243b1fe6bbb4bf536` : **SUCCESS 6/6 et consensus SUCCESS**.

Scores de tête SHADOW sur 1024 échantillons, pas sorties MIDI sur 256 : S791 CAL32 **82,2115 %**, contre **77,8367 %** pour sa source et **85,1177 %** pour le candidat expérimental S790. L'ajout d'experts n'a pas surpassé S790.

## Audits obligatoires

- Données : 80 FIT, 32 CAL et cinq plis hors composition, SHA S708 signé.
- Acoustique : régions graves, voisinages harmoniques, transitions temporelles et causalité des observations.
- Architecture : quatre spécialisations candidates, trois conservées sur FIT, huit masques autorisés et routeur par fragment.
- Apprentissage : chiffres par spécialiste et par composition, sélection strictement FIT, régression contrôlée.
- Métriques : recomptage TP/FP/FN, micro-F1 par composition, texture et registre, comparateur S790.
- LIVE : aucun changement du pipeline microphone ou du flux MIDI final, aucune preuve d'amélioration du MIDI officiel.

## Statut

**Terminé, six audits PASS et consensus PASS, candidat multi-têtes non promu.** Le modèle S791 est inférieur à S790 sur CAL32 SHADOW. La référence scientifique officielle demeure S708.
