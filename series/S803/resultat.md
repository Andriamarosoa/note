---
serie: S803
statut: termine
type: experience
reference: S708
branche: codex/note-s803-causal-current-frame-gates
run_id: 39363062723
commit_sha: 68b1685a0e5a386e7273ddfe7f963125a666cdc5
promotion: false
---

# S803 — trois expériences GuitarSet réelles rejetées ; sept audits validés

## Exécution

[GitHub Actions run #39363062723](https://github.com/Andriamarosoa/note/actions/runs/39363062723) : **SUCCESS**, commit testé `68b1685a0e5a386e7273ddfe7f963125a666cdc5`. Évaluation des **80/80 vraies pistes GuitarSet FIT80 OOF** à partir des archives signées S800 #39352880035, S801 #39354733640, S802 #39359182007. Données source : 306 536 frames, 189 124 frames Exact-K correctes ; les douze compteurs bruts S795 ont été reproduits sans tolérance. Les mesures MIDI256 et ROC sont archivées dans `S803_REAL_GUITARSET_FIT80_MIDI256_METRICS_AUC` du run. Le code de mesure/replay est resté inchangé depuis la première expérience; les changements ultérieurs ont corrigé les audits et le client NVIDIA.

Historique de débogage contrôlé : run [#39361223391](https://github.com/Andriamarosoa/note/actions/runs/39361223391) replay PASS mais six audits bloqués par import PyTorch absent ; run [#39361632654](https://github.com/Andriamarosoa/note/actions/runs/39361632654) replay et six audits PASS, avis NVIDIA expiré après 115 secondes malgré une relance ciblée ; run [#39362716888](https://github.com/Andriamarosoa/note/actions/runs/39362716888) bloqué par syntaxe du client NVIDIA, depuis corrigée et testée dans le smoke. **Ne pas agréger ces runs pour fabriquer un 7/7** : le run final #39363062723 porte lui-même les sept preuves et le consensus PASS.

## Métriques

**Pourcentages mesurés sur les événements réellement émis**. S708 CAL32 officiel est un split distinct : il ne doit pas être comparé numériquement aux variations FIT80 comme s'il s'agissait du même corpus.

| Métrique | S708 officiel CAL32 | Source S795/S800 FIT80 | rise_attack FIT80 | parent_conflict FIT80 | agreement_attack FIT80 |
|---|---:|---:|---:|---:|---:|
| Onset F1 | 74,2639 % | 73,2520 % | 72,9168 % | 72,8779 % | 73,2009 % |
| Exact-K | 62,0455 % | 61,6972 % | 61,7034 % | 61,7125 % | 61,6883 % |
| Active Exact-K | 56,2762 % | 57,0195 % | 57,0304 % | 57,0416 % | 57,0087 % |
| Offset accuracy | 77,3292 % | 69,8862 % | 69,4622 % | 69,3865 % | 69,8424 % |
| Full Note F1 | 57,1319 % | 50,6528 % | 50,1154 % | 50,0342 % | 50,5858 % |
| Frame Pitch F1 | 76,4626 % | 78,7278 % | 78,8056 % | 78,8147 % | 78,7418 % |

**Aucune des trois règles ne passe les six critères de non-régression.** Détails :
- `rise_attack` : **0** cellule supplémentaire bloquée vs `adaptive` S801, 118 NoteOn synthétiques ; **3/6** ratios en baisse au global et **19** couples composition×métrique négatifs. Son garde ajouté est inerte sur ces données ; la régression vient de sa base adaptive S801.
- `parent_conflict` : **0** cellule supplémentaire bloquée vs `consensus` S801, 136 NoteOn synthétiques ; **3/6** ratios en baisse et **17** couples composition×métrique négatifs. Garde inerte ; régressions héritées de consensus.
- `agreement_attack` : **99** cellules de têtes bloquées, **19** nouveaux NoteOn restants ; **5/6** ratios en baisse et **14** couples composition×métrique négatifs. Il agit effectivement mais supprime trop de vraies détections pour être admissible.

**ROC/AUC rétrospectif des vrais événements S802**, étiquettes réservées à l'audit et jamais fournies au décodeur :
- Consensus : meilleure variable isolée `on`, AUC **0,645488** ; bootstrap 95 % par piste **[0,544087 ; 0,748734]**.
- Adaptive : meilleure `on`, AUC **0,649985** ; bootstrap 95 % **[0,532914 ; 0,751871]**.
- Poly-gap : meilleure `off`, AUC **0,610919** ; bootstrap 95 % **[0,494773 ; 0,708808]**.

Le recouvrement TP/FP et l'incertitude des AUC n'autorisent aucune affirmation de seuil universel performant. Il existe seulement de faibles gains marginaux Exact-K ou Frame Pitch qui ne compensent pas les pertes de détection complète.

## Comparaison à S708

**Référence S708 maintenue sans modification, promotion=false.** S803 est un travail de recherche FIT80 sans nouvelle mesure CAL32/HELD/TEST. Les cinq têtes indépendantes natives et le routage dynamique par frame/hauteur sont conservés ; il n'y a pas encore une architecture entièrement séparée pour chacune des six métriques. Aucune modification du pipeline microphone LIVE, aucun lookahead, buffer, retiming ni prolongation de NoteOff. Le replay vérifie l'absence de retard algorithmique ajouté, mais **ne mesure pas la latence physique micro/CPU** : le test S793 Windows/iRig reste nécessaire avant toute future promotion.

## Audits

Six rapports issus **du même run #39363062723**, tous indépendants et PASS :

| Rôle | Statut | Preuve |
|---|---|---|
| Données | PASS | [Audit S803 #39363062723](https://github.com/Andriamarosoa/note/actions/runs/39363062723), `S803_AUDIT_data` |
| Acoustique | PASS | [Audit S803 #39363062723](https://github.com/Andriamarosoa/note/actions/runs/39363062723), `S803_AUDIT_acoustics` |
| Architecture | PASS | [Audit S803 #39363062723](https://github.com/Andriamarosoa/note/actions/runs/39363062723), `S803_AUDIT_architecture` |
| Apprentissage | PASS | [Audit S803 #39363062723](https://github.com/Andriamarosoa/note/actions/runs/39363062723), `S803_AUDIT_training` |
| Métriques | PASS | [Audit S803 #39363062723](https://github.com/Andriamarosoa/note/actions/runs/39363062723), `S803_AUDIT_metrics` |
| LIVE | PASS | [Audit S803 #39363062723](https://github.com/Andriamarosoa/note/actions/runs/39363062723), `S803_AUDIT_live` |

**Septième conseiller NVIDIA : PASS**, réel modèle `z-ai/glm-5.3-flash`, artefact `S803_SEVENTH_NVIDIA_INDEPENDENT_REVIEW` dans ce même run. **Consensus final 7/7 : PASS**, artefact `S803_SEVEN_OF_SEVEN_CONSENSUS` et message vérifié `NOTE_SEVEN_OF_SEVEN_PASS S803 z-ai/glm-5.3-flash promotion=false`. La relecture NVIDIA est un **avis non contraignant** et les six rôles CI sont des auditeurs déterministes, distincts des six agents personnalisés Work. Ils n'établissent ni gain musical ni latence physique.

L'avis externe observe notamment que `rise_attack` et `parent_conflict` sont inertes ; recommande de cesser d'ajuster de simples seuils mono-feature sur les mêmes observations FIT et d'auditer l'exactitude des NoteOff associés aux vraies collisions S802.

## Verdict

**S803 terminée, sept contrôles PASS, trois alternatives rejetées, S708 officiellement conservée.** La non-régression est respectée par la *décision de rejet* : aucune des variantes qui dégrade une caractéristique n'est intégrée à la référence ou au LIVE. Aucun score six-métriques >80 % revendiqué. Le coût CPU/latence microphone n'a pas été mesuré et reste une condition d'une éventuelle promotion ultérieure.

## Prochaine action

Priorités pour une future série séparée : (1) mesurer **les vrais NoteOff des 64 attaques TP** parmi les 95 collisions `s795_source_retrigger` consensus S802, par composition et hauteur ; (2) identifier le recouvrement exact des 62/46/51 faux positifs entre trois politiques ; (3) auditer des caractéristiques conjointes onset+offset strictement disponibles au fragment courant, avec incertitude par piste et opposition par composition ; (4) tester la robustesse de l'appariement 50/80/120 ms ; (5) prouver S793 zéro latence ajoutée physiquement avant toute promotion, toujours sans utiliser une attaque future ni retarder une fin de note. Aucune optimisation multi-têtes ne doit être proclamée indépendante des six métriques tant que ses architectures et garanties ne sont pas effectivement implémentées.
