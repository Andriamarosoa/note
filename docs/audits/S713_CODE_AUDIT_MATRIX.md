
## Correctifs S713 désormais vérifiés (code et CI)

Après la rédaction initiale de cette matrice, l'audit a produit les correctifs suivants sur **cette même branche** :
- **AUD-001 — Couverture logicielle corrigée** : `test/test_s713_live_cpu_and_events.py` impose de vrais `note_on` et `note_off` cohérents depuis la sélection S708 ; la CI exige aussi que le **vrai checkpoint** émette au moins une paire NoteOn/NoteOff sur un PCM test et que les événements de moteurs incrémental et référence soient égaux. **La qualité sur guitare enregistrée reste non validée**.
- **AUD-002 — Recalcul CPU réduit, ne remplace pas une mesure Windows** : caches temporels par couche et calcul des 192D seulement pour la ligne courante. Le run [S713 final](https://github.com/Andriamarosoa/note/actions/runs/39106989073) passe 19 tests, modèle S708 SHA256 `c81d98dbce8d096b54d71dedd88f521162ea95b19d0c059ab13baf69d5cf0588`, événements identiques sur WAV de contrôle : p95 **14,2701 → 5,0670 ms** et ratio travail/audio **0,7036 → 0,3806** sur GitHub Ubuntu, pas le matériel iRig. Le pas microphone de 256 échantillons vaut 5,805 ms à 44,1 kHz ; p99 et variations Windows restent inconnus.
- **AUD-004 — Corrigé** : `--status --s708-model` indique modèle réel, hash validé, politique CAL figée et pas de décision 1024 samples ; aucun affichage ambigu de la cadence native 256 samples.
- **AUD-005 — Contrats de score partiellement testés** : appariement par pitch, NoteOff de fin non crédité, double attaque interdite, réattaque et timestamp non antidaté ; protocole multi-sources toujours à revalider.

**Toujours ouverts :** AUD-003 (S707 Offset spécialisé non branché en LIVE), AUD-006 (harmoniques / accords E2–K6), AUD-007 (inventaire des 848 workflows sans suppression non prouvée), et validation sur vraie guitare/latence physique.

Voir [S713_LIVE_FIXES.md](S713_LIVE_FIXES.md) pour les commandes Windows et limites.

---

# S713 — Matrice des audits indépendants et preuves de code

**Point de départ (18 octobre 2026)** : branche `codex/note-s712-s708-strict-live-trial`, commit `a47dd8cb6bbf36a5649460b44c10f5b743259564`.
Cette matrice démarre un travail d'audit, pas une preuve de résolution.
Les agents ne sont pas automatiquement exécutés par ce dépôt ; les issues associées peuvent être ouvertes dans plusieurs sessions Work/Codex avec des branches individuelles.

## Objectifs mesurables et lignes de base (ne jamais additionner ces chiffres)
| Référence et protocole | Onset F1 | Exact-K | Offset conditionnel | F1 notes complètes |
|---|---:|---:|---:|---:|
| S708, GuitarSet DEV 12, chaîne événementielle expérimentale | 0.687520 | 0.589728 | 0.737345 (S707 hors ligne) | 0.503095 (S707 hors ligne) |
| S708, GuitarSet HELD 32, chaîne événementielle expérimentale | 0.729183 | 0.622400 | 0.686553 (S707 hors ligne) | 0.491470 (S707 hors ligne) |
| S711, GuitarSet DEV 12, autre candidat | 0.675421 | 0.589014 | 0.729648 | 0.488889 |
| S699, GuitarSet DEV, K seul shadow | — | 0.666551 | — | — |
| S675, ancien DEV, natif | 0.631046 | 0.544721 | 0.780808 | 0.488776 |

**Attention** : S708 « Exact-K » ici est la cardinalité *des notes actives effectivement décodées*, non l'Exact-K des attaques d'un événement S80/S397 ; et S675 0.780808 n'est pas mesuré dans la même configuration qu'Offset S708. Tout travail doit comparer deux modèles dans un même run/split/critère.

Provenance : [S708](https://github.com/Andriamarosoa/note/actions/runs/39094682419) ; [S711](https://github.com/Andriamarosoa/note/actions/runs/39099109305) ; [S699](https://github.com/Andriamarosoa/note/actions/runs/39070073982) ; [S675](https://github.com/Andriamarosoa/note/actions/runs/39059008484).

## Premiers constats de revue statique

### AUD-001 — P1 qualité de validation LIVE : le smoke-test S712 prouve l'exécution, pas les Notes [CONFIRMÉ]
- Code : `src/causal_note/s712_s708_live.py`, `test/test_s712_s708_live.py`, workflow `.github/workflows/note-s712-s708-live-trial.yml`.
- Preuve : [run S712 réussi](https://github.com/Andriamarosoa/note/actions/runs/39104080074), sorties `S712 VERIFIED WAV SMOKE 0 events`; 8 tests du code du décodeur/feature ont réussi, sans vraie émission musicale.
- Risque : faire passer l'intégration comme validation de transcription micro alors qu'aucun NoteOn/Off correct sur vraie source n'a été prouvé.
- Test demandé à agent A/E : fixture WAV guitare connue et annotation, qui provoque >=1 NoteOn et son NoteOff, appariements MIDI40–88, vérification identité des événements `push` et `finish`, comparaison parité `--wav` vs chunks PCM irréguliers, under/overrun.
- Verdict provisoire : **TEST COVERAGE FAIL** ; aucune prétention de régression de score ou corruption sans reproduction.

### AUD-002 — P1 risque de temps réel : budget CPU par callback [CONFIRMÉ pour CI, INCONNU Windows]
- Code : `src/causal_note/s705_temporal_heads.py` `TemporalHeadStream.step()` (rejoue le contexte complet), `src/causal_note/s712_s708_live.py` `_decide()`, `sharp_track()` (FFT répétées).
- Preuve : run S712, `worker_p95_ms = 11.6376`, sur callbacks 256 PCM à 44,1 kHz (=5,80499 ms d'audio) ; `worker_over_audio_ratio = 0.5636` global. Un pic individuel > durée d'un callback est possible même si le travail moyen reste sous le temps audio.
- Risque : remplissage de file `queue_blocks`, note tardive, glitch ou arrêt sur iRig Windows ; ces valeurs *ne sont pas* une latence MIDI matérielle.
- Test agent A/E : benchmark CPU p50/p95/p99 du **worker** et du **bloc décision 1024**, taux d'overrun pendant 5 minutes d'audio en flux, log des gaps et taille max file. Ne pas modifier les échéances sans preuves.

### AUD-003 — P1 disparité benchmark / modèle LIVE Offset [CONFIRMÉ]
- Code : `scripts/s708_count_guided_joint_loops.py` lignes 175–180 (`prepare_hazards`, `render_release`) vs `src/causal_note/s712_s708_live.py` (`GuidedSelector` sans hazard).
- Effet : F1 notes complètes et Offset du run S708 ne sont **pas** une mesure de S712 microphone ; la libération peut aussi influencer les attaques futures via la capacité active.
- Test agent A/D : reproduire les métriques sur le même WAV et mêmes règles, avec et sans S707, puis comparer à la chaîne streaming progressivement branchée. Aucun score 73,73%/50,31% revendiqué pour S712 avant parité de toutes les émissions.

### AUD-004 — P2 statut `--status` ambigu sur `--s708-model` [CONFIRMÉ par lecture]
- Code : `scripts/live_strict_notes.py`, `if args.status` signale le modèle natif `args.model`, pas le checkpoint S708 effectivement fourni.
- Effet : audit opérationnel ne distingue pas l'option `--s708-model`; risque de comparer logs de mauvaises variantes.
- Test agent E : `python -m scripts.live_strict_notes --status --s708-model <fichier>` doit déclarer explicitement l'adaptateur S708, la version, le SHA et la présence de `report.json`, sans prétendre à un benchmark micro.

### AUD-005 — P1 reproductibilité des scores et fuite expérimentale [À AUDITER]
- Code : `scripts/s704_joint_metrics.py` (matching, `full_note_f1 = 2*offset_correct/(2TP+FP+FN)`, clipping exact K), `scripts/s706_train_sharp_temporal_loops.py` (FIT/CAL/HELD), `scripts/s708_count_guided_joint_loops.py`.
- Question : l'étiquette « full-note F1 » représente-t-elle bien des NoteOn/Off appariés, sans crédit EOF ? Les splits excluent-ils compositions, interprètes et enregistrements dérivés et toutes les décisions de tuning sur DEV ?
- Tests agent D : note isolée, deux notes chevauchantes, pitch faux à bon temps, doublon, longue note sans NoteOff, réattaque au même pitch, EOF, zéro note, set d'hyperparamètres arrêté avant HELD. Publier numérateur/dénominateur et flux émis sérialisé. Distinguer **prototype** des résultats **prospectifs**.

### AUD-006 — P1 harmonique/fondamentale et biais accords [À AUDITER]
- Code : `src/causal_note/s706_sharp_spectra.py`, `src/causal_note/s704_joint_heads.py`, `src/causal_note/s708_cardinality_selection.py`.
- Preuve fonctionnelle : Onset F1 S708 DEV accord `0.652622`, solo `0.780430`. Un sous-comptage d'accords peut empêcher des attaques malgré un onset local élevé.
- Tests agent B/C : évaluer par MIDI E2–B2/C3–B3/C4+, polyphonie vraie K1..K6, prises accord vs notes isolées de mêmes hauteurs, réattaque de notes soutenues, analyse FP harmonique, K budget and réarmement. Conserver contrôle « sans tête harmoniques » et signaler toutes régressions appariées.

### AUD-007 — P2 dette CI : nombreux workflows historiques [CONFIRMÉ par inventaire, impact à mesurer]
- La branche S712 expose **848 fichiers sous `.github/workflows/`**. Cela **ne prouve pas** qu'ils se déclenchent en parallèle ; inspecter leurs déclencheurs/concurrency avant de supprimer ou regrouper.
- Tests agent E : inventaire `on`, `paths`, `workflow_dispatch`, `concurrency`, runs réellement doublonnés, protection des workflows de référence. Ne pas effacer l'historique.

## Règles de progression / promotion
- Une issue par agent, PR **DRAFT** sans changement de référence. Audit de code peut être validé par reproduction sans entraînement lourd.
- Une correction F1 doit venir avec SHA du checkpoint, partitions FIT/CAL/HELD/DEV/TEST, émissions noteOn/Off JSONL et six métriques associées.
- Un seul de `HELD` ou `DEV` meilleur n'est jamais une promotion. Régressions sur Offset, Exact-K, notes longues, réattaques ou LIVE ⇒ rejeter ou garder shadow.
- Relecture F obligatoire pour toute PR A/B/C/D/E. Un résultat de CI « success » signifie seulement que les tests programmés ont réussi.
- Objectif demandé : six métriques >=80% *sur une même source/protocole*, sans changer rétroactivement l'étendue de la tâche.
