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

# S803 — analyse réelle FIT80 et sélecteurs causaux indépendants par fragment

## Objectif

Contribuer à l'objectif NOTE **plus de 80 % sur six métriques** sans aucune régression, avec S708 comme seule référence officielle. Explorer des garde-fous acoustiques causaux sur les têtes gelées et leur sélection par frame/pitch, et déterminer si les fausses attaques S802 sont séparables sans vérité future, buffering, retiming ni retard NoteOff.

## Hypothèse

Les collisions native/synthétique de S802 mélangent de vrais et faux NoteOn. Des indicateurs instantanés (`on`, `on_rise`, `harmonic_parent`, probabilités des têtes) pourraient séparer TP/FP et limiter les faux positifs. Une séparation faible devait conduire au rejet des politiques plutôt qu'à une promotion sur de petites améliorations Exact-K.

## Modifications

À partir du HEAD S802 `67a49ed928f21b1a68e185e815bca9787573d8b4`, création d'une [branche S803](https://github.com/Andriamarosoa/note/tree/codex/note-s803-causal-current-frame-gates) et de la [PR #107 DRAFT](https://github.com/Andriamarosoa/note/pull/107). Nouveaux modules `scripts/s803_causal_frame_replay.py`, `scripts/s803_six_role_audits.py`, `test/test_s803_causal_frame_replay.py` et workflow `.github/workflows/note-s803-causal-frame-gates.yml`. Trois règles prédéclarées : `rise_attack`, `parent_conflict`, `agreement_attack`. Toutes s'appuient sur les cinq têtes existantes (general, bass, polyphonic, transient, harmonic) choisies dynamiquement par fragment/hauteur. **Les six métriques n'ont pas encore six architectures indépendantes complètes** : S803 conserve la spécialisation existante, sans déployer de nouvel ensemble LIVE.

Deux erreurs de CI ont été corrigées : import PyTorch inutile dans les six contrôles légers ; puis erreur de syntaxe d'une journalisation NVIDIA introduite pendant la correction d'un timeout. Un contrôle de compilation et les tests unitaires du client NVIDIA s'exécutent désormais dans le job initial. Le conseiller a une sortie bornée et un délai réseau adapté. Ces corrections n'altèrent pas les prédictions MIDI ni le modèle officiel.

## Protocole

Reproduction des vrais artefacts GuitarSet S800 FIT80 #39352880035, S801 #39354733640 et S802 #39359182007. Vérification SHA256 des **80/80 traces OOF** et stricte parité des 12 compteurs historiques S795 (dont `frames=306536`, `k_correct=189124`). Recalcul des six métriques à partir du flux de NoteOn/NoteOff MIDI256 réellement émis, sur cinq compositions séparées, veto de chaque régression par score ou composition. Analyses d'AUC sur les étiquettes S802 **a posteriori seulement**, avec bootstrap des pistes ; aucun seuil appris, aucune ouverture CAL32/HELD/TEST, aucune vérité ou attaque future utilisée pour décider.

## Audits obligatoires

Le [run #39363062723](https://github.com/Andriamarosoa/note/actions/runs/39363062723) a réellement exécuté en parallèle les rôles indépendants Données, Acoustique, Architecture, Apprentissage, Métriques et LIVE, avec **six PASS** et six artefacts de preuves. L'avis externe NVIDIA NIM du modèle `z-ai/glm-5.3-flash` a été reçu et archivé **dans le même run** après test du vrai secret GitHub. Le consensus final `NOTE_SEVEN_OF_SEVEN_PASS S803 z-ai/glm-5.3-flash promotion=false` est PASS ; cette validation certifie l'exécution contrôlée du protocole, **pas** un score musical amélioré ni une latence physique.

## Statut

**S803 terminée, audit 7/7 PASS, trois variantes scientifiquement rejetées, promotion=false.** Le replay signé confirme de petites améliorations Exact-K/Frame Pitch pour certaines variantes, mais des pertes Onset, Offset ou Full Note interdisent tout remplacement de S708. Aucun objectif musical 80 % atteint sur les six métriques et aucun test de latence physique microphone réalisé. Résultats, incertitudes et prochaines pistes détaillés dans `resultat.md`.
