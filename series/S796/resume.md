---
serie: S796
statut: en_cours
type: experience
reference: S708
branche: codex/note-s796-retrigger-forensics-no-extra-latency
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S796 — Correction des re-attaques synthétiques simultanées

S795 a terminé FIT80 et CAL32 sans valider la signature CAL32 ; l'exécution 39340511467 s'est arrêtée sur k_correct=78367 au lieu de 78366. Le résultat FIT80 est exploitable comme diagnostic, **pas comme validation scientifique CAL** : 25 politiques, cinq experts entraînés, une seule politique admissible (l'identité, donc zéro tête retenue).

Hypothèse ciblée : une note synthétique pouvait être fermée et rouverte à une même décision si sustain était entre 0,16 et 0,30. Bloquer ce re-déclenchement au même timestamp sans temporisation ni report des NoteOff. Le test de régression utilise deux décisions successives, avec NoteOn à 1024, NoteOff à 2048 et interdit un nouveau NoteOn à 2048. Aucun événement source S708 ne change.

Audits : données, acoustique, architecture, entraînement, métriques, LIVE séparés, consensus strict. L'efficacité musicale du nouveau garde, six métriques MIDI256, CPU et latence microphone : **non mesurés**. Aucun nouveau traitement sur le poste LIVE, aucune promotion, HELD/TEST fermés.

Ne pas relancer les 80+32 fichiers tant que les métriques FIT S795 et la cause de dérive signée n'ont pas été auditées. Prochain candidat : sélection causale par fragment et meilleure précision NoteOn/NoteOff, uniquement si six non-régressions et absence de latence sont démontrées.
