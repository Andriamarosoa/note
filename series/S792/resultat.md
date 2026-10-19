---
serie: S792
statut: termine
type: experience
reference: S708
branche: codex/note-s792-framepitch-six-metric-midi-replay
run_id: 39328341071
commit_sha: aec47655049d89ddf297348ca1ec89b08c8d1daf
promotion: false
---

# S792 — Résultats

## Exécution

- Expérience réelle : [GitHub Actions #39328341071](https://github.com/Andriamarosoa/note/actions/runs/39328341071) **SUCCESS**. SHA `aec47655049d89ddf297348ca1ec89b08c8d1daf`.
- Artefact scientifique : `S792_REAL_CAUSAL_MIDI256_SIX_METRICS`, `experiment/report.json`.
- S708 signé : SHA256 `c81d98dbce8d096b54d71dedd88f521162ea95b19d0c059ab13baf69d5cf0588` ; **parité exacte validée pour les six métriques et les compteurs d'événements** sur CAL32.
- GuitarSet FIT80 : 80 enregistrements, cinq compositions, 5 plis OOF. CAL32 : 32 enregistrements, BN2/Jazz2. HELD et TEST fermés.
- Candidat : prolongations causales de NoteOff réels sous soutien de la tête S790 gelée ; durées 0/1/2/4 décisions de 1024 échantillons.
- Horloge de mesure **officielle : 256 échantillons**, `scripts.s704_joint_metrics.metric`, six scores sur le même MIDI sérialisé.
- Contrôle documentaire S792 : workflow dédié, sans lancement de nouveaux tests scientifiques sur simple changement de Markdown.

## Métriques

Le choix FIT étant **0 bloc**, le candidat CAL32 est un replay à l'identique de S708.

| Métrique officielle | S708 CAL32 | S792 sélectionné | Différence |
| --- | ---: | ---: | ---: |
| Onset F1 | 74,2639 % | 74,2639 % | 0 |
| Exact-K | 62,0455 % | 62,0455 % | 0 |
| Active Exact-K | 56,2762 % | 56,2762 % | 0 |
| Offset accuracy | 77,3292 % | 77,3292 % | 0 |
| Full Note F1 | 57,1319 % | 57,1319 % | 0 |
| Frame Pitch F1 | 76,4626 % | 76,4626 % | 0 |

**Aucune amélioration validée sur le MIDI final.** Le score SHADOW S790 de 85,1177 % **n'est pas** la performance du flux MIDI émis.

## Quatre variantes — FIT80 validation hors composition, événements MIDI 256

| Durée maximale NoteOff | Onset F1 | Exact-K | Active Exact-K | Offset | Full Note F1 | Frame Pitch F1 | Éligible |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 bloc (S708) | 73,2520 % | 61,6972 % | 57,0195 % | 69,8862 % | 50,6528 % | 78,7278 % | **Oui** |
| 1 bloc | 73,2520 % | 62,4716 % | 58,0399 % | 67,7530 % | 49,1067 % | 79,4657 % | Non |
| 2 blocs | 73,2520 % | 62,9182 % | 58,6073 % | 66,1413 % | 47,9385 % | 79,8467 % | Non |
| 4 blocs | 73,2520 % | 63,4917 % | 59,3286 % | 66,1887 % | 47,9729 % | **80,4416 %** | Non |

Le gain Frame Pitch FIT du meilleur cas 4 blocs est **+1,7139 point**, mais il entraîne **−3,6976 points d'Offset** et **−2,6799 points de Full Note**, comparé à la référence FIT. La garde sans régression rejette donc cette solution, même si Frame Pitch franchit le seuil de 80 % dans FIT.

### Analyse du comportement événementiel

| Durée | NoteOff retenus au départ | Modifications temporelles effectives | Retriggers forçant la fermeture | Fermetures par perte du soutien | Fermetures par délai |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | 5 942 | 2 623 | 3 319 | 460 | 2 163 |
| 2 | 5 942 | 2 623 | 4 282 | 626 | 1 026 |
| 4 | 5 942 | 2 623 | 4 438 | 743 | 745 |

Les variantes longues atteignent aussi neuf fermetures anticipées pour contrainte de capacité au maximum sur FIT. L'amélioration de la persistance de la note n'implique donc pas une meilleure précision de fin de note, et peut modifier Exact-K/Active Exact-K sans réduire les erreurs de NoteOff.

### Analyse CAL32 par texture pour le candidat admissible (identique S708)

| Texture | Onset F1 | Exact-K | Active Exact-K | Offset | Full Note F1 | Frame Pitch F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Accompagnement | 69,3999 % | 44,4341 % | 32,0135 % | 74,3119 % | 51,3231 % | 73,4402 % |
| Solo | 84,7923 % | 79,6570 % | 79,6378 % | 82,6794 % | 69,7055 % | 83,4048 % |

Les accompagnements polyphoniques restent nettement plus difficiles que les solos. Ces valeurs ne doivent pas être interprétées comme de nouveaux gains.

## Comparaison à S708

- **S790 SHADOW CAL32** : 85,1177 % F1 sur des masques à 1024 échantillons. Ce résultat est réel dans son protocole, mais ne fabrique pas des notes MIDI manquantes.
- **S792 MIDI 256 CAL32** : 76,4626 % de Frame Pitch, parité exacte du flux S708, aucune amélioration après protection des cinq autres métriques.
- Aucune promotion, aucune altération du modèle S708/S707 et aucune prétention de performance microphone.

## Audits

Source : [Run GitHub #39328341071](https://github.com/Andriamarosoa/note/actions/runs/39328341071). Les six audits ont rendu des artifacts distincts ; le consensus obligatoire est **SUCCESS**.

| Audit | Statut | Preuve |
| --- | --- | --- |
| Données | PASS | `S792_AUDIT_data` |
| Acoustique | PASS | `S792_AUDIT_acoustics` |
| Architecture | PASS | `S792_AUDIT_architecture` |
| Apprentissage | PASS | `S792_AUDIT_training` |
| Métriques | PASS | `S792_AUDIT_metrics` |
| LIVE | PASS | `S792_AUDIT_live` |
| Consensus | PASS | `S792_SIX_REAL_MIDI_AUDIT_CONSENSUS` |

Le contrôle LIVE signifie que les événements reproduits ne sont pas rétrodâtés et que les sources officielles sont intactes ; **la latence microphone réelle n'a pas été mesurée**.

## Verdict

**S792 terminé, six audits PASS, non promu.** Validation négative mais concluante : prolonger naïvement les fins de notes aide Frame Pitch et Exact-K sur FIT, mais détériore les vraies fins de notes et Full Note F1. Sélection sur FIT : durée zéro ; six scores CAL strictement inchangés.

## Prochaine action

Auditer séparément les faux négatifs de pitch dus à l'absence de NoteOn, les NoteOff trop précoces et les notes qui se chevauchent. Introduire des experts de **création de NoteOn** ou de confirmation de sustain seulement avec preuve acoustique suffisante, et vérifier un flux MIDI causal intégral plutôt qu'un masque SHADOW. Toute décision doit être choisie sur FIT, puis évaluée sur CAL sans perte des cinq métriques protégées.
