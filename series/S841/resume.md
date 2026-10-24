---
serie: S841
statut: en_cours
type: experience
reference: S708
branche: codex/note-s841-real-e2-e3-octave256
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S841 — Données E2, E3, E2+E3 pour valider la transposition neuronale

## Objectif
Partir des **80 vraies pistes GuitarSet signées** pour vérifier la présence et le support des notes MIDI E2=40, E3=52 et des attaques concomitantes. Ensuite seulement construire de vrais couples audio source/cible séparément enregistrés, nécessaires pour apprendre la transposition latente S840 sans changement de hauteur par formule. La comparaison finale devra être IA avec vues latentes vs encodeur témoin sans vues latentes, sur les mêmes notes et mêmes enregistrements.

## Hypothèse
Un modèle transportant les caractéristiques dans un espace latent appris pourrait révéler des signatures difficiles dans l'accompagnement polyphonique. Mais l'absence de fondamentales ou d'événements clairement étiquetés ne peut pas être corrigée simplement en déclarant que deux attaques proches constituent un accord. E2 a une période physique ~12,2ms, supérieure aux 5,805ms du bloc autorisé ; cela constitue un vrai défi informationnel pour les entrées PCM256 sans aucun contexte passé.

## Modifications
`scripts/s841_audit_real_guitarset_e2e3_256.py` lit exclusivement l'archive WAV GitHub originale signée et les 80 traces de vérité terrain S800 ; agrège E2, E3 et le nombre de paires d'attaques proches sur les mêmes pistes par BN1/Funk2/Jazz1/Rock3/SS1 et solo/accompagnement. `tests/test_s841_real_guitarset_e2e3_audit256.py` vérifie l'échelle et le caractère insuffisant de simples attaques proches pour prouver le sustain simultané.

## Protocole
N'utiliser que les fichiers WAV mono 44100Hz int16 signés avec SHA, 306536 blocs au total. Aucun transpositeur DSP, aucun audio synthétisé, pas de GT introduit à l'inférence, aucun checkpoint S708 historique employé dans l'outil de transposition. Refuser de construire une paire audio de training E2/E3 sans vérification que les sons sont physiquement distincts et que les notes correspondent à des GT de prises isolées ; pour E2+E3, vérifier un vrai chevauchement de durée, et ne pas assimiler deux NoteOn séparés de 80ms à un accord. Conserver les cinq autres têtes et le MIDI inchangés.

À cette étape, seule la disponibilité de GT est auditée ; S840 conserve un entraînement physique réel encore à venir. Un F1 de classification de notes par trame n'est pas Onset F1 ; les TP/FP/FN Onset ±80ms sur les 80 pistes ne seront calculés qu'après production réelle de modèles appariés, sans contamination TRAIN/CAL/OUTER.

## Audits obligatoires
| Rôle | Critère |
|---|---|
| Données | WAV signés, 80 traces, compositions et textures identifiables |
| Acoustique | signatures des 256 nouvelles valeurs uniquement et zéro audio futur |
| Architecture | S840 isolé sans DSP, aucun retour sur les six autres caractéristiques |
| Apprentissage | nécessite paires physiquement enregistrées et disjointes ; aucune invention de paire |
| Métriques | compte E2/E3, simultanéité prouvée vs proximité, F1 final non mesuré |
| LIVE | 5,805ms par hop, p99 Windows/iRig non encore démontré |

## Statut
Recherche en cours, non promue ; la référence acoustique Onset PCM256 est toujours **69,2584331 % F1**, et non le score de S834/H0. Aucun gain E2/E3 prouvé avant les rapports GitHub et l'entraînement S840 réel.
