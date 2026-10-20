---
serie: S803
statut: en_cours
type: experience
reference: S708
branche: codex/note-s803-causal-current-frame-gates
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S803 — résultats, mesure en attente

## Exécution

Run GitHub S803 : non démarré au moment de la création des rapports ; consulter le workflow `note-s803-causal-frame-gates.yml`. Les preuves réelles devront comprendre les artefacts signés S800/S801/S802, les 80 replays MIDI256, six audits, la réponse NVIDIA et le verdict 7/7 du même run. Statut avant exécution : non mesuré.

## Métriques

| Métrique | S708 officiel GuitarSet CAL32 | S803 FIT80 candidat |
| --- | ---: | --- |
| Onset F1 | 74,2639 % | non mesuré |
| Exact-K | 62,0455 % | non mesuré |
| Active Exact-K | 56,2762 % | non mesuré |
| Offset accuracy | 77,3292 % | non mesuré |
| Full Note F1 | 57,1319 % | non mesuré |
| Frame Pitch F1 | 76,4626 % | non mesuré |

Les trois variantes `rise_attack`, `parent_conflict` et `agreement_attack` sont des propositions de recherche **sans gains présumés**. L'AUC sur étiquettes S802 est diagnostic seulement et ne se substitue pas aux six métriques d'événements. Aucune amélioration n'est validée tant que l'artefact de replay réel et ses comptes n'ont pas été examinés.

## Comparaison à S708

Référence S708 inchangée. La comparaison est appariée aux sorties FIT gelées ; la colonne officielle est CAL32 et n'est pas directement interchangeable avec le protocole FIT. Les scores FIT ne justifient ni remplacement du modèle officiel ni revendication de latence micro. Aucun ajout de buffering, fenêtre après attaque, prolongation de NoteOff ou utilisation du futur n'est autorisé.

## Audits

| Rôle | Statut | Preuve |
| --- | --- | --- |
| Données | UNKNOWN | run S803 en attente |
| Acoustique | UNKNOWN | run S803 en attente |
| Architecture | UNKNOWN | run S803 en attente |
| Apprentissage | UNKNOWN | run S803 en attente |
| Métriques | UNKNOWN | run S803 en attente |
| LIVE | UNKNOWN | run S803 en attente |

NVIDIA : UNKNOWN ; consensus 7/7 : UNKNOWN. Six audits déterministes et conseiller scientifique externe ne représentent pas l'exécution des six agents personnalisés Work. Un consensus vert n'implique ni nouvelle valeur de F1 ni test de latence physique.

## Verdict

**Non promu / verdict expérimental en attente.** Rejeter toute politique présentant un delta négatif même minime sur l'une des six métriques ou sur l'une des cinq compositions. Ne jamais transformer un succès de test logiciel en annonce de score 80 %.

## Prochaine action

Après exécution, extraire `s803_fit80.json`, publier scores et deltas exacts ainsi que la justesse des partitions, actualiser `resume.md` et `resultat.md` avec SHA et run ID puis décider d'une éventuelle S804. Une sélection sur CAL, un retiming, une fuite des vérités ou une latence CPU/micro non mesurée interdit promotion.
