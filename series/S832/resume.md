---
serie: S832
statut: en_cours
type: experience
reference: S708
branche: codex/note-s832-guarded-local256
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S832 — Veto local causal des corrections polyphoniques (256 échantillons)

## Objectif
Conserver sans partage inter-axes la capacité des experts internes à Exact-K à corriger K2, K3, K5, mais réduire leurs nombreuses régressions K0/K1/K4. Restaurer au minimum la stabilité du témoin S829 57,5247 % sur GuitarSet FIT80, mesurer séparément les gains, et ne promouvoir que sur preuve externe.

## Hypothèse
S831 `blend` corrige 39 215 erreurs et dégrade 51 094 cas, mais progresse sur K2/K3/K5. Des règles de risque spécifiques aux décisions du modèle et à sa confiance peuvent permettre de n'accepter que les changements dont l'avantage est suffisamment soutenu par les exemples CAL disjoints ; des veto par piste CAL limitent le risque de généralisation.

## Modifications
- `src/causal_note/s832_guarded_private_selector256.py` : candidat K à chaque trame à partir des propres trois têtes Onset/Sustain/Release S831 et des deux têtes K S829. Vingt propositions (= quatre modes × cinq seuils) ; seuils de confiance positive. Construction de cellules conditionnelles (mode, seuil, K avant, K proposé, bande de confiance), avec corrections/régressions sur la seule calibration. Minimum de preuves, majorités par piste et borne inférieure binomiale (Wilson) contre faux succès. Choix local via `choose` sans aucun accès GT à l'inférence.
- Validation CAL : 12 pistes de calibration pour apprendre des règles, 4 pistes CAL distinctes pour évaluer ; si aucune variante stable ne s'améliore, les règles restent vides et la sortie est strictement S829. Refit après choix seulement sur les 16 pistes CAL (aucun label OUTER).
- `scripts/s832_fit80_guarded_local256.py` : même PCM causal, cinq folds FIT80, audit par K/track/mode ; aucune modification des cinq autres caractéristiques.
- `test/test_s832_guarded_local256.py` : construction, seuils, fallback, anti-fuite GT, cardinalité sur 20 propositions.

## Protocole
80 GuitarSet S800 FIT signés, 306 536 fenêtres MIDI256, TRAIN 3 compositions, CAL 1, OUTER 1 ; 5 folds. Reproduire H0 et S829 en parallèle. Choix des règles sur CAL, évaluation des décisions OUTER **seulement ensuite**. Ne jamais tuner a posteriori sur les erreurs OUTER du run.

## Audits obligatoires
- Données : source S800, GT partitions FIT seulement, SHA original et 80 pistes.
- Acoustique : PCM44,1kHz, fenêtre causale 256, aucune attente future ajoutée.
- Architecture : aucune dépendance aux états/postérieurs des cinq autres caractéristiques.
- Apprentissage : vrais cinq modèles privés par fold, veto conditionnel appris exclusivement sur inner-CAL avec sous-validation par piste.
- Métriques : tous K0–K6 y compris K0, scores S832/S829/H0 même dénominateur ; corrections/régressions du veto.
- LIVE : grilles strictes 256, coût p95/p99 iRig non encore audité ; aucun claim de latency live.
- Avis NVIDIA/Work : aucun agent externe réellement consulté.

## Statut
En cours — aucune amélioration S832 revendiquée et `promotion:false`. La comparaison S831 56,9943 % vs S829 57,5247 % reste la seule mesure scientifique préalable.
