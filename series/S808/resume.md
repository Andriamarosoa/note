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
# S808 — deux têtes indépendantes, naissance et continuité

## Objectif

Tester sur les vraies pistes GuitarSet FIT80 l'hypothèse S807 selon laquelle Active Exact-K nécessite de dissocier les nouvelles attaques et la survie des voix déjà reconnues. Exiger aucune baisse sur les six métriques et aucune latence algorithmique supplémentaire. S708 demeure la référence officielle.

## Hypothèse

Une tête supervisée par les vraies attaques et une tête indépendante supervisée par l'occupation courante des hauteurs peuvent récupérer des voix manquantes sans influencer les événements du décodeur natif. Cette hypothèse n'est pas réputée vraie sans les six métriques d'une même transcription MIDI.

## Modifications

Expérience SHADOW locale reproductible, avec deux vrais modèles LogisticRegression différents (entrées respectives de 20 et 19 dimensions, supervision distincte), chacun entraîné sur les quatre autres familles de compositions et testé sur la cinquième. Trois politiques, définies avant le calcul : `birth_only`, `birth_plus_continuity`, `sustain_recovery_plus_continuity`. Les propositions NoteOn sont émises au **temps courant**, sans fenêtre future. La continuité peut anticiper la fermeture d'une voix additionnelle, mais **jamais prolonger la première fermeture S795**. Les notes S708/S707 natives sont protégées, ni supprimées ni décalées.

Le script intégral, ses quatre tests adversariaux, les 80 diagnostics et les résultats numériques sont livrés ensemble dans l'archive conversationnelle `S808_test_reel_FIT80.zip`; script SHA256 `31c4c10f84373f8219764832de63076a266503eba33981cf25900c20b7a8ce28`; ZIP SHA256 `5be93836634c628aaf595c3adec7afb2acab0301dc44fccbf7a1da207b070feb`. Ce code n'est **pas encore exécuté dans GitHub Actions**, afin de ne pas multiplier les runs échoués.

## Protocole

80 traces JSON/NPZ FIT80 réelles de l'artefact S800 [GitHub #39352880035](https://github.com/Andriamarosoa/note/actions/runs/39352880035). Le script reproduit exactement les **12 compteurs historiques par piste**, puis les six métriques agrégées S704 : Onset F1, Exact-K, Active Exact-K, Offset accuracy, Full Note F1 et Frame Pitch F1. Les compositions restent disjointes à l'entraînement, CAL32/HELD/TEST sont fermées, il n'y a aucun oracle à l'inférence ni recherche postérieure de seuil sur FIT. Quatre tests négatifs couvrent les changements après une trame future, la protection des événements natifs, la disjonction des folds et la non-prolongation des fins de notes supplémentaires.

## Audits obligatoires

Données : provenance S800 originale signée 80/80, compteurs MIDI natifs reproduits piste par piste. Acoustique : uniquement champs disponibles au moment de décision. Architecture : deux modèles indépendants et source S708 intacte. Apprentissage : composition évaluée exclue de ses deux régressions. Métriques : six axes et veto par composition. LIVE : 0 hold/anticipation dans le prototype, mais coût CPU et matériel iRig non mesurés. Ce sont des vérifications **locales**, et **non** six agents Work ni un consensus NVIDIA 9/9. Aucun candidat ne sera promu sans validation indépendante de la chaîne réelle.

## Statut

Le replay Python local FIT80 est terminé et reproductible depuis le ZIP joint. Les trois politiques ont échoué au veto de non-régression, S708 inchangée. Le statut reste `en_cours` dans la branche tant qu'une exécution GitHub Actions scientifique et ses audits indépendants n'ont pas été réalisés ; ne pas qualifier cette série de 6/6 CI PASS.
