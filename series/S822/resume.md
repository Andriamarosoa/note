---
serie: S822
statut: en_cours
type: experience
reference: S708
branche: codex/note-s822-s810-style-midi256-ablation
run_id: en_attente
commit_sha: en_attente
promotion: false
---
# S822 — Rejouer l'architecture acoustique et supervisée S810 à 256 échantillons

## Objectif

Tester si le gain S809→S810 SHADOW 1024 (+8,44 points Active Exact-K) vient d'une **tête indépendante entraînée** et de l'**agrégation 61 variables**, et non uniquement de la cadence 1024. Garder **feature/decision/metric = 256 échantillons**, sur les 80 WAV GuitarSet FIT80 original, 249411 trames actives, sans hold/lookahead.

## Hypothèse

En maintenant la même cadence et les mêmes splits d'une part et le même algorithme LogisticRegression de l'autre, la comparaison contrôlée entre les 23 coordonnées acoustiques brutes et leur résumé S810 en 61 variables mesure l'apport de l'agrégation. Le résultat peut être meilleur ou pire. La classe K0 est incluse dans l'entraînement, mais exclue du dénominateur Active Exact-K.

## Modifications

Nouveaux modèles de recherche `src/causal_note/s822_s810_style_teacher.py`, `scripts/s822_retrain_s810_midi256.py`. Deux têtes privées de classification K0–K6, `StandardScaler -> LogisticRegression`, distinctes des autres axes. Les deux ont les **mêmes 23 entrées physiques** : 22 indices S814 documentés + un indice spectral `182` prédéclaré ; condition `native23` = 23 statistiques moyennes; condition `aggregated61` = **exacte règle d'agrégation S810**, transposée à ces 23 features de la nouvelle DSP. Ce ne sont **pas** les 23 posteriors historiques S795 ; les reproduire à l'identique demanderait des ablations supplémentaires et risquerait de réintroduire des dépendances à un autre enseignant. Un modèle S810 n'est donc pas copié tel quel : seul son **schéma** est réentraîné sur la nouvelle acoustique sans supervision à l'inférence.

## Protocole

5 plis avec 48 tracks train, 16 inner-cal d'une autre composition, 16 outer-test d'une troisième composition ; mêmes pistes pour les deux ablations. Echantillonnage K par piste et poids inverses pour préserver prévalences. Apprendre chaque tête séparément, évaluer `native23`, `aggregated61` et une sélection entre elles sur inner-CAL exclusivement ; aucun choix sur le test. Comparer avec H0 FIT80 142213/249411 et S815 141946/249411, puis par K, undercount/overcount et composition. FIT80 est une cohorte historiquement explorée, non une validation externe.

## Audits obligatoires

SHA des 80 WAV et manifests, parité H0 S704 et horloges 256; aucune donnée CAL32/HELD/TEST. Toutes sorties concernent uniquement Active Exact-K SHADOW, pas le MIDI unifié. Source S708 et cinq autres axes inchangés; aucune latence supplémentaire due à une décision différée, mais temps CPU et latence physique restent à mesurer. Tout résultat ne peut être promu sans tests de non-régression des six axes sur source externe et validation matérielle.

## Statut

Expérience réelle à exécuter dans GitHub Actions. Aucun score S822 validé tant que les 5 folds et le rapport complet ne terminent pas.
