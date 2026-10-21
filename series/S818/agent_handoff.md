# S818 — dossier d'audit reproductible pour agents externes

> Ceci est un inventaire de ressources, pas une preuve que des agents NVIDIA ou Work ont été lancés. Aucun de ces modèles n'a été consulté durant S818.

## Ressources visibles

- Dépôt public : https://github.com/Andriamarosoa/note
- PR de recherche DRAFT : https://github.com/Andriamarosoa/note/pull/122
- Code des 6 têtes K>=1..6 : src/causal_note/s818_activek_ordinal.py
- Programme véritable 5 plis : scripts/s818_train_ordinal256.py
- Audit causal/indépendance : test/test_s818_activek_ordinal.py
- Workflow source : .github/workflows/note-s818-activek-ordinal.yml
- Run complet PASS : https://github.com/Andriamarosoa/note/actions/runs/39546770781
- Artefact généré (5 groupes de 6 checkpoints, décisions et métriques) : https://github.com/Andriamarosoa/note/actions/runs/39546770781/artifacts/12322741581
- Autre run dû à un correctif de compilation, à contrôler séparément : https://github.com/Andriamarosoa/note/actions/runs/39546788233
- Rapports : series/S818/resume.md et series/S818/resultat.md
- PCM original GuitarSet : URL et SHA enregistrées sous analysis/evidence/v273-s613-strict-live/protocol.json, vérifiées dans CI. Le code GitHub et le workflow sont publics, mais le téléchargement automatique des artefacts GitHub Actions peut exiger une authentification GitHub et les autorisations associées. Aucune connexion d'agent externe à GitHub n'est présumée.

## Résultat scientifique FIT80, 5 plis composition-exclus

- H0 historique : 142213/249411 = 57,01953803 %
- S815 privé : 141946/249411 = 56,91248582 %
- S817 identique à S815 : 141946/249411 = 56,91248582 %
- S818 six têtes ordinales : **135694/249411 = 54,40578002 %**
- Différences : -2,61375801 points vs H0, -2,50670580 points vs S815.
- S818 sous-comptages 73512 ; surcomptages 40205.
- Scores S818 : BN1 27669/47770 ; Funk2 27246/48827 ; Jazz1 22959/40548 ; Rock3 28487/54917 ; SS1 29333/57349.
- Objectif >85 % par caractéristique non atteint. S818 REJETÉE comme promotion. FIT80 historiquement exploré : ne pas présenter cette mesure comme une généralisation externe validée.

## Consignes de revue réellement indépendante

Chaque modèle/agent doit vérifier le code *et* les logs *et* les artefacts signés, formuler sa propre hypothèse testable avant de lire les propositions des autres, et rendre séparément :
1. Audit des fuites de données/train/CAL/test (48/16/16 par composition), dérives de partition et validité de la métrique Active Exact-K K1–K6.
2. Audit DSP causal des hautes polyphonies K4–K6, harmoniques/reconstruction et changements de timbre/tone, notamment sur Rock3 et SS1.
3. Audit mathématique des six prédicteurs ordinaux, class balance, rare K5/K6, calibration, monotonicité des probabilités et risque d'écrasement des classes.
4. Nouvelle architecture **privée par caractéristique** sans partage de poids ou de décodeur entraîné. Le modèle ne doit pas lire les sorties Frame Pitch/Onset/Offset/Full Note.
5. Test reproductible, scripts exacts, vrais scores GuitarSet et seconde source indépendante, verdict accept/reject avec analyse par K, musique et composition.
6. Coût d'inférence, 256 échantillons par hop à 44,1 kHz (5,80499 ms de granularité) et latence end-to-end matérielle distincte. Interdiction de futur, hold, post-onset et régression S708/LIVE.

Les six agents historiques et les modèles NVIDIA ne sont **pas** exécutés automatiquement par GitHub Actions ; l'utilisateur doit leur fournir l'accès aux ressources requises dans leur environnement de connexion, ou prévoir un appel explicite dans un workflow muni des credentials nécessaires. Ne jamais présenter un simple appel à un classifieur scikit-learn comme une collaboration avec un autre LLM.
