---
serie: S819
statut: en_cours
type: experience
reference: S708
branche: codex/note-s819-independent-model-review
run_id: en_attente
commit_sha: en_attente
promotion: false
---
# S819 — Consultations multi-modèles NVIDIA et contre-audit indépendant des erreurs Active Exact-K

## Objectif

Faire participer des modèles NVIDIA **réellement appelés** à la recherche de causes des erreurs Active Exact-K, après régressions S814–S818, tout en conservant séparation stricte des architectures, décisions causales MIDI256 et zéro régression des cinq autres tâches. Objectif final >85 % par caractéristique et validation externe sans test leakage. Aucun gain ne sera revendiqué sans benchmark réel.

## Hypothèse

S815 141946/249411 (56,9125 %) et S818 135694/249411 (54,4058 %) contre H0 142213/249411 (57,0195 %) sur GuitarSet FIT80 : la multiplication de têtes (et de sélection ordinale) n'a pas suffi. Les causes possibles doivent être évaluées par des avis **indépendants, non induits**, puis par des ablations falsifiables ; ne pas présumer harmonie, labels, loss ni calibration comme causes.

## Modifications

Dossier de preuve identique `series/S819/model_brief.md`. Script `scripts/s819_blind_nvidia.py` appelle via NVIDIA NIM le trio distinct **Gemma 4 31B**, **Nemotron 3.5 Lightning 30B**, **GLM 5.3 Flash**. Les trois avis sont enregistrés séparément, portent SHA256 dossier/prompt, un statut réel provider, une règle de substance et un refus automatique en cas de modèle inaccessible ; aucun agent ne reçoit la sortie de l'autre. Six contrôles GitHub indépendants ont les rôles Données, Acoustique, Architecture, Apprentissage, Métriques, LIVE, mais ne sont **pas** les six agents Work personnels. L'appel NVIDIA utilise uniquement GitHub Actions Secret `NVIDIA_API_KEY`, jamais affiché ni stocké dans les sources.

## Protocole

Base scientifiquement signée S818 [workflow #39546788233](https://github.com/Andriamarosoa/note/actions/runs/39546788233) et H0/S815 sur GuitarSet FIT80 80 pistes, 249411 trames Active-K au pas 256. Les modèles externes n'accèdent qu'à une **synthèse publique bornée** de comptes réels et de code, sans WAV, credentials, labels par morceau, ou CAL/HELD/TEST. Audit impartial AVANT sélection d'une nouvelle conception. Les propositions sont systématiquement revues de manière contradictoire puis testées avec les mêmes splits 48/16/16. Aucun conseiller ne décide de la promotion.

## Audits obligatoires

Au minimum : six rapports automatiques séparés, identités de modèles vérifiées, prompts et preuves SHA identiques, aucun texte du conseiller dans les prompts des autres, examen des limites/fuites, revue de propositions sans promotion. Le workflow échoue (fail-closed) pour avis manquant ou vide. Si fournisseur inaccessible, archiver le statut exact et ne pas inventer de consultation. Aucune commande ne modifie la référence S708, les 6 scores du MIDI unifié ou LIVE. La résolution native 256 échantillons (5,80499 ms) n'est pas une latence iRig mesurée.

## Statut

Workflow et modèle de vérification en préparation, avis NVIDIA **non encore obtenus**. Les Work agents personnalisés n'ont pas été lancés ; leur mise en route exige un accès Work distinct. Promotion=false.
