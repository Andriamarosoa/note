---
serie: S819
statut: termine
type: experience
reference: S708
branche: codex/note-s819-independent-model-review
run_id: 39552451395
commit_sha: 4adff035eea3fabb1d71f095c15c3ff8a370e549
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


## Résultat confirmé de la revue externe indépendante

Le [run #39552451395](https://github.com/Andriamarosoa/note/actions/runs/39552451395) est **SUCCESS** : 3 appels NVIDIA NIM réellement exécutés et 3 réponses non vides/substantielles reçues, **Gemma 4 31B**, **Nemotron 3.5 Lightning 30B**, **GLM 5.3 Flash**. Six audits locaux distincts ont passé les vérifications Données, Acoustique, Architecture, Apprentissage, Métriques et LIVE ; ils ne sont **pas** les six agents personnalisés Work. L'artefact du run archive les neuf rapports et le résultat de contrôle d'identité, du hash du même dossier impartial et des prompts. Les textes sont des *avis exploratoires*, non une preuve de score. Les erreurs et hypothèses rejetées sont détaillées dans `series/S819/external_reviews_critique.md`.

Constat critique : les avis convergent sur trois familles de tests possibles (preuves acoustiques/harmoniques, sampling/weights, sélection K4–K6), mais présentent des erreurs : Nemotron confond 128 samples et 128 ms ainsi que les 48 morceaux train et 16 cal ; GLM affirme à tort que sur/sous-comptages augmentent tous deux dans S818 (surcomptages diminuent de 41935 à 40205) ; Gemma qualifie à tort une exactitude K1→K6 de linéaire et confond parfois hop et fenêtre DSP. Aucune recommandation de modifier les hops, d'ouvrir CAL32/HELD/TEST ou de désactiver la causalité n'est acceptée.

Contre-analyse réelle depuis l'artefact S818 : K4 **43,99%**, K5 **9,64%**, K6 **0,78%** (versus S815 respectivement 36,93%, 29,78%, 17,40%). Bootstrap exploratoire de 20000 réplications depuis compteurs piste et composition en `series/S819/paired_bootstrap_result.md` : S818−H0 intervalle quantile 95% par composition **[−4,014 ; −0,931] points**, avec seulement 5 compositions, donc pas une validation externe. Aucun nouveau modèle S819 entraîné ni nouvelle référence promue.

Hypothèse ablatable prioritaire choisie après **contre-vérification** des vrais avis : expliquer pourquoi les 2 nouveaux experts S817 n'ont changé aucune décision, sans les réentraîner. Relecture des neuf politiques de calibration originales : base choisie sur 4 compositions, `max_high@0.90` sur la cinquième, avec gain interne infime ; `max_high@0.50` et `override@0.50` dégradent le plus souvent la calibration. Un test exhaustif de leur activation sur vrai PCM MIDI256 80 pistes, séparant cal et test extérieur pour éviter toute fuite, a été lancé : [run #39553290558](https://github.com/Andriamarosoa/note/actions/runs/39553290558). Ce diagnostic est **posthoc**, n'a pas le droit de choisir une politique sur les scores d'évaluation et ne peut pas autoriser une promotion. Six scores officiels S708 et LIVE restent inchangés.



## Clôture S819 : revue NVIDIA + contre-épreuve acoustique vraie

**Réalisée et vérifiée.** Trois modèles NVIDIA **réellement appelés** (Gemma 4 31B, Nemotron 3.5 Lightning 30B, GLM 5.3 Flash), trois retours archivés et six contrôles GitHub locaux PASS [run #39552451395](https://github.com/Andriamarosoa/note/actions/runs/39552451395) ; six agents personnels Work **non invoqués**. Les avis sont de simples hypothèses ; l'audit critique est dans `series/S819/external_reviews_critique.md`.

La contre-épreuve causale S819 a été exécutée en SUCCESS sur 80 pistes GuitarSet originales PCM SHA, 5 folds composition-exclus, 306536 frames audio à MIDI256, 249411 frames actives, sans nouvel entraînement [run #39553290558](https://github.com/Andriamarosoa/note/actions/runs/39553290558), artefact [#12324841883](https://github.com/Andriamarosoa/note/actions/runs/39553290558/artifacts/12324841883). Le score S815/S817 141946/249411 **56,91248582 %** et **0 décision modifiée S817** sont reproduits à l'identique. Les seuils hypothétiques `max_high@0.70` (4867 changements, **141263/249411 = 56,63864064 %**) et `max_high@0.50` (19487 changements, **139085/249411 = 55,76538324 %**) sont **régressifs**, car les surcomptages supplémentaires dépassent les sous-comptages supprimés. `override@0.70` 140924/249411 = 56,50272041 % et `override@0.50` 138705/249411 = 55,61302429 % régressent aussi. Le minuscule +2 bonnes décisions de `override@0.90` est posthoc et non promouvable ; ses 8 changements n'établissent aucune robustesse.

Constat : **oui, le sélecteur neutralisait les nouvelles têtes**, mais ce choix préservait une meilleure précision. L'hypothèse qu'un simple abaissement de seuil suffit est **réfutée** ; nouveaux signaux acoustiques discriminants à proposer/tester dans une série distincte, sans choisir sur les outer résultats FIT80 déjà explorés. Aucun apprentissage S819, aucune augmentation démontrée au-delà de H0 57,01953803 %, aucune certification externe, aucune nouvelle mesure LIVE ; **promotion=false**, unification=false, cible >85 % non atteinte. Voir le tableau complet des neuf politiques dans `series/S819/gate_forensic_result.md`.
