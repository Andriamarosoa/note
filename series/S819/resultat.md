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
# S819 — Résultats réels de la revue NVIDIA indépendante

## Exécution

En cours : 3 consultations externes indépendantes NVIDIA (Gemma 4 31B, Nemotron 3.5 Lightning 30B, GLM 5.3 Flash) et 6 contrôles techniques séparés du dépôt. Un nom de modèle configuré dans le code n'est **pas** un appel réussi. Toute réponse/erreur doit être archivée avec le numéro GitHub Actions avant d'annoncer son succès. Aucun score de musique S819 mesuré à ce stade.

## Métriques

| Caractéristique | S708 officielle CAL32 | S819 SHADOW |
|---|---:|---|
| Onset F1 | 74,2639 % | non mesuré |
| Exact-K | 62,0455 % | non mesuré |
| Active Exact-K | 56,2762 % | non mesuré |
| Offset accuracy | 77,3292 % | non mesuré |
| Full Note F1 | 57,1319 % | non mesuré |
| Frame Pitch F1 | 76,4626 % | non mesuré |

Comparateur **différent** FIT80 GuitarSet : H0 57,0195 % ; S815 et S817 56,9125 % ; S818 54,4058 %. Avis textuels NVIDIA ne sont pas des scores.

## Comparaison à S708

Aucune modification du code LIVE, S708 ou d'un système MIDI joint. Tous les avis sont non contraignants et portent seulement sur un résumé de résultats exploratoires passés. Les six métriques officielles ne sont pas revalidées à ce stade.

## Audits

| Domaine | Statut | Critère |
|---|---|---|
| Données | UNKNOWN | 80 sources et séparation FIT/CAL/TEST explicitement vérifiées dans le dossier d'avis |
| Acoustique | UNKNOWN | Causalité MIDI256 et hypothèses adverses |
| Architecture | UNKNOWN | Seuils K, poids privés et provenance |
| Apprentissage | UNKNOWN | Échantillonnage des rares K, priors et absence de fuites |
| Métriques | UNKNOWN | Comptes authentiques S815/S818 et confusion par K |
| LIVE | UNKNOWN | Aucune latence supplémentaire, sans benchmark physique prétendu |

## Verdict

En attente de **réponses réelles NVIDIA**, de contre-avis et d'une nouvelle hypothèse testée. Aucune promotion ni résultat >85 % déclaré. Les six agents Work ne sont pas remplacés par six scripts CI.

## Prochaine action

Exécuter les appels externes dans GitHub Actions via secret déjà utilisé dans les anciens workflows, vérifier les réponses individuelles, les critiquer sans les accepter automatiquement, puis créer un entraînement S819 si une expérience justifiée et reproductible peut être définie.


## Résultat confirmé de la revue externe indépendante

Le [run #39552451395](https://github.com/Andriamarosoa/note/actions/runs/39552451395) est **SUCCESS** : 3 appels NVIDIA NIM réellement exécutés et 3 réponses non vides/substantielles reçues, **Gemma 4 31B**, **Nemotron 3.5 Lightning 30B**, **GLM 5.3 Flash**. Six audits locaux distincts ont passé les vérifications Données, Acoustique, Architecture, Apprentissage, Métriques et LIVE ; ils ne sont **pas** les six agents personnalisés Work. L'artefact du run archive les neuf rapports et le résultat de contrôle d'identité, du hash du même dossier impartial et des prompts. Les textes sont des *avis exploratoires*, non une preuve de score. Les erreurs et hypothèses rejetées sont détaillées dans `series/S819/external_reviews_critique.md`.

Constat critique : les avis convergent sur trois familles de tests possibles (preuves acoustiques/harmoniques, sampling/weights, sélection K4–K6), mais présentent des erreurs : Nemotron confond 128 samples et 128 ms ainsi que les 48 morceaux train et 16 cal ; GLM affirme à tort que sur/sous-comptages augmentent tous deux dans S818 (surcomptages diminuent de 41935 à 40205) ; Gemma qualifie à tort une exactitude K1→K6 de linéaire et confond parfois hop et fenêtre DSP. Aucune recommandation de modifier les hops, d'ouvrir CAL32/HELD/TEST ou de désactiver la causalité n'est acceptée.

Contre-analyse réelle depuis l'artefact S818 : K4 **43,99%**, K5 **9,64%**, K6 **0,78%** (versus S815 respectivement 36,93%, 29,78%, 17,40%). Bootstrap exploratoire de 20000 réplications depuis compteurs piste et composition en `series/S819/paired_bootstrap_result.md` : S818−H0 intervalle quantile 95% par composition **[−4,014 ; −0,931] points**, avec seulement 5 compositions, donc pas une validation externe. Aucun nouveau modèle S819 entraîné ni nouvelle référence promue.

Hypothèse ablatable prioritaire choisie après **contre-vérification** des vrais avis : expliquer pourquoi les 2 nouveaux experts S817 n'ont changé aucune décision, sans les réentraîner. Relecture des neuf politiques de calibration originales : base choisie sur 4 compositions, `max_high@0.90` sur la cinquième, avec gain interne infime ; `max_high@0.50` et `override@0.50` dégradent le plus souvent la calibration. Un test exhaustif de leur activation sur vrai PCM MIDI256 80 pistes, séparant cal et test extérieur pour éviter toute fuite, a été lancé : [run #39553290558](https://github.com/Andriamarosoa/note/actions/runs/39553290558). Ce diagnostic est **posthoc**, n'a pas le droit de choisir une politique sur les scores d'évaluation et ne peut pas autoriser une promotion. Six scores officiels S708 et LIVE restent inchangés.



## Clôture S819 : revue NVIDIA + contre-épreuve acoustique vraie

**Réalisée et vérifiée.** Trois modèles NVIDIA **réellement appelés** (Gemma 4 31B, Nemotron 3.5 Lightning 30B, GLM 5.3 Flash), trois retours archivés et six contrôles GitHub locaux PASS [run #39552451395](https://github.com/Andriamarosoa/note/actions/runs/39552451395) ; six agents personnels Work **non invoqués**. Les avis sont de simples hypothèses ; l'audit critique est dans `series/S819/external_reviews_critique.md`.

La contre-épreuve causale S819 a été exécutée en SUCCESS sur 80 pistes GuitarSet originales PCM SHA, 5 folds composition-exclus, 306536 frames audio à MIDI256, 249411 frames actives, sans nouvel entraînement [run #39553290558](https://github.com/Andriamarosoa/note/actions/runs/39553290558), artefact [#12324841883](https://github.com/Andriamarosoa/note/actions/runs/39553290558/artifacts/12324841883). Le score S815/S817 141946/249411 **56,91248582 %** et **0 décision modifiée S817** sont reproduits à l'identique. Les seuils hypothétiques `max_high@0.70` (4867 changements, **141263/249411 = 56,63864064 %**) et `max_high@0.50` (19487 changements, **139085/249411 = 55,76538324 %**) sont **régressifs**, car les surcomptages supplémentaires dépassent les sous-comptages supprimés. `override@0.70` 140924/249411 = 56,50272041 % et `override@0.50` 138705/249411 = 55,61302429 % régressent aussi. Le minuscule +2 bonnes décisions de `override@0.90` est posthoc et non promouvable ; ses 8 changements n'établissent aucune robustesse.

Constat : **oui, le sélecteur neutralisait les nouvelles têtes**, mais ce choix préservait une meilleure précision. L'hypothèse qu'un simple abaissement de seuil suffit est **réfutée** ; nouveaux signaux acoustiques discriminants à proposer/tester dans une série distincte, sans choisir sur les outer résultats FIT80 déjà explorés. Aucun apprentissage S819, aucune augmentation démontrée au-delà de H0 57,01953803 %, aucune certification externe, aucune nouvelle mesure LIVE ; **promotion=false**, unification=false, cible >85 % non atteinte. Voir le tableau complet des neuf politiques dans `series/S819/gate_forensic_result.md`.
