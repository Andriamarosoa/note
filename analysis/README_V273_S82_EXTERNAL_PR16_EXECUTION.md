# S82 — Exécution réelle des 12 producteurs PR16 sur audio extérieur, sans annotations

Date : 11 octobre 2026. [Run GitHub Actions vérifié](https://github.com/Andriamarosoa/note/actions/runs/38175040409), état **SUCCESS**, artefact `v273-s82-original-pr16-external-inference` (#11715741073), `label_blind_pr16_predictions.npz` + `report.json`.

## Ce qui a été exécuté

- Deux fichiers externes **GOAT**, commit indépendant `55c6e6668ef4546e4635c5e7eaad3062cf7f036b`, `example1_di.wav` et `example2_di.wav` (44 100 Hz mono), respectivement 52,987 et 139,592 secondes.
- SHA256 WAV exemple1 `d05060b71ea19ba2093aef23ba411bafbbeb8f402973b45eb32933c2bd10a458`, exemple2 `0e12f7bbb3ca210d3e245f10d24ba19547cfe97d58db087de89bdd4bd375384b`.
- **60 événements / fichier**, plafonnement déterministe pour preuve d'inférence. Détection préliminaire **spectral-flux, audio uniquement** et NON générateur natif V8/V27.3. Les 120 événements sont sans `true_K`, MIDI jamais ouvert.
- Extracteur spectral original PR16, 58 descripteurs + trajectoire 42×49 + géométrie 46, normalisateurs originaux, **12 véritables checkpoints neuraux PR16 du fold 0** (`v273-open-model-0`, run 37837235723). Forward NumPy GELU exact, `120×12×7` probabilités. SHA256 archive NPZ `058895c6b9e2a432f60e539dfd49f3a4e2807a7c50f32f273d1211853c3fc417`.
- Rejeu CI : chemin `scripts/s82_external_pr16_neural_inference.py`, workflow `.github/workflows/v273-s74-promotion-gate.yml`, statut vérifié.

## Distribution des propositions sans vérité terrain

Parmi 720 argmax de têtes dans chaque audio :
| Audio | K0 | K1 | K2 | K3 | K4 | K5 | K6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| exemple1 | 489 | 228 | 3 | 0 | 0 | 0 | 0 |
| exemple2 | 471 | 139 | 58 | 46 | 6 | 0 | 0 |

Distribution descriptive uniquement, **AUCUNE exactitude** sans labels. Les sorties tendent vers K0/K1 sur des événements produits par un détecteur différent du générateur natif ; cet écart de domaine est potentiellement fort et ne prouve ni échec ni succès du nombre de notes réelles.

## Validation indépendante S80 : toujours inachevée

**S80 ne se réduit pas aux 12 têtes PR16** :
1. Il consomme un `s72_poly_poly_guard_K` calculé à partir de S69→S70→S71→S72.
2. Il exploite 193 politiques S46, elles-mêmes dérivées de **16 propositions et risques S45**, qui utilisent les informations de S43/S44 et le parent S18.
3. Le script S80 [rejoué sur les fragments de développement](https://github.com/Andriamarosoa/note/blob/codex/v273-scope-rescue/scripts/audit_v273_s80_traceable_repair.py) **prend en entrée les prédictions historiques S72/S46** au lieu d'un fichier audio. Il n'existe pas d'export gelé end-to-end original avec les mêmes règles pour un WAV neuf.
4. Le détecteur temporaire spectral-flux n'est pas le générateur original. Sans générateur natif commun, une comparaison aux labels n'aurait pas la même unité de prédiction.

**Verdict : indépendant sur la provenance du signal audio et pour le forward PR16 seulement. Aucun score indépendant S80 calculé ; pas de promotion.** Le vrai prochain développement technique est le runtime natif label-blind S18/S45/S46 et S69–S72, puis application des poids S80 gelés, et enfin ouverture des annotations externes pour un score apparié. Ne pas présenter 83,09 % /43,87 % comme scores externes.
