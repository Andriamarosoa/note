# Verdict final — spectre brut dans le CNN Exact-K

Run : [37634787024](https://github.com/Andriamarosoa/note/actions/runs/37634787024).
Code évalué : `db272e8bb1dea3547c1883fa9d8398571ae63fbd`.

**Modification non retenue. `freeze_local_combo` reste la référence.**
Les quatre folds 0, 1, 2 et 4 ainsi que la synthèse ont terminé avec succès.
Le statut GitHub confirme l'exécution ; le verdict expérimental est négatif.

Sur 59 309 groupes, dont 7 385 polyphoniques :

- Exact-K poly : 2 530 → 2 499, soit 34,2586 % → 33,8389 % (−0,4198 point).
- Poly : 203 corrections contre 234 régressions, solde **−31**.
- Global : 48 454 → 48 452, soit 81,6976 % → 81,6942 % ; 503 corrections
  contre 505 régressions, solde **−2**.
- Sous-comptages poly : 3 630 → 3 600 (−30), mais surcomptages :
  1 225 → 1 286 (+61).
- Le témoin de même capacité atteint 33,7712 % poly. Le brut lui gagne seulement
  5 exacts poly (+0,0677 point), avec 31 exacts globaux de moins.
- Soldes poly par fold : 0 **+1**, 1 **−10**, 2 **−43**, 4 **+21**.

Le signal positif trouvé avec les sondes n'est pas transféré au réseau natif
par cette réinjection et ce protocole d'ajustement. Cela rejette cette
modification précise ; cela ne démontre pas que toute exploitation des niveaux
spectraux bruts serait inutile. Aucun changement de budget, sélection d'époque
ou nouveau seuil n'a été choisi après les résultats.

## Vérification indépendante des sorties

Les SHA-256 des cinq archives GitHub ont été vérifiés. Les quatre fichiers de
prédictions ont été concaténés et triés par identifiant ; chaque tableau est
identique à celui de la synthèse, probabilités comprises. Les effectifs,
Exact-K0–K6, corrections et régressions ont été recomptés depuis les lignes.
La référence est reproduite : 48 454 exacts globaux et 2 530 polyphoniques.
Aucun nouvel entraînement n'a été exécuté pour cette vérification.

Les rapports et le manifeste de vérification sont conservés dans
`analysis/evidence/v273-raw-spectral-ab/`. Les poids et prédictions complets sont
les artefacts du run. Aucun modèle n'est promu. Les folds internes ont servi
à d'autres expériences ; le fold 3 et player05 restent exclus.

## Rapport numérique du run

# Native Exact-K: normalized plus raw spectral evidence

59,309 groups, 7,385 polyphonic; folds 0,1,2,4; same 92.9 ms cached audio.
Two matched 864-parameter extensions of the archived uniform count networks.
Eight weighted fine-tuning epochs per arm. Targets are only K0–K6.

| Model | Global Exact-K | Poly Exact-K | Poly under | Poly over |
|---|---:|---:|---:|---:|
| freeze_local_combo | 81.6976% | 34.2586% | 3630 | 1225 |
| normalized_duplicate | 81.7464% | 33.7712% | 3628 | 1263 |
| raw | 81.6942% | 33.8389% | 3600 | 1286 |

| True K | Rows | Reference | Raw | Net exacts |
|---:|---:|---:|---:|---:|
| 0 | 39652 | 95.922% | 95.892% | -12 |
| 1 | 12272 | 64.285% | 64.619% | +41 |
| 2 | 3445 | 34.340% | 33.440% | -31 |
| 3 | 2289 | 39.974% | 39.755% | -5 |
| 4 | 1207 | 34.548% | 34.714% | +2 |
| 5 | 355 | 4.225% | 5.070% | +3 |
| 6 | 89 | 0.000% | 0.000% | +0 |

| Fold | Raw vs reference: global net | Low-K net | Poly net | Raw vs capacity control: poly net |
|---:|---:|---:|---:|---:|
| 0 | +0 | -1 | +1 | -11 |
| 1 | -34 | -24 | -10 | +1 |
| 2 | +8 | +51 | -43 | -12 |
| 4 | +24 | +3 | +21 | +27 |

No automatic promotion. No epoch, threshold or arm is selected on these evaluation labels.
These composition folds have been used in earlier experiments; this is an exploratory comparison.
