# S819 — Contre-analyse sur les vrais scores enregistrés : K1–K6 et bootstrap

## Provenance

Rapport `result/report.json` du [workflow S818 final GitHub #39546788233](https://github.com/Andriamarosoa/note/actions/runs/39546788233), artefact [#12322628075](https://github.com/Andriamarosoa/note/actions/runs/39546788233/artifacts/12322628075). 80 pistes, 5 compositions indépendantes par pli, 249411 frames GT K>0. Rapport complet calculé et enregistré par pipeline MIDI256, sans nouvelle estimation LLM ni ouverture CAL32/HELD/TEST.

## Exactitude par vrai nombre de notes

| Vrai K | S815 préexistante | S818 ordinale, mesure authentique |
|---|---:|---:|
| K1 | 78,32 % | **78,51 %** (87938/112014) |
| K2 | 42,39 % | **35,41 %** (16397/46311) |
| K3 | 46,19 % | **41,73 %** (14263/34182) |
| K4 | 36,93 % | **43,99 %** (15481/35190) |
| K5 | 29,78 % | **9,64 %** (1573/16323) |
| K6 | 17,40 % | **0,78 %** (42/5391) |

La contre-intuition cruciale est qu'une amélioration sur K4 masque des pertes très graves sur K5 et K6. Tout sélectionneur doit donc être évalué par K et en précision globale, pas uniquement au global. Les observations ne prouvent pas à elles seules que l'échantillonnage ou les partiels harmoniques sont la cause.

## Bootstrap paire de piste, exploratoire, non validant

Ré-échantillonnage de **20000** échantillons avec remplacement, graine déterministe 819, scores recomposés depuis les compteurs entiers de chaque piste S818 : `source.correct` (H0), `frozen_s815.correct`, `new.correct`, et `new.frames` sur la même piste. Pour chaque tirage, différence de précision en points = `100*(correct_candidat - correct_h0)/sum(frames)`. Pas de recherche de paramètre, aucun GT caché.

- **Bootstrap 80 pistes** (échantillonnage des 80 enregistrements) : S815−H0 quantiles [2,5; 50; 97,5] % **[−1,990 ; −0,068 ; +1,618] points** ; S818−H0 **[−4,673 ; −2,589 ; −0,604] points**.
- **Bootstrap cinq compositions** (ré-échantillonnage des cinq unités groupées) : S815−H0 **[−1,732 ; −0,107 ; +1,329] points** ; S818−H0 **[−4,014 ; −2,614 ; −0,931] points**.

Ces intervalles sont des **quantiles bootstrap exploratoires**, pas une démonstration formelle de généralisation externe ni des tests statistiques indépendants : les pistes partagent joueurs/styles et il n'y a que **cinq compositions**. Le signe de la régression S818 est robuste à ces deux schémas simples sur FIT80, contrairement au minuscule écart S815−H0.

## Deltas S818 sur H0 par composition (mêmes pistes, en points)

| Composition | S815−H0 | S818−H0 |
|---|---:|---:|
| BN1 | −2,221 | −2,248 |
| Funk2 | +1,851 | +0,344 |
| Jazz1 | −1,909 | −1,689 |
| Rock3 | +1,907 | **−4,321** |
| SS1 | −0,668 | **−4,455** |

La faiblesse Rock3 et SS1 provient donc d'une **vraie régression de S818 relativement à H0**, et pas seulement d'une difficulté intrinsèque aux compositions. C'est un test contradictoire important des suggestions NVIDIA expliquant tout par la nature du timbre.

## Reproductibilité exacte

```python
import io,json,zipfile,numpy as np
with zipfile.ZipFile("s818_completed.zip") as z:
    r=json.loads(z.read("result/report.json"))
a=np.asarray([[t["source"]["correct"],t["frozen_s815"]["correct"],
               t["new"]["correct"],t["new"]["frames"]] for t in r["tracks"]])
rng=np.random.default_rng(819)
idx=rng.integers(len(a),size=(20000,len(a)),dtype=np.int32)
b=a[idx].sum(axis=1)
for j in (1,2):
    print(np.percentile(100*(b[:,j]-b[:,0])/b[:,3],[2.5,50,97.5]))
```

Ne jamais exploiter ces résultats FIT80 exploratoires comme vérité permettant de choisir des paramètres après coup. Le futur objectif >85 % demande une validation indépendante. `promotion=false`.
