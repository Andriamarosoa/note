# Pourquoi les résultats d'Exact K restent faibles

**Audit de conception suivant :** un [contrat entrée/cible incomplet](v273-design-contract.md)
est démontré par un contre-exemple reproductible : les quatre mêmes entrées
locales peuvent recevoir K=0 ou K=1 selon une proposition du groupe suivant
non transmise au modèle. La géométrie d’appartenance et sa disponibilité
temporelle doivent être ajoutées explicitement. Un composant et sept tests
sont préparés ; leur intégration et le gain d’Exact K restent à mesurer.

**Le défaut principal constaté est un comptage polyphonique insuffisamment
appris, déjà visible sur les données d'apprentissage.** Il s'accompagne d'une
compétition avec les classes 0/1 et d'un compromis d'objectif qui favorise
certains K au détriment des autres. L'audit précédent des seuls surcomptages
ne répondait pas entièrement à cette question.

Cet audit examine les **modèles natifs de comptage à 31 trames**, sans
correcteur, entraînés dans l'expérience `36351028493`. Le seul fold externe
est le **fold 3**. Leurs versions internes sont également évaluées sur leurs
propres partitions d'apprentissage et de validation initiales. Les poids et
les huit époques déjà terminées sont inchangés.

**Prolongement terminé et audité :** le [verdict à 16 époques](v273-training-budget-final.md)
confirme un gain polyphonique interne de 5,87 points sans pondération,
mais aucun gain net avec pondération (−0,24 point). Les deux modèles
régressent après le point intermédiaire à 12 sur la validation polyphonique
alors que l’apprentissage progresse. Les mesures figées ci-dessous restent
celles à huit époques.

**Test suivant terminé :** le [remplacement de la normalisation spectrale
par une échelle fixe](v273-normalization-final.md), comparé à entraînement
identique de 12 époques, gagne 47 comptes polyphoniques internes
(26,53 % → 28,75 %), mais perd 95 comptes K=1. Le solde global est −11 ;
celui des K=0 à 3 est −35. Les scores polyphoniques d’apprentissage restent
presque identiques (39,86 % et 39,80 %). Le changement ne satisfait pas le
critère de non-régression sur les petits K et ne résout pas la difficulté
générale du comptage. Les nouvelles erreurs K=1 sont auditées dans ce rapport.

## 1. Le comptage est faible même sur les exemples déjà vus

Les mêmes checkpoints finaux ont été rejoués en mode inférence, sans dropout,
sur les données vues et sur le fold externe :

| Modèle final | Apprentissage : Exact K pour vrais K≥2 | Fold 3 non vu : Exact K pour vrais K≥2 | Écart |
|---|---:|---:|---:|
| Sans pondération | 2 821 / 7 385 = **38,20 %** | 670 / 1 969 = **34,03 %** | 4,17 points |
| Avec pondération | 3 220 / 7 385 = **43,60 %** | 766 / 1 969 = **38,90 %** | 4,70 points |

Le modèle n'a donc pas un bon comptage sur les exemples appris qui
s'effondrerait seulement sur les nouveaux morceaux. **Plus de la moitié des
comptes polyphoniques sont déjà incorrects sur l'apprentissage**, dans les
deux variantes. L'écart de généralisation existe, mais ne suffit pas à
expliquer la faiblesse du score.

La même difficulté apparaît avec les checkpoints internes : 29,03 % sur
l'apprentissage et 19,09 % sur la validation pour l'uniforme ; 41,01 % et
30,65 % pour le pondéré. Cette partition montre un écart d'environ dix
points. Il ne faut donc pas déclarer tout problème de généralisation absent.
Les modèles internes et finaux ont des données et des graines différentes ;
leur comparaison ne constitue pas un test causal du nombre de données.

**Ce résultat localise une insuffisance d'apprentissage du comptage.** Il ne
permet pas, à lui seul, de choisir entre budget d'optimisation, capacité du
réseau, information conservée par les entrées et ambiguïtés de supervision.
Dire « il faut seulement plus d'époques » serait prématuré.

## 2. L'objectif majoritaire masque la faiblesse sur les petits accords

Dans l'apprentissage final, **51 924 / 59 309 exemples, soit 87,55 %, sont
K=0 ou K=1**. Le réseau est optimisé avec une entropie croisée à sept classes
et un seul compte comme cible. Sans pondération, ces exemples constituent
87,55 % de la masse des poids d'exemples ; ce pourcentage n'est pas la part
de la perte numérique, qui dépend aussi des erreurs de chaque exemple.

Sur le fold 3, l'Exact K global atteint 83,57 % sans pondération, alors que
le score polyphonique reste à 34,03 %. Le premier nombre est porté notamment
par les 10 285 groupes K=0, correctement reconnus à 96,94 %.

La performance de chaque K rend cette limite visible :

| Vrai K | Groupes du fold 3 | Uniforme : apprentissage | Uniforme : fold 3 | Pondéré : apprentissage | Pondéré : fold 3 |
|---|---:|---:|---:|---:|---:|
| 0 | 10 285 | 97,15 % | 96,94 % | 94,98 % | 94,90 % |
| 1 | 3 025 | 67,69 % | 70,35 % | 64,55 % | 67,17 % |
| 2 | 806 | **34,43 %** | **30,15 %** | **42,64 %** | **38,59 %** |
| 3 | 628 | **52,16 %** | **49,68 %** | **37,18 %** | **34,39 %** |
| 4 | 405 | 31,48 % | 25,93 % | 63,05 % | 53,58 % |
| 5 | 109 | 17,18 % | 9,17 % | 39,15 % | 20,18 % |
| 6 | 21 | 0,00 % | 0,00 % | 0,00 % | 0,00 % |

Pour K=2 sans pondération, **362 sous-comptages** et **201 surcomptages**
accompagnent les 243 comptes exacts. Un audit du seul surcomptage ignore donc
la plus grande partie de son déficit. K=2 et K=3 concentrent **879 des
1 299 erreurs polyphoniques** de l'uniforme, soit 67,67 % ; avec pondération,
ils représentent **907 / 1 203**, soit 75,39 %.

Les classes K=5/6 sont très peu apprises, mais elles ne représentent que
120 erreurs polyphoniques sans pondération et 108 avec. Leur rareté ne peut
pas expliquer à elle seule le score médiocre de l'ensemble.

## 3. Un effet causal démontré : le choix de pondération déplace la difficulté

L'expérience contrôlée conserve données, poids initiaux, graines, ordre des
lots et huit époques. Seuls les poids des exemples dans la perte changent.

**La pondération améliore le score polyphonique total : 670 → 766 comptes
exacts, soit +96 et +4,88 points.** Elle corrige 284 erreurs polyphoniques
et dégrade 188 réponses jusque-là exactes. La qualifier simplement de cause
du mauvais score, parce qu'elle augmente les surcomptages, serait incorrect.

Son effet est très inégal :

| Vrai K | Variation des comptes exacts avec la pondération |
|---|---:|
| 0 | −210 |
| 1 | −96 |
| 2 | +68 |
| 3 | **−96** |
| 4 | **+112** |
| 5 | +12 |
| 6 | 0 |

Le gain agrégé vient surtout de K=4, au prix d'une forte dégradation de K=3.
Sur K=2/3 réunis, les comptes exacts passent de **555 à 527 sur 1 434**,
soit 38,70 % → 36,75 %. L'objectif pondéré actuel ne résout donc pas le
problème des petits accords. Ce compromis existe aussi sur l'apprentissage :
la précision K=3 y passe de 52,16 % à 37,18 %.

La masse des poids d'exemples K=0/1 descend de 87,55 % à 65,87 % avec cette
pondération. C'est une intervention réelle sur l'objectif ; son effet
montre que le choix de perte influence matériellement les frontières entre K.
Il ne démontre pas que le simple déséquilibre des effectifs explique toutes
les erreurs, ni qu'une pondération encore plus forte les corrigerait.

## 4. Deux obstacles distincts dans les décisions

À probabilités figées, on peut examiner le meilleur score parmi K=2 à 6.
Cette opération utilise le vrai statut polyphonique pour l'analyse : **c'est
un oracle diagnostique, pas une correction utilisable ni un nouveau score**.

La partition suivante est exhaustive et sans chevauchement :

| Erreur sur un vrai K≥2 | Uniforme | Pondéré |
|---|---:|---:|
| Le réseau choisit 0/1, alors que le vrai K est premier parmi 2–6 | **364** | **260** |
| Le réseau choisit 0/1 et se trompe aussi dans le classement entre 2–6 | 137 | 78 |
| Le réseau choisit bien un K≥2, mais pas le bon nombre | **798** | **865** |
| Total des erreurs polyphoniques | **1 299** | **1 203** |

Le premier obstacle est donc la compétition avec 0/1 ; le second est une
distinction insuffisante entre les nombres de notes. Même si le statut
polyphonique était fourni parfaitement, le classement actuel ne donnerait
que 1 034 / 1 969 comptes exacts en uniforme et 1 026 / 1 969 en pondéré,
soit environ 52 %. Ce n'est pas une limite théorique d'une future architecture :
c'est la limite de ce classement déjà appris sous cet oracle précis.

**Changer uniquement la décision « une ou plusieurs attaques » laisse
935 erreurs de classement en uniforme et 943 en pondéré.** La pondération
déplace surtout le compromis entre classes, sans résoudre cette distinction
fine. Les informations additionnelles ou un apprentissage différent doivent
être évalués directement sur cette difficulté.

## 5. Ce que les audits de données expliquent, et ce qu'ils n'expliquent pas

Les défauts temporels trouvés auparavant ont été corrigés dans les entrées de
cette expérience. Le passage contrôlé de 23 à 31 trames gagne 34 comptes
exacts polyphoniques sans pondération et 38 avec : un progrès réel, limité.
Il n'a pas résolu le déficit d'apprentissage observé ici.

L'audit acoustique a vérifié les cibles et 1 541 spectres depuis l'audio
original. Il a documenté une concurrence entre groupes dans certains
surcomptages, ainsi que des erreurs sans annotation étrangère. Ces constats
doivent informer une future supervision des attaques, mais ne suffisent pas
à attribuer le faible score global aux harmoniques, aux résonances ou à un
problème général de cache.

Le graphe natif combine des représentations spectrales moyennes/maximales
et les candidats pour classer directement K. Les têtes explicites de
localisation, de sélection et d'affectation des attaques ont été retirées
du modèle de comptage. C'est une limite de formulation à tester, **pas une
preuve que cette architecture ne peut jamais apprendre le comptage**.

Dans l'ancien contrôle de la chaîne V27.3 complète, 635 des 1 133 erreurs
polyphoniques étaient déjà fausses dans les trois propositions de comptage.
Un simple choix entre ces propositions ne pouvait pas les réparer. Ce
diagnostic historique soutient la priorité au comptage natif ; il n'est pas
une comparaison directe avec les nouveaux checkpoints à fenêtre corrigée.

## Priorité de correction justifiée

La priorité est **l'apprentissage de la distinction entre K=2, K=3 et K=4**,
avec un suivi explicite des erreurs par K. Augmenter une classe au détriment
d'une autre ou retirer des sorties trop hautes ne démontre pas un meilleur
comptage.

Le [test contrôlé du budget](v273-training-budget-final.md) est maintenant
terminé. Il montre une marge d’amélioration par prolongation, surtout sans
pondération, puis une divergence entre progrès sur les exemples vus et
recul polyphonique interne entre 12 et 16. Augmenter le budget seul n’est
donc pas une correction générale démontrée.

La suite doit isoler une modification native de représentation, supervision
ou régularisation, avec son propre témoin et des courbes par K. Les pertes
auxiliaires de localisation ou d’appartenance au groupe restent des pistes
à tester. Les audits ne départagent pas encore ces mécanismes. Aucun nouveau
modèle n’est promu ; le détail des confusions et leurs régressions est
maintenant documenté au-delà du point à huit époques.

## Vérifications et sources

Les quatre checkpoints ont produit **267 794 inférences** : 205 332 sur
leurs lignes d'apprentissage et 62 462 sur leurs partitions non vues. Les
62 462 probabilités et décisions non vues reproduisent exactement les
sorties archivées (écart maximal nul). Les poids sont identiques avant et
après l'audit. Les matrices, pertes, effectifs par classe et partitions
d'erreurs ont été recalculés depuis les fichiers de prédictions.

- [Exécution réussie](https://github.com/Andriamarosoa/note/actions/runs/36375624830).
- [Archives avec prédictions vues/non vues](https://github.com/Andriamarosoa/note/releases/tag/v273-learning-bottleneck-36375624830).
- [Diagnostic recalculé](v273-performance-causes-diagnosis.json) et [provenance](v273-performance-causes-sources.json).
- Rapports complets : [uniforme](v273-learning-uniform-audit.json), [pondéré](v273-learning-weighted-audit.json).
- [Expérience contrôlée des fenêtres et poids](v273-window-pair-results.md), [audit acoustique](v273-acoustic-residual.md), [ancien contrôle complet](v273-control-fold3-audit.md).

La référence officielle V27.3 à 42,6019 % concerne une autre chaîne et une
autre population ; les chiffres de ce composant ne la remplacent pas. Le
fold 3 reste un fold de développement déjà inspecté, sans validation
indépendante supplémentaire revendiquée.

Reproduction après extraction vérifiée des deux audits et deux modèles :

```sh
PYTHONPATH=.:src python scripts/summarize_v273_performance_causes.py \
  --audits /path/to/audits --models /path/to/models --output /path/to/diagnosis.json
```
