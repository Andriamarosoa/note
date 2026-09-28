# Verdict : prolonger le comptage natif de 8 à 16 époques

**Le budget explique une partie du déficit du modèle uniforme, mais la
prolongation à 16 époques n’apporte aucun gain polyphonique au modèle
pondéré.** Les deux modèles apprennent mieux les exemples vus ; le gain de
validation observé à 12 époques se réduit ensuite. Le résultat intermédiaire
ne permettait donc pas de conclure à une solution générale par davantage
d’époques.

Les deux entraînements et l’audit automatique ont terminé avec succès dans
l’exécution `36379790003`. Les trois archives finales ont été téléchargées,
vérifiées par SHA256 puis par inventaire complet. Le recalcul local des
probabilités reproduit exactement le diagnostic et le rapport publiés par
le calcul automatique.

## Résultat principal : 16 contre 8

Même validation interne : 15 952 groupes, dont 2 111 vrais K≥2. Aucun groupe
du fold externe 3 évalué dans ce test. Composant natif à 31 trames, mêmes
entrées, architecture, objectifs et ordres de lots ; aucun correcteur.

| Objectif | Validation poly à 8 | À 12 | À 16 | Différence 16 − 8 |
|---|---:|---:|---:|---:|
| Sans pondération | 403/2 111 = 19,09 % | 569/2 111 = 26,95 % | 527/2 111 = **24,96 %** | +124 ; **+5,87 points** |
| Avec pondération | 647/2 111 = 30,65 % | 754/2 111 = 35,72 % | 642/2 111 = **30,41 %** | −5 ; **−0,24 point** |

Le gain à 16 du modèle uniforme est positif sur les cinq compositions
internes. Le modèle pondéré progresse sur deux compositions et recule sur
trois par rapport à 8. La variation de −5 comptes au total ne constitue pas
une preuve statistique de dégradation ; elle signifie qu’aucun gain net
n’est obtenu au point principal prévu.

## Cause partielle établie, difficulté restante

L’architecture inchangée permet un gain après reprise : le budget à huit
époques laissait donc de la performance accessible dans cette expérience.
Mais l’apprentissage des exemples vus et le transfert aux morceaux de
validation suivent ensuite des évolutions différentes.

| Objectif | Apprentissage poly : 8 → 16 | Validation poly : 8 → 16 | Écart apprentissage–validation : 8 → 16 |
|---|---:|---:|---:|
| Sans pondération | 29,03 % → 42,53 % | 19,09 % → 24,96 % | 9,94 → 17,56 points |
| Avec pondération | 41,01 % → 54,34 % | 30,65 % → 30,41 % | 10,36 → 23,93 points |

Entre 12 et 16, le score d’apprentissage monte dans les deux variantes,
alors que la validation polyphonique baisse : −42 comptes sans pondération,
−112 avec. Le modèle pondéré recule sur les cinq compositions internes
pendant cet intervalle. Sa perte pondérée en validation augmente de 0,9705
à 1,0452, alors que celle des exemples vus descend de 0,6588 à 0,5935.
**Cette divergence est un signe de surapprentissage sur cette partition.**
Elle ne permet pas, à elle seule, d’identifier la caractéristique acoustique
qui manque ou le détail d’architecture responsable.

## Où se perd le gain entre 12 et 16 ?

| K vrai | Uniforme : différence de comptes exacts | Pondéré : différence de comptes exacts |
|---:|---:|---:|
| 0 | +4 | +115 |
| 1 | +61 | +84 |
| 2 | **−46** | **−47** |
| 3 | **−76** | **−72** |
| 4 | +80 | +35 |
| 5 | 0 | −28 |
| 6 | 0 | 0 |

Les régressions constatées à 12 ne restent donc pas identiques à 16. K=1
et K=4 récupèrent des comptes, mais K=2 et K=3 perdent une partie de leurs
gains. Les changements sont vérifiés groupe par groupe :

- Sans pondération, parmi les 102 cas K=2 corrects à 12 devenus incorrects
  à 16, 74 sont maintenant prédits à 1. Parmi les 95 cas K=3 dégradés,
  64 sont prédits à 4 et 30 à 2.
- Avec pondération, parmi les 115 cas K=2 dégradés, 94 sont maintenant
  prédits à 1. Parmi les 94 cas K=3 dégradés, 62 sont prédits à 2 et 19 à 4.

Ces nombres décrivent les anciennes bonnes réponses perdues ; ils ne
doivent pas être confondus avec les soldes nets du tableau, qui incluent
aussi les erreurs corrigées.

Le score global peut masquer ce problème. Entre 12 et 16, il monte de
81,34 % à 81,48 % sans pondération et de 80,44 % à 80,98 % avec pondération,
pendant que le score polyphonique baisse. Dans le modèle pondéré, les
199 comptes exacts gagnés sur K=0/1 compensent globalement les 112 perdus
sur K≥2. Une amélioration globale ne démontre donc pas un meilleur comptage
des accords.

## Décision et suite justifiée

La prolongation uniforme démontre un gain partiel ; la prolongation
pondérée à 16 n’est pas une solution au faible score de validation. Aucun
checkpoint n’est promu et aucune nouvelle mesure externe n’est lancée.
Le point 12 reste le meilleur des trois points observés pour le score
polyphonique interne, sans être proclamé meilleur budget universel.

La suite doit viser les confusions natives 1↔2 et 2↔3↔4 et leur stabilité
sur des morceaux non vus. Un prochain test devra modifier un seul facteur
de représentation, de supervision ou de régularisation, conserver son
témoin et rapporter les courbes par K. Augmenter encore le nombre d’époques
seul n’est pas justifié comme correction générale par ces résultats.

## Portée et preuves

Une seule graine de reprise et une seule partition interne. Les variables
d’Adam et ses 2 712 mises à jour initiales ont été restaurées exactement ;
le compteur atteint 5 424 au point 16. Le dropout redémarre avec la graine
déclarée : il ne s’agit pas d’un entraînement ininterrompu reproduit bit à
bit. La comparaison principale 16 contre 8 était fixée avant les résultats.

Ces résultats concernent le composant natif sur sa validation interne.
**La référence officielle V27.3 à 42,6019 % reste inchangée.** Aucun score
externe nouveau ni résultat de toute la chaîne V27.3 n’est revendiqué.

- [Exécution et audit réussis](https://github.com/Andriamarosoa/note/actions/runs/36379790003).
- [Archives : checkpoints, probabilités et rapport automatique](https://github.com/Andriamarosoa/note/releases/tag/v273-training-budget-36379790003).
- [Diagnostic complet et audit des régressions](v273-training-budget-final-diagnosis.json).
- [Provenance et contrôles](v273-training-budget-final-sources.json).
- [Protocole initial](v273-training-budget-protocol.md).

Après extraction vérifiée des archives dans les dossiers `budget-uniform`
et `budget-weighted`, le rapport principal est reproductible avec :

```sh
PYTHONPATH=.:src python -B scripts/summarize_v273_training_budget.py \
  --root /path/to/results --config analysis/v273-native-paired-config.json \
  --output /path/to/recomputed-verdict
```
