# Comptage direct avec/sans décroissance — résultat

**La décroissance gagne 2,70 points polyphoniques dans ce compteur direct,
avec une baisse globale et davantage de fausses polyphonies. V27.3 reste
la référence.**

[Exécution terminée](https://github.com/Andriamarosoa/note/actions/runs/35072220720),
commit `48c159b367766eb623d607bf1e1cd9d04ce288f1`, le 16 septembre 2026.
Le [protocole initial](https://github.com/Andriamarosoa/note/blob/48c159b367766eb623d607bf1e1cd9d04ce288f1/analysis/decay-direct-count-protocol.md)
a été publié avant ces nouveaux scores.

## Ce qui a réellement été entraîné

Deux régressions logistiques prédisent directement K=0..6 à partir des
caractéristiques acoustiques, avec ou sans les indices de décroissance.
Aucune prédiction V27.3 ni probabilité V27.2 n'entre dans ces modèles.
Aucun correcteur n'est appliqué après leur sortie. Chaque variante apprend
sur trois compositions et prédit la quatrième, soit huit ajustements réels.
V27.3 est utilisée uniquement pour la comparaison finale.

## Comparaison appariée

Les scores portent sur les mêmes 14 001 lignes de développement, dont
1 776 polyphoniques, réparties en quatre compositions.

| Variante | Corrects polyphoniques | Exact-K polyphonique | Exact-K global | Fausses polyphonies |
|---|---:|---:|---:|---:|
| V27.3 gelée, comparaison seule | 796 / 1 776 | 44,8198 % | 80,3585 % | 851 |
| Compteur direct sans décroissance | 263 / 1 776 | 14,8086 % | 74,0447 % | 542 |
| Compteur direct avec décroissance | 311 / 1 776 | 17,5113 % | 73,8304 % | 617 |

Face au compteur direct témoin, la décroissance donne **48 corrects
polyphoniques supplémentaires**, soit **+2,7027 points**. Le gain apparaît
sur les quatre compositions. En revanche, elle perd **30 corrects globaux**
et ajoute **75 fausses polyphonies** (vrai K<2 prédit K>=2).

Les critères figés échouent donc déjà sur la précision globale et les fausses
polyphonies face au témoin, ainsi que sur la précision face à V27.3.
Le gain polyphonique constaté ne suffit pas à justifier une promotion.

## Vérification et portée

Les quatre tests ont réussi dans l'environnement figé. Les huit ajustements
ont convergé sans modification des réglages ; leurs probabilités ont été
rejouées depuis les coefficients sauvegardés et leurs scores vérifiés depuis
les prédictions enregistrées. La synthèse téléchargée reproduit les comptes
du journal, et ses pourcentages ont été recalculés.

Artefact agrégé `10437076642`, ZIP SHA-256
`6dfce6a2887a9126177d77697b1eba72db023696422f6a1ed3aebbe754a43fb9`.
Le [JSON publié](decay-direct-count-results.json) contient uniquement les
résultats agrégés, sans prédictions individuelles ni coefficients.

Ces compositions avaient déjà été étudiées. Il s'agit d'un diagnostic de
développement sur un compteur léger, avec les caractéristiques existantes,
leur quantification et leurs fenêtres longues dans les graves. La comparaison
des deux compteurs est appariée ; V27.3 possède une architecture et un
historique d'apprentissage différents. Ce résultat laisse ouvertes d'autres
formulations, sans établir de gain pour V27.3 elle-même.

V27.3 conserve son score agrégé officiel de **42,6019 %**, distinct des
44,8198 % de ce sous-ensemble. Le workflow de ce diagnostic passe en
réexécution manuelle. Aucun modèle n'est promu et aucun entraînement
supplémentaire n'est lancé par cette synthèse.
