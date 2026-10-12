# S132 — Origine S4/S5 : recherche du constructeur supprimé dans les commits historiques

## Périmètre

Exécution réussie : [S132 — run 38189691837](https://github.com/Andriamarosoa/note/actions/runs/38189691837).

L'analyse ne se limite **pas** aux pointes des branches : elle remonte les
**1 182 commits Git accessibles avant le 8 octobre 2026 à 21 h 10 UTC**
et examine les lignes ajoutées/supprimées dans `scripts/`, `src/`,
`.github/workflows/` et les notebooks pour les signatures
`owned/S4`, `416/527` et des opérations de flux temporel.

- **0 commit** avec ajout/suppression de `416` ou `527` dans les sources
  ciblées avant le checkpoint.
- **12 commits** contenant des modifications `S4/owned` au sens large,
  liés à d'autres travaux de propriétés acoustiques, pas au
  constructeur `owned527` original certifié.
- **10 commits** avec des termes de dynamique temporelle communs, sans
  producteur original S5 démontré.
- Aucun fichier `scripts/` ou workflow dans le
  [commit historique 7165e1e](https://github.com/Andriamarosoa/note/commit/7165e1ef54510aa9effc9f597b2783c832ca907a),
  qui conserve les poids et les prédictions des séries 4/5.
- Aucune fonction d'entraînement historique S4/S5 explicitement identifiée
  dans la liste des scripts de cette révision.

Le protocole décrit **111D** (`summary58 + geometry46 + freeze7`),
puis **416D** supplémentaires dont les familles statistiques sont décrites
textuellement mais sans algorithme numériquement univoque.
Les quatre modèles S5 attendent 527 valeurs dans l'ordre d'entraînement.

### Preuve indépendante et limites

Le workflow [37837235723](https://github.com/Andriamarosoa/note/actions/runs/37837235723)
extrait les vrais audios, entraîne les premiers modèles neuronaux du corpus
puis sauvegarde les artefacts d'ouverture de K0–K6 ; il **n'entraîne pas**
directement S4 `owned527` ou S5. Le seul commit allant du protocole S5
[`8a9410a`](https://github.com/Andriamarosoa/note/commit/8a9410a4232c653615eb32db401410202831377f)
aux poids
[`7165e1e`](https://github.com/Andriamarosoa/note/commit/7165e1ef54510aa9effc9f597b2783c832ca907a)
conserve les fichiers entraînés, **aucun générateur d'entrée**.

S110 avait déjà audité les archives S5 : aucune matrice 527D authentique
sauvegardée et identifiable. Les sorties historiques sont présentes,
mais ne contiennent pas assez d'information pour inverser le calcul
des 416 colonnes source.

Le scan `git log -G` ne détecte ni code local non commité, ni objet Git
devenu inaccessible, ni source de fichier dont aucun changement ne porte
les signatures. Il reste donc possible que l'ancien constructeur soit
présent dans un ancien répertoire de travail, un notebook non publié, un
artefact externe, une sauvegarde ou un cache privé. Ce constat
n'autorise pas à dire que le script n'a jamais existé.

### Chemin de reprise strict

1. Chercher un script/notebook local datant du 8 octobre 2026 calculant
   `owned`, les 416 statistiques, ou manipulant les modèles S4/S5 de
   type `HistGradientBoostingClassifier` et le SHA de design
   `fd4d4d32872e5d3856bf994db386e9f2976ce57a418a21b56b876bf6f6d349cd`.
2. Restaurer ce code sous un chemin archivé et SHA-vérifié sans le mélanger
   avec les hypothèses S118/S123/S125.
3. Recalculer les 527 caractéristiques sur les 15 952 événements du fold 0,
   vérifier **l'identité échantillon-par-échantillon des 7 probabilités**
   avec les sorties originales S5 (et non simplement le taux d'accord).
4. Rejouer les folds 1/2/4 puis GOAT sans MIDI et reconnecter les 16
   familles originales S45 à S46→S69→S80 ; seulement alors appliquer
   le protocole indépendant de promotion.

Une nouvelle architecture entraînée sur les mêmes audios devrait porter
un autre nom (ex. `S80-deploy-v1`) et être validée de zéro, sans réutiliser
l'ancien pourcentage de S80.

**Verdict : source exacte non retrouvée dans le Git audité ; S80 non promu.**
