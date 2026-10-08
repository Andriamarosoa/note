# Série 23 — Diagnostic de récupération du meilleur producteur poly historique

La référence S18 demeure la seule référence conservatrice promue *pour ce corpus de développement* (pas encore modèle de production) : 82,9183 % global et 40,5958 % poly, corrections 2 934 / régressions 2 210 face à freeze. Le meilleur producteur poly historique série 4 avait 42,8165 % poly. Pour confronter **les mêmes événements** au réseau A→B→A, ses prédictions par ID et la provenance de ses producteurs doivent d'abord être récupérées exactement.

## Actions effectives

- Run 37859970897 : archive d'origine `v273-open-summary.tar` de la release `v273-open-k0-k6-37837235723` téléchargée, inventoriée. Elle ne contient qu'un `predictions.npz` et `report.json` au niveau racine, produits par le workflow d'ouverture initial, **pas** le jeu de décisions consolidées de série4.
- Le dépôt possède 14 fichiers `analysis/evidence/v273-open-k0-k6/series4/models/fold*__{control,owned}__poly{1,2}.joblib`. L'inventaire des fichiers a réussi ; par contre l'ouverture des poids et donc leur vérification/exécution reste bloquée par la compatibilité de sérialisation NumPy `BitGenerator`/PCG64.
- Run 37860148298, Python 3.12/NumPy 1.26.4 : `ValueError: <class 'numpy.random._pcg64.PCG64'> is not a known BitGenerator module`.
- Run 37860209570, Python 3.11/NumPy 1.26.4 : même erreur.
- Run 37860288874, compatibilité de constructeur RNG ajoutée : chargement plus loin mais `TypeError: state must be a dict`.

**Ne pas inventer de sélection série4 à partir des seuls scores agrégés.** Il faut retrouver les fichiers de prédictions originaux par ID ou la version exacte de NumPy/joblib de la sérialisation et contrôler toutes les sorties, puis comparer les corrections/régressions appariées. La récupération n'est pas achevée. Les poids historiques n'ont été ni modifiés ni écrasés.

## Situation des nouvelles boucles récurrentes

Série21 : 96 portes de fiabilité entraînées, **aucune** n'améliore strictement S18 sans sacrifier de corrections ; meilleur global 82,9402 % mais poly 39,6344 %, 181 corrections vs 168 régressions. Rejeu indépendant et archive durables : run 37859537595.

Série22 : 108 règles de confirmation mutuelle A/B et modèles de confiance, **aucune** ne respecte le critère zéro perte ; meilleur global 83,0161 % mais poly 39,3365 %, 255 corrections vs 197 régressions. Run 37859786345. Le parent S18 est intact.

## Suite techniquement fondée

1. Audit de l'environnement exact et/ou récupération des distributions prédictives série4 en format NPZ/JSON plutôt que joblib opaque, avec chaque fold, classe et ID natif certifiés.
2. Comparaison S4/S18 par vrai K, transition et morceau, sans importer de label comme variable.
3. Entraîner une porte de confiance A/B qui choisit entre **deux producteurs réellement différents** plutôt que seulement entre les passages d'un même producteur, sur d'autres morceaux ; audit par fold et zéro correction perdue.
4. Validation sur compositions inédites après gel complet de l'architecture.

Ne pas attribuer à cette enquête un nouveau gain Exact-K : c'est un résultat de provenance **négatif mais reproductible**.
