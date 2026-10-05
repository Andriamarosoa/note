# Point de reprise — Exact-K

## Dernière étape terminée

Audit du classement des candidats et contrôle de capacité, le 5 octobre 2026.
Base de cette étape : commit `aad8dfb83c96cf976fc7a46c3347853b514b58f2`.
Le commit contenant ce fichier est le checkpoint de reprise de cette étape.
Résultats : [README](README-candidate-ranking-audit.md),
[protocole](v273-candidate-ranking-protocol.md),
[preuves](evidence/v273-candidate-ranking/).

- Même cohorte diagnostique de 488 cas. Saillance historique : sur les 816 notes
  des 272 K3, 312 sont représentées dans le pool-8, 499 arrivent après la huitième
  place et cinq sont entièrement supprimées par NMS. Aucune hors-grille.
- Couverture complète K3 : 3/272 avec pool-8, 267/272 avec pool-64. La grille brute
  couvre 272/272. Ce diagnostic ne constitue pas une cause unique du surcomptage.
- Contrôle factoriel pool 8/64 × pente du gabarit 0,5/2, saillance historique
  figée. Nets VAL : −15, −17, −31, +2. 1 666 lignes uniques, 845 lignes VAL
  d'action. Le +2 a 137 corrections et 135 régressions, et régresse dans deux folds.
- Tous les bras sont rejetés par FIT : meilleurs nets −66, −40, −53, −68 selon
  fold externe 0, 1, 2, 4. Abstention partout, net zéro. Aucun modèle promu.
- Malgré pool-64, les triplets ne couvrent les trois K3 que dans 0/272 cas avec
  pente 0,5 et 31/272 avec pente 2. Le critère de reconstruction reste limitant.
- 24 tests locaux réussis. Les deux bras pool-8 reproduisent exactement les
  résidus et comptes précédents. Rejeu des 16 modèles : écart de probabilité zéro.
- Spectres 488 cas, saillances, traces, résidus, fréquences choisies, scores et
  modèles sauvegardés dans `analysis/evidence/v273-candidate-ranking/evidence.zip`.
  Le rejeu CI recalcule les saillances et NMS depuis ces spectres sans audio.

## Prochaine étape autorisée

Auditer le critère de reconstruction quand le pool contient les notes attendues.
Distinguer, par contrôles mesurés, remplacement par harmonique, sous-harmonique,
fréquence voisine ou note étrangère. Les catégories peuvent se chevaucher ; un
rapprochement fréquentiel ne prouve pas une causalité ou l'identité d'une source.
Comparer le coût du triplet choisi à celui contraint aux notes attendues, puis
vérifier si une correction justifiée du critère fonctionne sans annotations.
Les annotations restent diagnostiques, jamais des entrées d'inférence.

Une inspection exploratoire, non utilisée pour sélectionner un modèle, a montré
que les remplacements sont hétérogènes. Ne pas présenter cette inspection comme
une expérience causale ; la prochaine étape doit produire ses propres traces et
contrôles. Ne pas relancer les grilles de pentes ou de capacités déjà rejetées
sans information nouvelle. Aucun nouveau réseau complet à entraîner à ce stade.

## Commits, runs et reprise fiable

- Audit acoustique initial et tri stable : `0bc2c59226a3d87a44e085febd1e4d36b885d4fe`.
  Le run CI `37362400113` est terminé avec succès, état revérifié le 5 octobre.
- Gabarits et neuf pentes : `aad8dfb83c96cf976fc7a46c3347853b514b58f2`.
  Le run CI `37366044960` était encore en attente avant cette publication.
- Exports figés de référence : run réussi `37356100423`, commit `cd5ed339`.
- Vérifier le HEAD et les derniers runs de `codex/v273-failure-clustering` avant
  toute nouvelle modification ; ne pas dupliquer un run en cours. Les calculs
  scientifiques ci-dessus ont été exécutés localement et sont archivés.
- Pour reprendre sans stockage temporaire : extraire l'archive de cette étape,
  les cas de `v273-residual-acoustics/evidence.zip`, puis utiliser le rejeu décrit
  dans le README. Ne pas supposer que les fichiers temporaires existent encore.

## Contraintes persistantes

Folds 0, 1, 2, 4 uniquement ; fold 3 exclu. Base et populations figées. Pas de
nouvel entraînement neuronal complet sans défaut précis et intérêt établi.
Pas de promotion à partir d'un gain supposé, descriptif sur VAL ou d'une régression.
Le projet utilise normal + compressé ; cet audit résiduel du chemin normal ne
prouve aucun effet dans le chemin compressé. Préserver les travaux concurrents,
publier sur la branche de recherche uniquement, sans fusion automatique.
L'automatisation reste utile pour l'audit du critère ; aucune correction Exact-K
sélectionnable ni blocage définitif d'accès ou de données n'a encore été établi.
