# S389 — Diagnostic de couverture K5 : S80 contre YourMT3+

Date : 2026-10-13. Audit **lecture seule** sur les mêmes 59 309 événements GuitarSet, mêmes 7 385 vrais événements polyphoniques, identifiants et K vérifiés identiques.

## Sources gelées

- S80 reconstruit, sorties S272 : [run 38314291222](https://github.com/Andriamarosoa/note/actions/runs/38314291222), artefact `11775693225`.
- YourMT3+ décodeur complet : [run 37605163312](https://github.com/Andriamarosoa/note/actions/runs/37605163312), artefact `11475499916`.
- S80 historique OOF pour diagnostic de couverture uniquement : [run 38187186049](https://github.com/Andriamarosoa/note/actions/runs/38187186049), artefact `11720306504`.

## Résultats réels pour vrai K5 (n = 355)

| Indicateur | S80 S272 | YourMT3+ |
|---|---:|---:|
| Vrai K5 correctement détecté | 99 | 189 |
| Rappel Exact-K5 | 27,8873 % | 53,2394 % |
| Vrai K5 prédit comme K4 | 224 | 115 |
| K5 correctement trouvé par ce système et raté par l'autre | 47 | 137 |

Les deux trouvent K5 sur 52 cas ; les deux le ratent sur 119 cas. Parmi les **137** vrais K5 trouvés uniquement par YourMT3+, S80 répond K4 dans **124** cas.

### Par fold

| Fold | Vrai K5 | S80 correct | YourMT3+ correct |
|---|---:|---:|---:|
| 0 | 115 | 21 | 56 |
| 1 | 64 | 18 | 35 |
| 2 | 90 | 33 | 50 |
| 4 | 86 | 27 | 48 |

### Localisation du défaut

Sur les 355 vrais K5, la prédiction S80 finale est **strictement identique à S72** : aucune action finale K0/K1/K2 ne corrige K5. Sur cette population, S72 choisit K4 pour 224 cas, K5 pour 99, K3 pour 19, K2 pour 12 et K6 pour 1.

L'union des sorties gelées **freeze_local_combo historique, ancien S80 OOF, S72, S79, PR16 policy22 et S80 S272** propose K5 pour seulement **112/355** vrais K5. Cette *couverture oracle descriptive*, qui examine la vérité, est non déployable et reste inférieure aux **189/355** de YourMT3+. Si on exclut l'ancien S80 OOF de cette union, elle atteint **109/355**. Le déficit ne peut donc pas être résolu par un simple réordonnancement de ces décisions ; il faut une nouvelle source de propositions K5 avant S72.

## Portée et garde-fous

- **Interdiction de transformer ce diagnostic en score indépendant** : sur les événements GuitarSet, la nouvelle exécution quatre branches S272/S273 utilise trois modèles non tenus à l'écart de leurs données d'entraînement. Le préentraînement YourMT3+ peut aussi contenir GuitarSet.
- Les mesures sont authentiques et alignées, mais la comparaison est descriptive et la latence entre modèles n'est pas égale.
- **Aucune correction, régression nouvelle ou promotion revendiquée** ; aucun entraînement ni réglage n'a été effectué sur les étiquettes ci-dessus.
- Travail suivant : entraîner *uniquement sur les folds d'entraînement* un nouveau producteur de propositions K5, à partir de caractéristiques acoustiques amont, et geler son modèle avant d'ouvrir un lot de compositions véritablement inédites avec suffisamment de vrais K5 et annotations fiables. S80 reste la référence de travail, les anciennes références servent seulement à l'audit.
