# S37 — neuf parcours S35, même juge de fiabilité gelé S36b

**Protocole déposé avant le premier run.** Le candidat S36b apporte 2 corrections nettes et 0 régression sur la cohorte de développement, confirmé après rejeu de 38 modèles. Il n'exploitait comme producteurs que S35 λ1/seuil0,05 et S35 λ4/seuil0. La présente contre-épreuve vise à voir si **les sept autres sorties S35**, déjà calculées, apportent des corrections **complémentaires**.

**Gel absolu des poids de juge** : reprendre, sans réentraîner ni recalibrer, les 38 fichiers `piece__audio_flow.joblib` et `piece__full_path.joblib` de S36b. Leurs étiquettes de formation appartiennent aux autres morceaux du même fold. Les 9 parcours S35 ont eux-mêmes les poids et exclusions de folds confirmés en S35. S18 et toutes les probabilités/producteurs restent inchangés. H9 est exclue ; H8 reste masquée.

## Choix dynamique après les chemins

Pour chaque morceau tenu à l'écart du fit de sa porte de risque, les neuf propositions K0–K6 `series35__...` sont comparées sans vérité réelle : calculer `fix_estimé−lambda·regress_estimé` pour chaque proposition différente de S18. Choisir le **maximum** seulement si sa valeur dépasse un seuil. Lorsque plusieurs actions ont la même valeur, conserver l'ordre de catalogue S35 déterministe ; ne pas utiliser l'étiquette réelle pour départager. Sinon conserver S18. Ce choix d'une sortie parmi les neuf parcours est un **arbitre final** ; les parcours pré-calculés ont toujours été évalués en amont : aucune économie de calcul mesurée. Les chemins internes S35 restent dynamiques et leur ordre est préservé.

- Deux vues : `audio_flow` et `full_path`.
- Pénalité λ figée à **1, 2, 4**.
- Seuil de décision figé à **0, 0,005, 0,02, 0,05, 0,10**.
- Total 30 politiques nouvellement évaluées, plus S18, S35, S36b (confirmé).

Auditer toutes les corrections/régressions/neutres, le K réel 0–6, K initial, fold, et le producteur choisi pour chaque événement. Conserver toutes les propositions rejetées et les scores, même en échec. **Critère strict** : nouvelles corrections positives, zéro bonne réponse S18 détruite, poly≥S18. L'expérience utilise des compositions et seuils historiques déjà discutés : même si réussie, ce n'est **pas une validation indépendante** ; aucune promotion automatique sans jeu de compositions totalement nouveau. Ne pas créer de masques propres aux deux ID 57164/37676, ni utiliser YourMT3+ dans l'arbitre.