# S45 — nouvelle tête de fiabilité **par famille correctrice** (S1...S41)

Fixé avant entraînement. La série S44 a étudié 16 familles, 20 champions bruts/poly et **320** compromis K et consensus sans vrai K dans les règles. Aucune de ces règles n'a un gain positif sans aucune régression. Il faut désormais **apprendre la signature des corrections propres à chaque famille**, plutôt que couper ses propositions de façon identique.

## Modèles

- Chaque famille S44 participe avec **son champion de corrections brutes** gelé, identifié par le SHA256 S43; cela conserve S1, S2–S4, mémoire repair, S24/S25, S40, S19, S11, S38, S12, S20, S27, S41, S35. Séparer les résultats de chacune.
- Pour chacun des **19 morceaux évalués**, entraîne un estimateur de risque propre à cette famille sur les **autres morceaux du même fold**. Fit et normalisation n'ont jamais accès à l'étiquette vraie du morceau évalué.
- Entrées : 139 variables audibles historiques (32 votes, 58 audio, 49 flux), K S18 et K proposé one-hot (14), amplitude de transition, sa direction et nombre d'autres familles corroborantes. L'accord externe est compté une fois par **autre famille**, pas par variantes répétées.
- Deux sorties continues, apprises par Ridge L2 alpha=80 : `Pfix` (S18 était faux et proposition juste), `Pbreak` (S18 était correct et proposition fausse). Ces valeurs sont **scores de risque**, non probabilités calibrées. Préserver le modèle, le scaler et les identifiants de fit.
- Trois compromis préfixés par famille : `benefit = clip(Pfix,0,1)−lambda clip(Pbreak,0,1)`, lambda 1/2/4. Quatre seuils de promotion fixes 0, 0,02, 0,05, 0,10; sinon conserver S18.
- Le sélecteur final comparant **toutes les familles** doit attribuer à un seul groupe la proposition du plus grand bénéfice strictement positif; seuils identiques, et retourner S18 si aucune ne satisfait le seuil. Pas de vérité au classement des propositions par événement.

## Contrôles

- S18 immuable 49178/59309 global, 2998/7385 poly. Audit chaque vraie K0–K6, K initial 0–6, fold, famille, correction/régression/neutre et refus; publier front de Pareto **corrections versus pertes**, en donnant priorité au poly.
- Aucun H9 ou YourMT3+ dans la banque, les entrées, la supervision ou les actions. Les vrais K du morceau évalué n'interviennent dans aucune décision, uniquement dans le rapport ex post.
- **Réserve majeure** : les champions ont été choisis ex post sur le même corpus S43 (test déjà connu), et certains producteurs historiques ont une provenance de fit hors-fold incomplète. Le fit excluant le morceau ne corrige pas ce biais de sélection ni une fuite antérieure potentielle. Chiffres exploratoires, **pas** de validation indépendante, aucune promotion.
- Ne jamais éliminer les correctifs les plus riches simplement parce qu'ils perdent aujourd'hui; conserver toutes les versions négatives et les poids en archive.