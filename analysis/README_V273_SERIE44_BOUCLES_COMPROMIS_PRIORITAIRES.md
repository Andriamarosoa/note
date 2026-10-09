# S44 — boucles dédiées aux familles les plus correctrices : compromis avant nouvelle tête

**Préenregistré avant le run.** Fondé sur S43 (910 vecteurs distincts), classer les familles par corrections BRUTES comparables à S18 et mesurer non seulement le gain net, mais combien de corrections K2–K6 restent quand on bloque des régressions.

### Familles dédiées, sans oublier les négatives

Historique S1 acoustique et S2–S4 source/repair, mémoire `repair_coherent`, S25 B-first appris, S24 B-first figé, S40 CNN, S19 récurrent ABA, S11 votes/temps/audio, S38 morphologie, S12 logistique audio, S20 répétition ABA, S27 arrêt anticipé, S41 fusion et S35. Pour chaque famille, retenir le candidat à **maximum corrections brutes vs S18** dans le catalogue S43, et si distinct le candidat possédant le plus de **corrections poly** ; ne pas dédupliquer ni perdre les variantes originales de l'archive S43.

Toutes les variantes sont **réutilisées, sans nouveau fit**, depuis `all_distinct_native_predictions.npz`. Elles sont identifiées par un SHA256 et 59 309 IDs natifs, mêmes vrais K que S43. Refus de toute variante YourMT3/H9, oracle ou labels dans le masque.

**Treize ablations fixes par sélection** : toutes décisions, parent K0–K1, parent K2–K6, parent K2–K4, parent K3–K4, parent K≥3, transitions montantes, descendantes, pas de K>=2→K0/K1, écart max 1, source K2–K4 avec écart max1, source K≥2 et transition montante, destination K2–K6. Tests additionnels de corroboration conditionnelle par proposition K, seulement par **AUTRES familles**, jamais par plusieurs variantes de la même famille : au moins 1 ou 2 familles convergentes (l'unanimité n'est pas supposée). Croiser ces corroborations avec absence de dégradation K2–K6→K0/K1. Tous filtres utilisent uniquement les K proposés, l'ancienne classe K et l'accord observable des autres experts.

Mesurer pour chaque boucle, règle et K initial/vrai K0–K6 : correction, régression, changement neutre, gain net global, gain net poly, corrections exclusives conservées, par fold. Conserver les cas perdus et les propositions refusées ; classer la frontière de Pareto : aucun autre compromis de la même famille ne peut avoir ≥ corrections et ≤ régressions avec une inégalité stricte. Classement par corrections et **non** par seul net, sans prétendre que ces ablations sont de nouvelles architectures entraînées.

**Contrat** : même cohorte déjà explorée. Les seuils structuraux sont testés rétrospectivement ; leur sélection après observation n'est pas une validation indépendante. S18 reste intact, H9 exclue, aucune promotion. L'identité de true K n'est jamais utilisée pour une décision, uniquement pour les audits. Seules les archives disponibles sont couvertes ; les anciennes recherches sans vecteurs restent hors couverture documentée.