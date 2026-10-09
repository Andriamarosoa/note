# S43 — audit exhaustif des corrections préservées, par ordre décroissant

**Protocole avant toute exécution.** But : « auditer toutes les boucles qui ont eu des corrections, à commencer par le plus grand ; ces corrections peuvent devenir de nouvelles têtes/sélections ». Ne pas rejeter une famille au seul motif que ses régressions excèdent ses corrections. Mais ne jamais transformer les vérités d'évaluation en condition d'inférence.

## Corpus commun et distinction absolue des références

Pour les 59 309 événements natifs / 7 385 poly (folds 0,1,2,4) :
- `freeze_local_combo` : 48 454 bonnes réponses globales, 2 530 poly.
- `S18` : 49 178 bonnes réponses globales, 2 998 poly. S18 corrige **2 934** décisions de freeze mais régresse sur **2 210**, soit +724 net.
- Classer deux rangs **séparés**, jamais mélanger `corrections vs freeze` et `corrections vs S18`. Les deux utilisent le même vrai K et le même `global_index`.
- `YourMT3+/H9` ne figure dans AUCUNE banque de têtes ni fusion ni source d'apprentissage ; son score externe peut être cité séparément. Aucun oracle de vérité ne peut être présenté comme une politique.

## Sources à auditer sans pertes

1. Répertorier les archives de sorties réelles des séries S9–S41 ayant `predictions.npz`, y compris S36a et S36b distinctement, S33/S39 diagnostics sans sorties nouvelles à marquer « audit only ».
2. Répertorier toutes les archives natives préexistantes dans `analysis/evidence/v273-*/`, notamment `v273-regression-loops/all-candidate-decisions.npz` (catalogue jusqu'à 208 candidats + post-hoc de 3), `v273-audit-memory/variants/all-variant-decisions.npz` (41 variantes issues de 17 archives), `v273-open-k0-k6/series2/` etc. Si l'un des fichiers ne dispose pas de 59 309 prédictions, mêmes IDs et mêmes vrais K, le **noter comme non comparable**, pas l'ignorer silencieusement.
3. Consigner chaque run trouvé, artifact id, statut d'accès (téléchargé, indisponible, schéma incompatible, manque fichier), identité du source et hash de prédictions. L'historique complet du dépôt peut contenir des expériences plus anciennes non archivées ; **ne jamais dire « toutes » sans cette réserve**.
4. Dédupliquer les vecteurs de prédiction IDENTIQUES pour l'analyse des nouvelles têtes, conserver les alias et toutes les corrections/rejets dans les tables.
5. Extraire pour chaque variante et chaque vrai K0–K6, fold 0/1/2/4 : corrections, régressions, changements neutres, K initial → K proposé, volume, ratios observables.
6. Classer d'abord le **nombre brut de corrections positives**, PUIS les corrections poly, PUIS le gain net, sans supprimer les variantes négatives. Une variante qui ne modifie pas S18 peut corriger freeze sans constituer une tête nouvelle au-dessus de S18.
7. Calculer **pour les premières têtes distinctes** les chevauchements exacts de corrections : correction unique à une variante, chevauchements, couverture additionnelle cumulative, union-oracle strictement étiquetée `non déployable`.
8. Proposer un « registre de têtes nouvelles » : famille signal et transition admissible (utilisant UNIQUEMENT K courant/proposé, audio/persistance/attaque, historique d'autres folds), champs `candidate`, `evidence_only`, `not_available`. Les audits test-vrai-K ne sont **jamais** des masques de production.
9. Conserver les sources négatives, même si net extrêmement négatif, et les variations K0/K1 et K2/K3/K4 dans des rapports séparés pour éviter que le global masque la régression poly.
10. Aucun gain n'est validé sur compositions réellement inédites parce que cette cohorte a guidé la recherche S1–S41. Les mesures sont **rétrospectives sur développement**.

Exécution : script `scripts/audit_v273_s43_all_corrections.py` et workflow d'inventaire `.github/workflows/v273-series43-correction-inventory.yml`. L'audit devra sortir `ranked_all_variants.csv`, `best_per_loop.csv`, `head_registry.json`, `correction_overlap.csv`, `source_manifest.json`, `report.md`, `report.json`, `excluded_sources.json`. Il doit être rejoué sur les fichiers sources archivés et ne jamais modifier S18, S35 ou S42.