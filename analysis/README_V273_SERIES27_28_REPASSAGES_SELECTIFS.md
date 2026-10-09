# V27.3 — Boucles S27/S28 : ordre des têtes et nécessité d'un repassage complet

**Conclusion démontrée pour le prototype à deux têtes A/B de la série 20 : chaque repassage complet n'est pas nécessaire pour reproduire la décision finale Exact-K.** Cette conclusion ne vaut pas automatiquement pour toutes les futures têtes spécialisées (flux, harmoniques, attaque, morphologie), qui n'ont pas été évaluées ici.

## Données et protocole

59 309 événements natifs, dont 7 385 poly K2–K6, folds 0/1/2/4 et 19 morceaux. S27 utilise exactement les **19 poids et scalers S20** ; décisions comparées sur 14 stratégies figées, 4 décodeurs chacune (56 politiques) et parents S18/freeze (58 politiques). S27 reproduit les quatre distributions A originales de S20 à **4,17×10⁻⁷** près (max), ce qui garantit que la base de comparaison n'a pas changé. Il compte les **exécutions échantillon×tête réellement réalisées** en ne calculant que les sous-lots actifs.

| Variante S27 | Appels de A / événement | Appels de B / événement | Appels totaux relatifs | Nombre de K finaux différents de full-4 |
|---|---:|---:|---:|---:|
| `full_4` | 4,000 | 3,000 | 100,00 % | 0 |
| `full_2` | 2,000 | 1,000 | 42,86 % | 87 |
| `full_3` | 3,000 | 2,000 | 71,43 % | 16 |
| `B_initial_only` | 4,000 | 1,000 | 71,43 % | 4 |
| `B_if_K_changed` | 4,000 | 1,019 | **71,70 %** | **0** |
| `B_if_delta_002` | 4,000 | 1,594 | 79,92 % | **0** |
| `B_if_delta_005` | 4,000 | 1,364 | 76,62 % | **0** |
| `B_if_delta_010` | 4,000 | 1,224 | 74,63 % | **0** |
| `early_stable_002` | 2,595 | 1,595 | **59,86 %** | **0** |
| `early_stable_005` | 2,365 | 1,365 | 53,29 % | 1 |
| `early_stable_010` | 2,227 | 1,227 | 49,33 % | 1 |
| `msg_only_parent_candidate` | 4,000 | 3,000 | 100 % | 775 |
| `msg_only_top2` | 4,000 | 3,000 | 100 % | 760 |
| `msg_only_top3` | 4,000 | 3,000 | 100 % | 691 |

La **distribution de probabilité** n'est pas toujours identique entre une variante sélective et `full_4`. La colonne « différents K » concerne seulement l'argmax final, et vaut 0 **sur la cohorte observée**, pas une garantie théorique. Les appels ne sont pas un chronométrage CPU/GPU et ne tiennent pas compte du poids différent des couches de A et B.

**Nuance décisive :** éviter des recalculs de B est largement possible ; en revanche, **effacer six des sept compatibilités transmises par B** (message partiel) modifie souvent les décisions, sans réduire le nombre d'exécutions de B. Il ne faut pas confondre *sélectionner les têtes à réévaluer* et *tronquer leur analyse*.

S27 meilleur gain exploratoire global, `early_stable_010 margin0.6` : **82,9402 % global / 40,3927 % poly**, 63 corrections vs 50 régressions face au parent S18. Aucun candidat ne conserve strictement toutes les anciennes corrections en apportant un gain. Source originale : https://github.com/Andriamarosoa/note/actions/runs/37864168525. Archive intégrale : https://github.com/Andriamarosoa/note/releases/tag/v273-series27-research-37864333179.

## Série28 — ordre ET profondeur sélectionnés par un ordonnanceur appris

S28 évalue huit actions fixées avant le run : rester sur S18, A-first complet (4 A), A-first 2 A, A-first B réutilisée, A-first B conditionnelle, A-first seulement 2 compatibilités B, A-first arrêt sur stabilité, et B-first passage2/3. L'ordonnanceur entraîne séparément **152 modèles Ridge multi-sortie**, avec pour chacun des 19 morceaux les autres morceaux du même fold uniquement. **Ses 97 entrées** sont le signal/votes originaux et la classe du parent S18, jamais la vérité de l'événement évalué, les résultats YourMT3+, ni les sorties de tous les experts (qui demanderaient de les exécuter auparavant).

Il prédit séparément la possibilité de corriger le parent et le risque de casser le parent ; la valeur estimée est `gain=fix−λ·regress`, λ∈{1,2,4}, seuil∈{0,.02,.05,.10}, soit 12 politiques apprises plus S18 et freeze.

| S28 | Global | Poly | Corrigées vs S18 | Anciennes correctes détruites |
|---|---:|---:|---:|---:|
| S18 | 82,9183 % | **40,5958 %** | 0 | 0 |
| λ=1, seuil=0 | **82,9486 %** | 40,4739 % | 47 | 29 |
| λ=1, seuil=0,02 | **82,9419 %** | **40,5958 %** | 16 | 2 |
| λ=2, seuil=0,02 | 82,9385 % | **40,5958 %** | 14 | 2 |

La route de seuil0,02 maintient le nombre de prédictions poly correctes, gagne 14 bonnes réponses nettes mais **sacrifie encore 2 bonnes décisions historiques**. **Aucune politique S28 ne respecte la conservation stricte.** Cette preuve est intéressante : l'ordre/profondeur peuvent être appris, mais la fiabilité des messages reste insuffisante pour une promotion.

Entraînement S28 : https://github.com/Andriamarosoa/note/actions/runs/37864237149 ; second run avec résultats identiques : https://github.com/Andriamarosoa/note/actions/runs/37864268647.

## Décision de conception

Dans un futur réseau multi-têtes, réserver **une mémoire d'état par événement et par tête** (`last_message`, `last_K_distribution`, `last_call`, `benefit_score`, `revisit_needed`). L'ordonnanceur décide *initialement* du meilleur parcours à partir du signal, puis à chaque étape de **repasser une tête, réutiliser son message, ou arrêter**, avec 1–3 chemins concurrents lorsqu'une contradiction acoustique résiste. Cela n'est pas encore le système multi-têtes complet : S27 teste deux têtes A/B, S28 un ordonnanceur statique initial entre huit parcours. La vraie boucle de politique apprise étape par étape devra être **entraînée et validée sur nouveaux morceaux**, avec audit correction/régression avant intégration.

**Référence maintenue : série18**, 49 178/59 309 global corrects (82,9183 %), 2 998/7 385 poly (40,5958 %), 2 934 corrections historiques et 2 210 régressions vs freeze. Le benchmark YourMT3+ n'est pas dépassé. Les résultats de recherche sont obtenus sur une cohorte déjà exploitée pour concevoir d'anciennes corrections ; aucun ne constitue une validation indépendante sur musique inédite.
