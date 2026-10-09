# V27.3 — Boucles de fiabilité S30 à S34 (9 octobre 2026)

**Référence S18 strictement inchangée** sur 59 309 événements (dont 7 385 poly K2–K6) : 82,9183 % Exact-K global, 40,5958 % poly, 2 934 corrections historiques / 2 210 régressions face à freeze. Les modèles sont tous en recherche : **aucun** n'est promu, et la cohorte a servi à choisir plusieurs hypothèses. Aucun chiffre ne constitue une évaluation indépendante sur musique totalement inédite.

## Correction comptable importante

La politique S29 λ2/seuil0,02 change **26** décisions : **17** deviennent correctes alors que S18 était erroné, **4** détruisent une bonne réponse S18 et **5** remplacent une réponse fausse par une autre réponse fausse (changement neutre). La précédente formule « 21 changements » n'avait compté que les corrections et régressions. Les 26 lignes sont archivées dans les expériences S30–S34.

## Résultats et tests

| Série | Intervention | Meilleur Exact global | Exact poly associé | Corrections vs S18 | Régressions vs S18 |
|---|---|---:|---:|---:|---:|
| S18 | Référence inchangée | 82,9183 % | 40,5958 % | 0 | 0 |
| S29 | Scheduler d'état A/B→A/STOP | 82,9402 % | 40,5958 % | 17 | 4 |
| S30 | A-first/B-first et modèles de corroboration | 82,9402 % (même S29) | 40,5958 % | 17 | 4 |
| S31 | Signal acoustique complet 335 + logistique pièce-exclue | 82,9402 % | **40,6093 %** | 16 | 3 |
| S32 | 28 intersections/votes de trois scores appris | 82,9385 % (meilleure S32) | 40,5958 % | 15 | 3 |
| S34 | Expert binaire K1/K2 `all335_hgb`, confiance 0,65 | **82,9419 %** | **40,6093 %** | **16** | **2** |

La S34 propose aussi des variantes qui protègent 3/4 erreurs en n'autorisant que 9–12 corrections, mais **aucune des 24** ne combine au moins une nouvelle correction avec zéro régression par rapport à S18.

Références expérimentales et archives :
- S30 run 37865974760, archive https://github.com/Andriamarosoa/note/releases/tag/v273-series30-research-37866187333.
- S31 run 37866124959, archive https://github.com/Andriamarosoa/note/releases/tag/v273-series31-research-37866271628.
- S32 run 37866372090, archive https://github.com/Andriamarosoa/note/releases/tag/v273-series32-research-37866442360.
- S33 audit descriptif run 37866555589, archive https://github.com/Andriamarosoa/note/releases/tag/v273-series33-research-37866870632.
- S34 entraînement run 37866775260 ; archivage permanent effectué via workflow `v273-series34-preserve.yml`. 

## Défaillance de conception mise à nu : distinction K1/K2

L'audit S33 sur la meilleure S29 révèle que **14 des 17 corrections** avaient la transition `S18 K2→S29 K1`, et le vrai K1. Mais **3 des 4 régressions** avaient **la même transition K2→K1**, avec vrai K2, tandis que la quatrième régression allait K1→K2 alors que vrai K1. Les cinq changements neutres comprennent des vrais K0, K3 et K5.

Le gain et l'erreur sont par conséquent largement concentrés dans la **même famille acoustique : déterminer si une seconde note indépendante est réelle ou parasite**. Il serait incorrect de bloquer toutes les transitions K2→K1, puisque cela supprimerait quatorze corrections. Il serait incorrect d'accepter toutes les propositions d'une tête quand A-first et B-first sont d'accord : la S30 a montré que les modèles se trompent souvent ensemble.

### Les quatre événements erronés S29 documentés

| Identifiant natif | Fold | Pièce | Vrai K | K S18 | K S29 | Confiance S31 all335 |
|---:|---:|---|---:|---:|---:|---:|
| 8550 | 2 | Rock1-130-A | 2 | 2 | 1 | 0,9275 |
| 21773 | 2 | Jazz1-200-B | 2 | 2 | 1 | 0,9545 |
| 49626 | 4 | Funk2-108-Eb | 2 | 2 | 1 | 0,0434 |
| 68650 | 4 | Rock1-90-C# | 1 | 1 | 2 | 0,7542 |

Certains cas incorrects restent estimés très fiables par la porte all335 ; **la confiance seule n'est pas une preuve d'identité d'une note**, et ne remplace pas une analyse de l'existence d'une attaque fondamentale indépendamment des harmoniques. Ces scores ne sont pas déclarés calibrés.

## Décision architecturale

Conserver l'architecture **ordonnanceur dynamique par événement** `A` / `B→A` / `STOP`, avec historique de décisions et coûts. La série34 ajoute une **tête d'expertise K1/K2** possible, entraînée sur les vrais cas K1/K2 d'autres morceaux, non sur une exception définie par les quatre erreurs. Cela ne suffit pas encore à atteindre zéro perte, et ne règle pas les classes K3/K4 ou l'écart important à YourMT3+.

Le prochain test scientifiquement prioritaire est un **nouveau jeu de compositions verrouillé avant entraînement/choix de seuil**, idéalement issu de musiciens et enregistrements non présents dans le développement. Les poids des producteurs S18/S20 ont été conçus après observation de cette cohorte ; même un modèle de confiance fit sur d'autres morceaux ne garantit donc pas une validation réellement indépendante. Une fois ce jeu fixé, éprouver la robustesse de la présence d'une **deuxième fondamentale** face aux harmoniques et attaques ; ne jamais introduire les vrais labels de test dans la porte en production.
