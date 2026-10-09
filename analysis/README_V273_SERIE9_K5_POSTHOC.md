# V27.3 — Série 9 : veto K5, audit a posteriori

Statut : exploration sur données déjà exposées ; sélection de la règle faite **après** observation des pertes K5 du meilleur sélecteur de série 8. Ce document n'est donc pas une préinscription et ne prétend pas fournir une confirmation indépendante.

Parent poly P : `open_k0k6_series5__raw__tilt1.5__freeze__mix1`. Parent série 8 : `series8__poly_parent__votes__C0.3__cost0.5`. Parent global G : `open_k0k6_series5__prior_corrected__tilt1.5__freeze__mix1`.

Douze veto statiques en fonction des **prédictions accessibles uniquement** (P, G et leurs comparaisons, jamais vrai K) ont été explorés. À chaque veto, remplacer la décision de série 8 par celle du parent P ; ne jamais modifier d'autres lignes.

Le meilleur descriptif de la série, `series9__veto_P_ge5_and_G_le3`, garde P si P ≥ 5 et G ≤ 3. Sur les mêmes 59 309 fragments déjà étudiés :
- Série 8 parent : 49 161 globalement corrects, 2 988 polyphoniquement corrects.
- Série 9 exploratoire : **49 162 corrects globalement**, **2 989 corrects polyphoniquement**.
- Face à série 8 : **2 corrections, 1 régression**, net **+1 global et +1 poly**.
- Face à P : +13 corrections, −8 régressions au global (net +5). 
- Vrais K5 corrects : 50/355 (contre 48/355 en série 8 et 51/355 pour P). K6 inchangé : 2/89.

Ce petit gain est conservé comme nouvelle sélection candidate, sans supprimer les parents ou les 11 variantes moins performantes. La vérification CI doit reproduire les nombres à partir de l'archive issue du run 37851172129, indépendamment du calcul local. Ce rejeu n'apporte pas de nouvelles données. Aucune promotion ; toute généralisation demande une validation sur des morceaux jamais examinés. La cible YourMT3+ reste 51 328 globalement corrects et 4 028 polyphoniquement corrects.

## Rejeu GitHub vérifié

Le run [37851548963](https://github.com/Andriamarosoa/note/actions/runs/37851548963) est terminé avec succès et reproduit exactement 49 162 / 59 309 global, 2 989 / 7 385 poly, deux corrections, une régression et un changement neutre contre le parent S8. Il contrôle tous les 14 vecteurs (12 veto + 2 parents). Les **quatre** décisions changées sont archivées ci-dessous pour que le diagnostic survive à l'expiration des artefacts GitHub Actions.

| ID natif | Fold | Vrai K | S8 | S9 | Effet |
|---:|---:|---:|---:|---:|---|
| 43075 | 0 | 5 | 3 | 5 | correction |
| 4001 | 1 | 5 | 3 | 5 | correction |
| 4134 | 1 | 4 | 3 | 5 | neutre |
| 20299 | 1 | 3 | 3 | 5 | régression |

Le retour vers K5 corrige bien deux vrais K5, mais crée aussi un faux K5 à partir d'un K3 correct. **La protection K5 ne peut donc pas être élargie naïvement.** Aucune causalité acoustique n'est déduite de ces quatre exemples.
