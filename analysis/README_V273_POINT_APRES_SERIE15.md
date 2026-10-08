# Note / V27.3 — Point de conservation après la série 15

État mesuré sur la cohorte de développement **59 309 événements**, dont 7 385 poly K2–K6, folds 0/1/2/4 (fold3 exclus). Ces résultats n'ont **pas** encore de confirmation sur de nouvelles compositions et le parent série9 est choisi après exploration de cette cohorte.

## Statistiques appariées à freeze_local_combo

| Candidat | Corrections | Régressions | Net | Exact global | Exact poly |
|---|---:|---:|---:|---:|---:|
| Série9 parent | 2934 | 2226 | +708 | 82,8913 % | 40,4739 % |
| Série13 conservatrice | 2934 | 2223 | +711 | 82,8964 % | 40,4739 % |
| Série14 conservatrice | 2934 | 2222 | +712 | 82,8980 % | 40,4739 % |
| **Série15 OR conservatrice** | **2934** | **2219** | **+715** | **82,9031 %** | **40,4739 %** |
| Série15 OR agressive (contrôle, non promue) | 2850 | 2110 | +740 | 82,9453 % | 40,4739 % |

La série15 conservatrice **réunit 7 corrections distinctes**, sans recouvrement ni régression : 3 de série13 et 4 de série14. Aucun des 2934 gains initiaux n'est perdu sur ce corpus. Les 7 cas sont tous des vrais K0, précédemment prédits K1 : 892, 25912, 36116, 14480, 34704, 44773, 45494. Répartition folds : 0 (2), 1 (1), 4 (4).

## Preuves et fichiers

- Protocole de conservation post-hoc : [README_V273_SERIE15_CONSERVATION_UNION.md](README_V273_SERIE15_CONSERVATION_UNION.md).
- Vérification GitHub : https://github.com/Andriamarosoa/note/actions/runs/37855272619, run réussi. La grille des neuf politiques confirme tous les calculs appariés et les cas nominaux.
- Archive permanente, avec modèles, probabilités de source, 9 vecteurs de décisions et résultats par fold : https://github.com/Andriamarosoa/note/releases/tag/v273-series15-research-37855272619.
- Code du rejeu : [../scripts/verify_v273_series15_union.py](../scripts/verify_v273_series15_union.py).

## Verdict et blocage à lever

**Ne pas promouvoir automatiquement.** Le gain est réel uniquement sur le corpus déjà exposé et très faible. Le meilleur score poly observé dans les séries plus anciennes reste 42,8165 % (série4), supérieur aux 40,4739 % de ce candidat. Benchmark YourMT3+ épinglé : 86,5434 % global, 54,5430 % poly ; aucune variante actuelle ne le dépasse simultanément.

Suite prioritaire : après les 691 régressions natives K0→K1, auditer et proposer des arbitres spécifiques pour freeze K1→K2 (319 régressions), K2→K3 (204), et la conservation réelle des K3/K4, puis confronter les politiques sur **morceaux entièrement inédits** avant toute affirmation de généralisation. Les variantes positives et négatives déjà stockées ne doivent jamais être supprimées.
