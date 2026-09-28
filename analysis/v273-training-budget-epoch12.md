# Budget d’entraînement : point intermédiaire vérifié à 12 époques

**Les deux objectifs progressent entre 8 et 12 époques sur le comptage polyphonique interne.** Le gain est visible sur les données vues et sur la validation. Le résultat principal, fixé à 16 époques, reste en cours ; ce point intermédiaire ne le remplace pas.

| Objectif | Apprentissage : 8 → 12 | Validation : 8 → 12 | Gain validation |
|---|---:|---:|---:|
| uniform | 29.03 % → 39.70 % | 19.09 % → 26.95 % | +166 / 2111 (+7.86 points) |
| weighted | 41.01 % → 52.05 % | 30.65 % → 35.72 % | +107 / 2111 (+5.07 points) |

La validation compte 2 111 groupes polyphoniques ; l’apprentissage en compte 5 274. Les 15 952 groupes de validation au total restent les mêmes. Le fold externe 3 n’est pas évalué.

## Ce que cela dit sur la cause

À architecture, données et objectif constants, reprendre les mises à jour améliore le score polyphonique des deux modèles au point 12. Une partie de la faible performance à huit époques peut donc être améliorée par la prolongation prévue, sans modifier l’architecture. Le point à huit époques laissait ce gain accessible dans cette expérience.

Cela ne démontre pas que le budget explique toutes les erreurs, ni que prolonger améliore tous les K. Le dropout repart avec une graine déclarée, tandis que les variables d’Adam et son compteur sont restaurés exactement. Il s’agit d’une seule reprise et d’une seule partition interne.

## Régressions et gains par K

| K vrai | Effectif validation | Uniforme : exacts 8 → 12 | Pondéré : exacts 8 → 12 |
|---:|---:|---:|---:|
| 0 | 10763 | 10374 → 10387 (+13) | 10157 → 10174 (+17) |
| 1 | 3078 | 2171 → 2019 (-152) | 2060 → 1903 (-157) |
| 2 | 993 | 194 → 321 (+127) | 323 → 416 (+93) |
| 3 | 594 | 154 → 221 (+67) | 150 → 233 (+83) |
| 4 | 374 | 55 → 21 (-34) | 174 → 66 (-108) |
| 5 | 115 | 0 → 6 (+6) | 0 → 39 (+39) |
| 6 | 35 | 0 → 0 (+0) | 0 → 0 (+0) |

K=2 et K=3 progressent dans les deux variantes ; K=1 et K=4 régressent. Les gains agrégés ne valent pas validation d’une solution complète. L’écart apprentissage–validation polyphonique s’élargit aussi : de 9,94 à 12,75 points sans pondération et de 10,36 à 16,33 points avec pondération. Les gains sur les exemples vus se transfèrent donc seulement en partie à la validation.

| Objectif | Erreurs poly corrigées | Bonnes décisions poly perdues | Exact K global : 8 → 12 | Compositions internes en progrès |
|---|---:|---:|---:|---:|
| uniform | 256 | 90 | 81.17 % → 81.34 % | 5/5 |
| weighted | 285 | 178 | 80.64 % → 80.44 % | 4/5 |

## Vérification et sources

Les probabilités brutes des deux points sont recalculées en scores, les identités des exemples et des checkpoints sont contrôlées et les prédictions à huit époques sont comparées à l’audit figé précédent. Les 4 068 mises à jour d’Adam au point 12 sont cohérentes avec le protocole. Aucun correcteur ni promotion officielle.

- Calcul : https://github.com/Andriamarosoa/note/actions/runs/36379790003
- Archives et futur verdict à 16 : https://github.com/Andriamarosoa/note/releases/tag/v273-training-budget-36379790003
- Données de ce contrôle : [v273-training-budget-epoch12.json](v273-training-budget-epoch12.json)
- Protocole : [v273-training-budget-protocol.md](v273-training-budget-protocol.md)

Ces scores concernent le composant de comptage natif sur sa validation interne, pas le score officiel V27.3 et pas une nouvelle mesure du fold externe 3.
