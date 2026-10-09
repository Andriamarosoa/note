# Série26 — promotion croisée des décisions : B d'abord, confirmé par A d'abord

**Hypothèse préfixée** : commencer par la faible tête B n'améliore pas une sélection si celle-ci reste isolée. En revanche, B→A peut révéler un autre chemin de décision que A→B→A **confirme ensuite** ; seules les décisions proposées dans les deux ordres deviennent candidates. La vérité n'entre dans aucun masque ou marge.

Données déjà observées : S20 (19 poids A-first) et S25 (19 poids B-first), leurs distributions hors morceau sur les mêmes 59 309 fragments natifs ; parents et folds identiques. Pas de nouvel entraînement. L'écart d'identité entre sources sera vérifié bit-à-bit. Scores calculés sur le développement déjà exposé ; **pas de validation indépendante**.

Six structures d'accord entre deux ordres :
1. `p2` : deux sorties de passage 2 d'accord ;
2. `p3` : deux sorties de passage 3 d'accord ;
3. `p4` : deux sorties de passage 4 d'accord ;
4. `p2p4` : les quatre sorties de passage 2 et 4 d'accord ;
5. `r2r4_s4` : B-first passage 2 et 4, A-first passage 4 d'accord ;
6. `r1r2_s2` : B-first passages 1 et 2, A-first passage 2 d'accord.

Trois interprétations, indépendantes de la vérité :
- `two_A_conf` : le minimum des marges candidates-vs-parent dans les deux ordres est positif et au-dessus du seuil ;
- `B0_supports` : même condition et B0 donne à la proposition une compatibilité plus forte que celle de la classe parente ;
- `B0_challenges_parent` : même condition et B0 ne propose **pas** le K du parent (les candidats A peuvent alors réinterpréter cette mauvaise proposition).

Cinq seuils fixés sur la marge des probabilités A : 0.10, 0.20, 0.30, 0.40, 0.50. Total **90 politiques** + parents. Chaque sortie est soit un K effectivement proposé par les deux réseaux, soit la classe initiale S18 ; jamais une classe créée par audit.

Réaliser les métriques global/poly et K0–K6, folds, corrections, régressions et neutralités face à S18 et freeze. Critère d'amélioration sûre de développement : **au moins une nouvelle correction, aucune ancienne correction perdue, poly pas inférieur à S18**. Tout gain sur ce corpus doit ensuite être vérifié sur composition non utilisée dans les choix de conception. Archivage complet même en cas d'échec.