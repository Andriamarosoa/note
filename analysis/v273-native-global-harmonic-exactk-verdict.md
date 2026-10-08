# V27.3 — AUDIT Exact-K global natif énergie / trajectoires harmoniques

**8 octobre 2026 — deux runs terminés, aucun changement de la référence.**

- [Run K0..K6 global, libre de proposer toutes les classes](https://github.com/Andriamarosoa/note/actions/runs/37740493178)
- [Run K0..K6, protection de la polyphonie](https://github.com/Andriamarosoa/note/actions/runs/37740904634)

## Population, référence et protocole

La population complète est **59 309** événements natifs assignés à K0..K6,
folds **0, 1, 2, 4**, **190 enregistrements**; fold 3 et player 05
**exclus**. La référence `freeze_local_combo` provient exactement de
l'artefact reproductible `yourmt3-exactk-cohort` du run 37605163312,
vérifié par SHA256 et métriques fixes.

- Global : **48 454 / 59 309 = 81,6976 %**.
- Polyphonique K2..K6 : **2 530 / 7 385 = 34,2586 %**.
- Couverture du correcteur : **7 493** lignes où la référence a prédit
  K2, K3 ou K4. Critère **strictement fondé sur la prédiction**, jamais
  sur `true_k`. Les **51 816 autres** prédictions sont inchangées.
- Un classifieur logistique `C=.1`, entrées harmoniques et classe originale,
  est réentraîné à chaque fold sur les **autres** folds; le seuil de
  correction est choisi seulement avec des prédictions hors-fold des
  folds d'entraînement. Les nouveaux scores sont calculés sur les
  **59 309** lignes, y compris K0, K1, K5 et K6, et non extrapolés
  depuis le diagnostic K2/K3/K4 équilibré.
- Observation de 80 ms avant à 160 ms après le candidat : ce correcteur
  **n'a pas la même contrainte de causalité** que la référence.

### Partie I : maximiser l'Exact-K global sans préserver la polyphonie

| Variante | Exact-K global | Gain global (pp) | Exact-K poly | Gain poly (pp) | Net |
|---|---:|---:|---:|---:|---:|
| Référence intacte | 81,6976 % | — | **34,2586 %** | — | — |
| Spectre harmonique statique | 81,8122 % | +0,1147 | 29,9391 % | −4,3196 | +68 |
| **Naissance + persistance + extinction** | **82,3956 %** | **+0,6980** | 30,8057 % | −3,4529 | **+414** |
| Toutes les caractéristiques harmoniques | 82,3652 % | +0,6677 | 30,4130 % | −3,8456 | +396 |
| Fondamentales seules | 82,3450 % | +0,6475 | 31,4692 % | −2,7894 | +384 |
| Partiels désaccordés | 82,2792 % | +0,5817 | 29,7901 % | −4,4685 | +345 |
| Trames temporelles brouillées | 82,1764 % | +0,4788 | 30,9005 % | −3,3582 | +284 |

Cycle de vie : **970 corrections / 556 régressions**.
La décomposition nette selon le **vrai K** est :
K0 **+162**, K1 **+507**, K2 **−186**, K3 **−12**,
K4 **−64**, K5 **+7**, K6 **0**. La performance globale s'améliore
par K0 et K1 et **dégrade le cœur du problème polyphonique**.
Ne pas promouvoir cette variante.

### Partie II : protection polyphonique avec seuils appris uniquement en entraînement

Nouvelle règle : seules les prédictions de correction **K2 à K6**
sont admises; aucune proposition K0/K1 n'est appliquée.
Ainsi le gain en nombre de bonnes réponses global et poly est identique
(mais l'amélioration en points de pourcentage diffère à cause des
dénominateurs). Le choix de seuil est fait sur les seuls folds
d'entraînement.

| Représentation | Règle | Exact-K global | Exact-K poly | Gain poly (pp) | Net |
|---|---|---:|---:|---:|---:|
| Référence intacte | — | 81,6976 % | 34,2586 % | — | — |
| Cycle de vie harmonique | 7 classes, corriger uniquement vers K2+ | 81,7599 % | 34,7596 % | +0,5010 | +37 |
| Cycle de vie harmonique | 3 classes K2–K4 | 81,8139 % | 35,1930 % | +0,9343 | +69 |
| Harmoniques complètes | 7 classes, corriger vers K2+ | 81,7886 % | 34,9898 % | +0,7312 | +54 |
| **Fondamentales seules** | **5 classes K2–K6** | **81,8375 %** | **35,3825 %** | **+1,1239** | **+83** |
| **Fondamentales seules** | **3 classes K2–K4** | **81,8375 %** | **35,3825 %** | **+1,1239** | **+83** |
| Brouillage temporel | 5 classes K2–K6 | 81,7296 % | 34,5159 % | +0,2573 | +19 |

La meilleure variante permet **247 corrections / 164 régressions**
et 799 modifications, dont 388 modifications entre deux prédictions
fausses. **Ce gain est expérimental : plusieurs variantes ont été
comparées après examen des résultats des folds internes**. Ce n'est
ni un test neuf ni une justification de changement de référence.

Par fold (gains nets, fondamentale seule, apprentissage K2..K6) :
- Fold 0 : **+31** sur 15 952 événements ; +1,4685 point poly.
- Fold 1 : **+9** sur 13 868 événements ; +0,5108 point poly.
- Fold 2 : **+13** sur 14 665 événements ; +0,7827 point poly.
- Fold 4 : **+30** sur 14 824 événements ; +1,6207 point poly.

Les quatre folds sont positifs. Le correcteur privilégie néanmoins
certaines classes aux dépens d'autres :

| Vrai K | Corrects référence | Corrects, poly-protégé | Différence |
|---|---:|---:|---:|
| K0 | 38 035 | 38 035 | 0 |
| K1 | 7 889 | 7 889 | 0 |
| K2 | 1 183 | 1 300 | **+117** |
| K3 | 915 | 899 | **−16** |
| K4 | 417 | 388 | **−29** |
| K5 | 15 | 26 | **+11** |
| K6 | 0 | 0 | 0 |
| **Global** | **48 454** | **48 537** | **+83** |

Le traitement requalifie de nombreux K3 en K2 :
parmi les changements, **400 sont K3→K2**.
Il augmente donc le rappel K2, mais **aggrave encore K3 et K4**
dans l'absolu. K5 reste très faible et K6 à zéro.

Avec 12 000 rééchantillonnages par **enregistrement** (190 groupes,
graine 27402), les intervalles bootstrap descriptifs
95 % du gain de cette variante sont :
- Exact-K global : **[+0,0649 ; +0,2160] point**.
- Exact-K poly : **[+0,5234 ; +1,7489] point**.

**Ces intervalles ne corrigent pas** la sélection a posteriori entre
plusieurs variantes, l'exposition antérieure des folds, ni le
recoupement entre enregistrements d'un même interprète. Ils ne doivent
pas être présentés comme une preuve indépendante de généralisation.

### Comparaison descriptive YourMT3+

La référence `freeze_local_combo` est à **81,6976 % / 34,2586 %**
(global / poly). Le comparatif YourMT3+ avait donné environ
**86,5434 % / 54,5430 %** sur le protocole publié. Le meilleur
compromis énergétique exploratoire donne
**81,8375 % / 35,3825 %**, donc reste à environ
**4,71 points global** et **19,16 points poly** de YourMT3+.

Attention : YourMT3+ reçoit davantage de contexte audio, et son
préentraînement peut recouper GuitarSet; une équivalence de protocoles
d'entraînement, de causalité et d'indépendance n'est pas démontrée.

## Décision et prochains audits

1. **Aucune promotion.** L'expérience libre fait gagner l'Exact-K global
   mais fait chuter l'Exact-K poly.
2. La variante protégée produit un gain poly **modeste** sur les quatre
   folds, mais seule K2 progresse sensiblement, K3/K4 régressent.
3. L'hypothèse de dynamique énergétique utile est soutenue comme
   **caractéristique diagnostique** ; la séparation réelle des sources
   harmoniques et le flux de Navier–Stokes ne sont **pas démontrés**.
4. Étudier des erreurs K3→K2 et K4→K2/3 par enregistrement, attaques,
   enveloppes et composantes déjà tenues. Ajouter un apprentissage ou
   un décodage de cardinalité qui conserve les voix coexistantes,
   mais **sans** ajustement supplémentaire sur ces mêmes résultats.
5. Tester sur un jeu réellement inédit ou sur de nouvelles compositions
   séparées **avant** toute intégration; ne pas réutiliser player05
   pour régler les seuils.

## Sources reproductibles

- `scripts/extract_v273_harmonic_global.py`
- `scripts/summarize_v273_harmonic_global.py`
- `scripts/summarize_v273_harmonic_global_polyguard.py`
- `.github/workflows/v273-harmonic-global-exactk.yml`
- `.github/workflows/v273-harmonic-global-polyguard.yml`
- Runs GitHub liés ci-dessus, artefacts `v273-harmonic-global-exactk-report`
  et `v273-global-polyguard-audit`, avec prédictions détaillées K0–K6.
