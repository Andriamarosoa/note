# V28.0-E — résultat de la comparaison interne

Run terminé avec succès le 9 septembre 2026 :
[`34351021229`](https://github.com/Andriamarosoa/note/actions/runs/34351021229).
Commit : `7473be3c0a6b4dbb5344ef90744f66a872720f69`.

Les deux variantes ont effectué 12 époques complètes depuis zéro, sur les
mêmes 46 921 lignes, puis ont été comparées sur les mêmes 14 001 lignes de
validation interne, dont 1 776 polyphoniques. Chaque variante a reçu 17 604
mises à jour Adam. La sélection de checkpoint suit le protocole préétabli :
exact-K polyphonique maximal, puis NLL polyphonique minimale, puis époque la
plus ancienne.

| Variante | Époque retenue | Poly exact-K | Corrects poly | Global exact-K | NLL poly | Paramètres |
|---|---:|---:|---:|---:|---:|---:|
| Harmonique | 2 | 39,5833 % | 703 / 1 776 | 77,6516 % | 1,61235 | 110 402 |
| Sans agrégation harmonique | 12 | 35,8108 % | 636 / 1 776 | 78,1944 % | 1,64086 | 67 298 |

L'harmonique gagne 67 comptages polyphoniques exacts, soit **+3,7725 points**.
Le contrôle est meilleur en exact-K global. À l'époque 12, l'harmonique est
redescendu à 32,2635 % en polyphonie ; son meilleur état reste celui de
l'époque 2. Il ne faut donc pas confondre le dernier état et le checkpoint
retenu.

La préférence interne est **harmonic**. Le comparateur a relu les prédictions,
recalculé les métriques et vérifié les identités, digests, initialisations
communes, ordres d'entraînement et budgets des deux variantes.

Artefact : `v280-e-internal-comparison`, ID `10123785283`.
SHA-256 ZIP : `67a9c1bbd6d134e6fb26eff5c84bf0b7416c5b6e7248680c3a1053a5219f46f4`.
SHA-256 `comparison.json` :
`7f40e09cd67ba0c49cfead12384603e9069e8c978e294ef260c4550b376d76aa`.

Ce résultat porte sur un seul split et un seul seed. Les variantes n'ont pas
le même nombre de paramètres ; le gain ne sépare donc pas l'effet de
l'agrégation de celui de la capacité supplémentaire. Les compositions ont
déjà servi au développement. Ce résultat n'est pas une validation indépendante
et ne se compare pas directement aux 42,6019 % OOF de V27.3.

V28.0-F poursuit avec l'architecture harmonique, en choisissant à nouveau
l'époque à l'intérieur de chaque fold externe. Aucun poids ni budget d'époque
de V28.0-E n'est réutilisé pour ces réentraînements.
