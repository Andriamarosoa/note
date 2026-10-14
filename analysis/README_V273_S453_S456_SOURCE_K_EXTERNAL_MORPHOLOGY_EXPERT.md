# S453–S456 — Diagnostic de l’Exact-K hors GuitarSet et expert acoustique expérimental

**Statut : recherche uniquement. Aucune promotion S80 ni modification Live/Offset.**

## S453 — Résultat vérifié sur le corpus IDMT Dataset3

Exécution : [S453, succès](https://github.com/Andriamarosoa/note/actions/runs/38533183193).

Le projet a restauré les **véritables** poids et décisions V88, S5, S8, S9, S13–S18, S72, S79 et S80. Les identifiants des événements natifs ont été conservés et les annotations XML ont été réappariées indépendamment du score S451 avec la même tolérance pré-déclarée. Les événements sans annotation sont exclus, jamais considérés comme des vrais K0.

Population : 506 événements, dont 47 associés à une annotation, 45 avec notes dans la plage MIDI 40–88, 21 accords réels K2+.

| Étape | K exacts sur vrais K2+ | Sous-comptages | Surcomptages | K0 erronés |
|---|---:|---:|---:|---:|
| V88 freeze_local_combo | 4/21 | 6 | 11 | 3 |
| S5 consensus 3/4 | 6/21 | 8 | 7 | 3 |
| S8 consensus 3/4 | 7/21 | 6 | 8 | 3 |
| S18 consensus 3/4 | 7/21 | 6 | 8 | 3 |
| S72 | 7/21 | 6 | 8 | 3 |
| S80 final original | **7/21** | **6** | **8** | **3** |

Pour les **sept vrais K3**, S80 final prédit : K2 pour un accord, K3 pour trois, K4 pour trois. Les trois surcomptages K3→K4 sont déjà présents à V88; la solution ne peut pas être un correcteur statique K2→K3. Les erreurs de K0 ne doivent pas être confondues avec les 459 événements XML non appariés.

**Diagnostic limité** : le groupe IDMT Dataset3 a été déjà ouvert en S299. La comparaison S453 explique ses erreurs, mais ne constitue pas un protocole aveugle inédit et ne justifie pas de régler un seuil sur les 21 vrais accords pour obtenir un meilleur score.

## S454 — Architecture expérimentale entraînée seulement avec les GuitarSet

Script : `scripts/s454_train_K_source_multiclass_morphology_songheld.py`.

Entrées originales : 58 variables de résumé, 245 variables de flux, 490 de morphologie, 294 de physique, 3 de géométrie ; K prédit S80 injecté en 7 variables one-hot et une variable normalisée : **1098 dimensions**. Aucune annotation musicale ou vrai K n’est fourni à l’inférence.

Quatre modèles supervisés entraînés sur **des compositions différentes de celles du fold de test**, avec **un second découpage interne de compositions** pour sélectionner les seuils d'intervention par K prédit. Les bénéfices doivent être positifs à l'intérieur du train, et aucun recul sur vrais K2, K3 et K4 n'est autorisé dans les validations internes pour ouvrir une sélection. La référence historique S80 reste inchangée ; les propositions K0–K6 sont exportées séparément.

Les scores GuitarSet sont des résultats de développement : l'ancien S80 a un historique d'exposition à GuitarSet. Ne pas utiliser ce rapport pour affirmer une validation indépendante.

## S455 et S456 — Test externe *sans* adaptation à IDMT

- S455 doit vérifier SHA des quatre poids S454, appliquer chaque modèle aux vrais vecteurs acoustiques IDMT S283 avec les **K originaux S297**, et exiger **trois propositions identiques parmi quatre modèles** pour créer un *nouveau candidat K*.
- S455 doit figer les 506 décisions alternatives et conserver séparément les 506 K de S80 **avant** toute lecture d'annotation. Les noms de notes S450 restent inchangés jusqu’à une expérimentation spécifique.
- S456 doit réévaluer strictement les mêmes 45 événements vérifiés par S453 et comparer corrections, régressions et valeurs par vrai K ; les 459 événements non appariés n'auront jamais une classe K0 inventée.
- **Si la balance est nulle, négative ou instable par K : refus d'une promotion.** Même un gain sur IDMT déjà exposé est insuffisant pour changer la référence officielle. Il faut une nouvelle source aveugle et vérifier par composition et instrument.

## Conditions avant « WAV → notes » autonome

La recherche a une chaîne auditable `WAV → V84/V88 → S5/S8/S18 → S45/S46/S69–S80 → K → S430/S446 → noms des notes`, exécutée sur les archives historiques via workflows GitHub ; **elle ne dispose pas encore d'une commande CLI générique unique acceptant un WAV arbitraire**.

Un futur lanceur doit donc orchestrer les producteurs natifs inchangés avec leurs poids historiques, les identifiants d'événements, les bornes temporelles, les sommes de notes et les SHA ; refuser explicitement les poids ou vues manquants, sans substituer une version abrégée à S80. Les tests devront vérifier le K0–K6, les noms E3/G3/B3, les compositions nouvelles, les latences et l’absence de régression avant promotion.
