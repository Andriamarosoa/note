# S496–S501 — Architecture conjointe : Exact-K, naissance de note, hauteur, déduplication

**Statut : expériences validées par GitHub Actions, aucun changement promu dans S80.** Date : 14 octobre 2026. Les résultats sur IDMT et MAESTRO ont un historique d'exposition ; ce sont des évaluations rétrospectives et non de nouvelles validations aveugles.

## Question d'architecture

> À quoi sert un Exact-K précis si plusieurs candidats spectraux correspondent à une seule naissance de note ?

**Réponse technique :** un événement musical canonique doit être une variable distincte d'un candidat acoustique. Les candidats V88/S80 fournissent des preuves, mais ne créent pas chacun une note de sortie. La déduplication associe un groupe de hauteurs à **au plus un** événement natif, ou crée directement **une attaque canonique** par groupe de notes indépendamment détecté. Le comptage K devient une contrainte **souple**, pas une vérité qui annule les vraies attaques. **K_new** (notes qui commencent) doit rester distinct de **K_active** (notes qui sonnent encore).

## S496 : premier graphe conjoint sur la grille native source

[GitHub Actions S496 — SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/38589165797).

Entrées : 506 événements IDMT source S80 original, 575 événements MAESTRO source V88, prédictions de hauteurs Basic Pitch 0.4.0 scellées avant annotation. Le graphe utilise une affectation bipartite *Hungarian* à ≤80 ms, pénalité souple pour source K0 et divergence de K, et **unique propriétaire** pour chaque groupe de naissance acoustique. Trois sorties sont gelées **sans XML/MIDI** :
1. `guarded_sourceK` — ne touche jamais un original K≥1 ; ne ressuscite K0 qu'avec une preuve de note unique et non concurrente suffisamment forte.
2. `joint_Knew_shadow` — autorise une suggestion de révision K lorsque le nombre de notes **naissantes** contredit le K source et que la preuve est suffisamment forte.
3. `joint_Kactive_shadow` — version analogue avec les notes encore **actives** à l'attaque BP+30ms.

Sur cinq vrais WAV guitare, 60 groupes de notes indépendantes, dont 55 associés une seule fois à un candidat natif. Sur trois WAV piano, 234 groupes, dont 232 associés. Les 506/575 candidats ne sont **pas** des attaques musicales annotées.

## S497 : corrections et régressions K de S496

[GitHub Actions S497 — SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/38589539566).

| Cohorte strictement identique | K origine correct | Politique prudente | Joint K_new | Joint K_active |
|---|---:|---:|---:|---:|
| IDMT guitare, vraie chaîne S80, 45 attaques | 25 | 25 | **27** | **28** |
| IDMT guitare, vrais poly 21 | 7 | 7 | 8 | **9** |
| Piano MAESTRO, V88 seulement, 322 attaques | 42 | 46 | **47** | 41 |
| Piano MAESTRO, vrais poly 73 | 7 | 7 | 7 | 7 |

S497 : `joint_Kactive` IDMT **3 corrections, 0 régression**, sur les 45 attaques positives ; `joint_Knew` piano **5 corrections, 0 régression** sur les 322 positives, mais **4 événements non annotés changés**. La politique `joint_Kactive` piano a **1 régression**. Les vraies attaques IDMT mal classées K0 ne sont pas encore récupérées de manière sûre.

Attention : **absence de régression parmi les attaques positives ne signifie pas zéro faux déclenchement dans le flux complet**. Les 459 événements natifs IDMT non appariés à des attaques XML ne sont pas des négatifs confirmés.

## S498 : événement musical canonique, les candidats natifs sont des preuves

[GitHub Actions S498 — SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/38590187399).

Les attaques sont définies comme **un groupe de notes nouveau et unique**, pas comme chaque candidat spectral. On attache à cet événement canonique tous les candidats natifs situés à ±80 ms, sans les cloner comme nouvelles notes. On conserve l'information de K historique comme prior souple dans le choix des K musicaux.

- IDMT guitare : **506** candidats natifs S80 → **60** événements canoniques BP ; **116** relations entre groupes et candidats natifs (61 relations supplémentaires ne produisent pas de note de plus). 5 groupes BP ne trouvent aucun candidat natif proche.
- MAESTRO piano : **575** candidats natifs V88 → **234** groupes de notes canoniques ; 628 relations candidates, sans sortie dupliquée.
- Ces 60/234 groupes ne constituent **pas** des scores de déduplication exacts. L'absence de vérité négative ne permet pas d'affirmer que 446/341 événements sont des « faux » et certaines notes annotées peuvent manquer aux groupes BP.

Un véritable **ré-attaque de la même hauteur** est conservée si elle correspond à un second onset BP (même si la première note résonne encore).

## S499 : Exact-K et noms des notes sur les événements canoniques

[GitHub Actions S499 — SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/38590578138).

Appariement de chaque attaque canonique aux 45 attaques IDMT supportées ou aux 322 attaques MAESTRO, tolérance ±80 ms, un événement vrai par groupe maximum. Groupes sans vérité associée restent **inconnus**, non déclarés faux.

| Même cohorte annotée | K historique | BP notes « birth » | BP notes « active » | Conjoint, K_new | Conjoint, K_active |
|---|---:|---:|---:|---:|---:|
| Guitare IDMT 45, modèle source = S80 complet | 25 | 31 | 32 | 33 | **34** |
| Poly guitare 21 | 7 | 9 | 11 | 10 | **12** |
| Piano MAESTRO 322, modèle source = V88 seul | 42 | 151 | 25 | **146** | 45 |
| Poly piano 73 | 7 | **21** | 19 | 14 | 12 |

Le meilleur K conjoint guitare (active) fait **11 corrections et 2 régressions**, net **+9** sur les 45 vraies attaques ; la meilleure version K nouveau sur piano fait 113 corrections/9 régressions, net +104, mais reste inférieure au K autonome BP 151/322. Les conventions de K sont différentes : guitare IDMT = notes actives à XML+30 ms ; piano MAESTRO = nouvelles notes à onset MIDI.

## S500–S501 : Exact-K dirige la **sélection réelle des hauteurs**

[S500 — SHA original avant labels, SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/38591108937). [S501 — comparaison post-gel, SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/38591422460).

S500 sélectionne les **noms des notes** par ordre de preuves acoustiques indépendantes : `E3`, `G3`, etc., avec un second contrôle expérimental pénalisant légèrement les harmoniques plausibles (+12, +19, +24 demi-tons) lorsqu'un K cible impose un choix. **Si K demande plus de notes qu'il n'existe de preuves acoustiques, le décodeur s'abstient d'en inventer.** On pré-déclare quatre combinaisons : `new/active × confiance/pénalité harmonique`.

S501 observe :

| Notes exactes par attaque / accords poly | BP canonique seul | Décodeur dirigé par K |
|---|---:|---:|
| Guitare IDMT, 45 attaques | 27 | **29** |
| Guitare IDMT, 21 vrais accords poly | 7 | **8** |
| Piano MAESTRO, 322 attaques | 125 | **125** |
| Piano MAESTRO, 73 vrais poly | **18** | 11 |

La meilleure version guitare (active) conserve l'Exact-K canonique **34/45, 12/21 poly** ; sur les noms exacts, elle fait **3 corrections et 1 régression** par rapport aux notes BP brutes, dont 2 corrections et 1 régression poly. Sur piano (K_new), les noms exacts restent **125/322** : 7 corrections et **7 régressions**, dont **7 régressions poly et zéro correction poly**, alors que le comptage passe à 154/322 via suppression de notes. Ce résultat démontre que **K plus précis n'implique pas meilleure identité des notes** et qu'un compteur rigide peut effacer des accords réellement polyphoniques.

Les stratégies « confiance » et « réduction légère des harmoniques » ont produit ici les mêmes nombres agrégés, donc aucune preuve d'un bénéfice de cette pénalisation simple.

## Décision d'architecture / verrous restants

1. **Architecture retenue en R&D :** note birth canonique → graphe des candidats natifs comme preuves → K nouveau et actif distincts → réconciliation souple → sélection de noms des notes avec possibilité d'abstention.
2. **Ne pas promouvoir le décodeur actuellement :** régressions résiduelles sur les attaques annotées, et effets inconnus sur les événements non annotés. La politique de sélection de notes dirigée par K a même supprimé 7 accords poly corrects sur le piano.
3. **Étape de recherche :** entraîner un mécanisme de fiabilité *multi-source* et un veto de suppression de notes poly déjà bien étayées, avec des corpus de développement **distincts** des cinq WAV IDMT et trois MAESTRO déjà analysés. Il faut des véritables annotations de non-naissance de notes pour mesurer les faux événements et la précision des onsets sur flux complet.
4. **Validation futur :** appliquer le **S80 complet** et le système de noms sur MAESTRO, et évaluer GuitarSet/IDMT/autres instruments sous conventions harmonisées. Ne pas présenter les résultats V88 du piano comme des résultats S80 officiels.
5. **Référence officielle conservée :** S80 historique 81,98 % exact-K sur 7 385 événements GuitarSet vrais poly, et 7/21=33,33 % sur les 21 accords IDMT correspondants. Toute nouvelle référence exige démonstration de gain strict non-régressif sur des compositions/instruments jamais vus.

**La valeur réelle d'Exact-K :** il est désormais utilisé comme **contrainte de cohérence du graphe et du décodeur de hauteurs**, sans faire croire qu'il peut résoudre seul l'identité des événements ni garantir leur existence.
