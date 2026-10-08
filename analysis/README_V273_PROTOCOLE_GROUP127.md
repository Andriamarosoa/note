# V27.3 — Apprentissage explicite de 127 regroupements complets

**Protocole fixé avant entraînement et avant lecture de ses résultats,
8 octobre 2026. Référence : `freeze_local_combo`, candidat précédent +31.**

L'utilisateur demande une fiabilité conditionnelle au fragment et au groupe
complet. Une sélection risquée seule ne doit pas être éliminée avant de
tester ses associations : plusieurs verdicts faux peuvent devenir justes
ensemble. Ce protocole met en œuvre un premier test mesuré de cette idée.

## Sept candidats explicitement définis

Le code précédent comporte cinq spécialistes entraînés, une action gelée
et des adaptateurs dépendant d'une transition. Il ne fournit pas sept
sélections interchangeables donnant chacune un verdict complet.

Pour construire exactement 127 groupes par fragment, le catalogue fixé est :

| Bit / candidat | Définition |
|---|---|
| 0 / S1 | Classe de `freeze_local_combo`, encodée comme vote d'action |
| 1 / S2 | Distribution du spécialiste spectral |
| 2 / S3 | Distribution du spécialiste naissance/amortissement/persistance |
| 3 / S4 | Distribution du spécialiste harmonique complet |
| 4 / S5 | Distribution du spécialiste des fondamentales |
| 5 / S6 | Distribution du spécialiste sources/cohérence |
| 6 / S7 | Moyenne arithmétique des distributions S2–S6 |

S7 est une combinaison dérivée des mêmes spécialistes, pas un sixième
expert indépendant. Les groupes peuvent donc réutiliser une même source
d'information. Le vote S1 n'affirme pas que la référence est correcte à
100 % : sa justesse est apprise séparément par le critique neuronal.

Les 127 masques non vides contiennent 7/21/35/35/21/7/1 groupes de tailles
1 à 7. S1 est optionnel, ce qui laisse 63 groupes sans S1. KEEP est une
128e option explicite à gain nul. S1 seul et certains autres groupes peuvent
produire le même verdict que KEEP : 127 masques ne signifient pas 127
classes ou verdicts distincts.

Ce catalogue remplace, dans cette expérience seulement, les sélections
binaires KEEP/cible dépendantes des adaptateurs Cxy. Il ne prétend pas
reproduire à l'identique leur ancien inventaire par transition.

## Verdict conjoint et supervision

Le verdict de chaque groupe est défini avant l'entraînement du critique :
moyenne arithmétique des vecteurs de ses membres, puis argmax K2–K6. Une
égalité avec la classe initiale conserve celle-ci. Aucun filtre individuel
ni seuil de risque d'un membre ne limite les groupes.

Cette fusion fixe rend les résultats conjoints observables et stables :

- **correction** : groupe juste, référence fausse ;
- **régression** : groupe faux, référence juste ;
- **neutre** : autre cas, y compris groupe qui conserve la référence.

Un groupe qui évite les erreurs de ses membres en retrouvant la référence
déjà juste est donc une protection, pas une nouvelle correction.

Le réseau reçoit les identités des membres, leurs distributions complètes
masquées, moyenne et dispersion conjointes, taille, entropie et marge, le
contexte acoustique de 43 caractéristiques, la classe initiale et la
destination du groupe. Deux couches partagées non linéaires de largeur
64 puis 32 apprennent la fiabilité de tous les groupes avec les mêmes poids.
Il n'y a pas 127 réseaux indépendants.

**La fusion elle-même n'est pas apprise dans ce premier test.** Les
interactions non linéaires sont apprises par le critique de fiabilité ;
les exemples de complémentarité peuvent déjà exister dans les verdicts
moyennés. Comparer d'autres fusions demanderait une expérience distincte.

## Un risque de référence commun

Le réseau apprend une seule probabilité `r=P(référence juste|fragment)`.
Pour chaque groupe qui propose un changement, il apprend
`q_S=P(verdict de S juste|référence fausse, fragment, groupe S)`.

```text
P(correction_S) = (1-r) q_S
P(régression_S) = r
P(neutre_S)     = (1-r) (1-q_S)
gain_S         = P(correction_S) - P(régression_S)
```

Un groupe sans changement a analytiquement les probabilités `[0,0,1]` et
un gain nul. Le risque commun reçoit le contexte, les sept votes, la classe
initiale et un résumé des audits des groupes changeants. Cela évite cinq
ou 127 probabilités contradictoires de justesse de la même référence.

La perte est la BCE non pondérée de justesse de la référence, plus la BCE
conditionnelle des verdicts conjoints lorsque la référence est fausse.
Les termes des groupes changeants sont **moyennés à l'intérieur de chaque
événement**, puis entre événements. Les 127 groupes ne sont pas traités
comme 127 observations indépendantes. Pour un événement avec changement,
la perte correspond à la vraisemblance factorisée de ses issues conjointes.

Le groupe de gain maximal strictement positif est retenu ; sinon KEEP.
Les égalités de gain avec KEEP conservent la référence. Les sorties sont
normalisées, mais leur calibration empirique après maximisation sur les
groupes reste à mesurer, sans garantie préalable.

## Similarité aux corrections et régressions du même groupe

Les modèles producteurs ne prédisent jamais leurs propres événements
d'entraînement. Un audit destiné à un fold récepteur R exclut R, le test
externe, et le fold prédit par chacun de ses producteurs de référence.

Le scaler de voisinage est ajusté uniquement sur les références autorisées.
Le voisinage contient les 64 fragments les plus proches dans les 43
observables standardisées et tronquées à ±6, parmi la même classe initiale.
Le nombre est réduit si moins de 64 références sont disponibles.

Pour chaque groupe S, seuls les voisins où **ce même S produit la même
destination** contribuent à ses comptes correction/régression/neutre.
Le prior global utilise le même groupe, la même classe initiale et la
même destination, avec un pseudo-compte de 1 par issue. Les taux locaux
sont lissés par 12 fois ce prior global. Le support global, le support
local et la distance moyenne du voisinage sont également fournis.

Les audits reposent donc sur les verdicts conjoints réels ; ils ne sont
pas obtenus en additionnant les bilans des membres. Les IDs de fit et les
chemins imbriqués sont exportés. La normalisation du contexte brut reste
un prétraitement sans labels sur le train externe.

## Trois bras, budget identique

| Bras | Supervision des groupes | Historique fourni |
|---|---|---|
| `direct` | Explicite, conjoint | Aucun : les neuf emplacements d'audit sont nuls |
| `global` | Identique | Taux globaux de chaque groupe/destination et support global |
| `local` | Identique | Global + voisinage des fragments, support et distance |

Architecture et nombre de paramètres identiques ; seules les entrées
d'audit autorisées diffèrent. Pour chaque bras : 30 epochs, seed 27402,
Adam 0,002 et lots de 192 événements. Aucune pondération de classes, aucun
nouveau spécialiste, aucun hyperparamètre retouché après résultat.

Les folds sont séparés par enregistrement. Les 59 309 événements natifs
restent alignés ; seuls les 7 493 prédits initialement K2/K3/K4 peuvent
changer. Les scores poly portent sur tous les 7 385 vrais K≥2. Les vrais
K0/K1 participent aux labels de risque, sans devenir des destinations.

Le précédent modèle `coherent` à +31 est comparé via son archive vérifiée,
sans nouvel entraînement. La comparaison à ce modèle change plusieurs
choix simultanément ; les trois nouveaux bras isolent l'apport des audits
dans leur architecture commune.

## Vérifications et arrêt

Les tests couvrent 127 masques, H0 optionnel, complémentarités de deux et
sept candidats, absence de veto individuel, labels de protection/neutres,
audits conjoints, absence d'IDs interdits, invariance aux labels des folds
exclus, risque commun, vraisemblance par événement, gradients et entraînement
court réel. Les entrées sont aussi préparées sur un vrai fold avant lancement.

Les artifacts contiennent les poids, les 127 propositions/probabilités/
gains par événement, les sept votes, audits, masques choisis, bilans K0–K6
et par fold. Les complémentarités sont comptées lorsque le groupe est juste
et tous ses membres seuls sont faux. L'oracle diagnostique inclut KEEP et
n'est jamais utilisé pour choisir les groupes.

Un décodage limité aux sept singletons, avec les mêmes poids, mesure la
différence des décisions disponibles. Il n'est pas présenté comme une
ablation réentraînée. Les nets, régressions K3/K4 et calibration descriptive
du gain choisi sont publiés même si les résultats sont défavorables.

Les folds 0/1/2/4 sont déjà exposés ; aucun player05/fold3 ni jeu inédit ne
sera évalué ici. Résumés acoustiques inchangés jusqu'à +160 ms. Une perte
nette interdit le remplacement ; même un gain reste exploratoire jusqu'à
validation indépendante. `freeze_local_combo` reste la référence.
