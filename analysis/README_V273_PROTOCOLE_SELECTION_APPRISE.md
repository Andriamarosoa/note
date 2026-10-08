# V27.3 — une correction apprise devient une sélection

Protocole fixé avant la première exécution de cette expérience. Parent :
`566ba56ef14985661d96b8c28f8a52e58683eafa`, audit du consensus catégoriel.

L'audit identifiait 1 313 erreurs polyphoniques sans bon verdict dans le
catalogue existant. Ce nombre décrit ce catalogue ; il ne constitue pas une
limite définitive du système. Une règle de correction apprise peut devenir
une nouvelle sélection si elle produit un verdict sur un fragment inédit
sans connaître son vrai K. Sa fiabilité et son utilité restent à mesurer.

## Expérience fixée

Deux bras, une seule exécution, sans recherche de seed, seuil ou architecture :

| Bras | Sélections | Groupes complets | Sélecteur |
|---|---:|---:|---:|
| control7 | catalogue précédent | 127 | 12 994 paramètres |
| corrector8 | même catalogue + S8 apprise | 255 | 13 538 paramètres |

S8 reçoit les 43 observables acoustiques existants et le K initial encodé
sur 7 classes. Réseau `50 → 64 GELU → 32 GELU → 7 logits`, 5 575 paramètres.
Il apprend une classification de K sur **tous** les exemples autorisés,
incluant les succès du verdict initial et ses erreurs. Il ne reçoit ni
l'indicateur « erreur », ni le vrai K du fragment à prédire, ni une sélection
de cas réussis construite après examen du fold évalué.

Entraînement : CE catégorielle sans poids de classes, Adam 0,002, batch 192,
30 époques, seed 27403. Standardisation apprise uniquement sur les lignes
d'entraînement du correcteur, puis bornage à ±6 écarts-types. Les poids et
les paramètres de standardisation sont conservés pour l'inférence.

Le catalogue conserve ses actions K2 à K6. S8 prédit une distribution sur
K0 à K6 ; la masse K0/K1 est transférée au K initial au moment de produire
son vote K2–K6. Cette projection conservatrice garde une somme de 1 et
n'ajoute pas de nouvelles actions. Elle ne signifie pas que le verdict
initial est juste quand la vérité est K0/K1. Cette restriction rend l'essai
comparable et laisse les 2 041 erreurs éligibles de vrai K0/K1 hors couverture.

Toutes les combinaisons non vides sont produites par moyenne des vecteurs
de votes, avec KEEP en cas d'égalité avec le vote initial. H0 reste facultatif.
**Aucune sélection n'est écartée parce qu'elle échoue seule.** Les 127 groupes
originaux, leurs votes et leurs historiques doivent être conservés exactement.
S7 reste la moyenne dérivée des cinq spécialistes, pas une source indépendante.

Le sélecteur reprend le consensus catégoriel : même architecture cachée,
seed 27402, 30 époques, Adam 0,002, batch 192, BCE de justesse du verdict
initial + une CE conditionnelle par événement initialement faux. Les logits
des groupes proposant le même K sont moyennés ; OTHER représente l'absence
d'une bonne alternative disponible. Chaque groupe complet a ses labels
correction/régression/neutre, jamais une somme de labels de ses membres.
Seuls les historiques globaux sont activés, comme dans le comparateur.

L'ajout de S8 augmente de 544 les paramètres du sélecteur et modifie ses
entrées. La comparaison mesure le système étendu ; ce n'est pas une ablation
causale à capacité identique. Les 14 correcteurs sont des instances de
validation croisée, pas 14 sélections ajoutées au catalogue.

## Indépendance des producteurs

Folds 0/1/2/4 seulement ; séparation par enregistrement. Fold3/player05
exclus. Cohorte native : 59 309 événements, dont 7 493 à K initial 2/3/4.
Les autres prédictions restent figées. Les caractéristiques conservent
leur contexte sonore futur de +160 ms.

Les cinq spécialistes réutilisent le cache vérifié du run 37777844182.
S8 dispose d'un cache commun à partir des 14 sous-ensembles non vides de
1, 2 ou 3 folds. Le cache ne contient des probabilités que **hors** des folds
d'entraînement de chaque instance ; toutes les autres lignes sont NaN.

- Pour évaluer un fold externe, spécialistes et S8 sont entraînés sur les
  trois autres folds.
- Pour former le sélecteur, leurs votes sont produits hors fold récepteur,
  avec les deux folds d'entraînement restants.
- Pour les historiques d'un récepteur d'entraînement, les références et
  leurs producteurs excluent à la fois ce récepteur et le fold externe.
  Les producteurs de ces références sont alors entraînés sur un seul fold.
- Chaque chemin enregistre les folds, les identifiants d'entraînement et
  leur SHA-256. Les tests modifient les labels exclus et vérifient que les
  entrées du récepteur restent identiques.

Le contrôle à sept sélections est réentraîné avec le code généralisé. Ses
entrées doivent être identiques aux archives. L'identité des prédictions et
l'écart de scores avec le consensus archivé seront publiés, sans imposer
artificiellement une identité numérique entre machines.

## Mesures annoncées avant résultats

1. Corrections, régressions et net sur la cohorte native, par vrai K et par
   fold, face au verdict figé, au consensus archivé et au contrôle réentraîné.
2. Couverture : erreurs dont aucun ancien groupe n'avait le bon K et qu'un
   groupe contenant S8 rend accessibles. Mesurer combien sont effectivement
   corrigées. La couverture oracle utilise les labels uniquement pour l'audit.
3. S8 seule : verdict direct et bilan correction/régression, sans en faire
   un veto préalable à ses combinaisons.
4. Parmi les nouveaux cas accessibles : S8 juste seule, ou réussite obtenue
   uniquement par une combinaison ; récupération réelle dans les deux cas.
5. Diagnostics de décodage avec mêmes poids et toutes les entrées conservées :
   anciens groupes seuls ; anciens groupes + singleton S8 ; tous les singletons ;
   singleton S8 soumis au sélecteur. Ce ne sont pas des réseaux réentraînés et
   ils n'isolent pas causalement l'information de S8.
6. Gain annoncé contre gain observé ; cohérence des probabilités, égalité
   des scores pour un même K, provenance et empreintes des artefacts.

Les seuils et hyperparamètres ne seront pas adaptés aux résultats de cet
essai. Une sélection médiocre seule mais utile en combinaison sera décrite
comme telle. Une couverture accrue sans gain de sélection sera également
signalée : inventer de bonnes propositions et savoir les choisir sont deux
propriétés distinctes.

Ce travail crée **une** sélection supplémentaire suivant un protocole fixé,
pas une boucle de création automatique illimitée. Les folds sont déjà exposés
aux analyses : succès éventuel = indice exploratoire, pas preuve de
généralisation ni autorisation de promotion. Une validation sur données
réservées reste nécessaire après le développement.
