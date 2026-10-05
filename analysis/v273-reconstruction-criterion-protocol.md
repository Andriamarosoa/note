# Audit du critère de reconstruction — protocole

Décidé le 6 octobre 2026 à partir du checkpoint `83dda55`, avant le calcul des
résultats de cette étape. Le pool de 64 candidats contient les trois fréquences
attendues dans 267/272 vrais K3, mais le triplet choisi n'est complet que dans
0 cas avec le gabarit historique et 31 avec la pente `1/h²`.

## Phase A — diagnostic conditionné par les annotations

Population figée : 488 cas des folds internes 0, 1, 2 et 4. Réutiliser les
spectres et les sorties vérifiées de l'audit de capacité ; ne charger ni audio
ni fold 3. Comparer les deux bras pool-64 déjà mesurés, avec saillance
historique et gabarit gaussien de pente 0,5 ou 2.

Pour chaque cas dont le pool couvre toutes les notes attendues :

1. Recalculer le meilleur couple/triplet global et vérifier exactement les
   fréquences et résidus archivés.
2. Parmi les combinaisons du même pool qui couvrent toutes les notes attendues
   par appariement univoque à 55 cents, trouver celle dont le résidu est minimal.
   Comparer son coût au choix global. Ce choix contraint est exclusivement un
   oracle de diagnostic ; il ne devient jamais une entrée d'inférence.
3. Pour chaque composante globale non appariée à une fondamentale attendue,
   tracer plusieurs relations non exclusives : voisine à 55–150 cents,
   harmonique ou sous-harmonique entière d'une note attendue (ordres 2 à 10),
   fondamentale/harmonique/sous-harmonique d'une note étrangère active dans la
   fenêtre post, proximité absolue avec une autre composante, et similarité des
   colonnes du dictionnaire. Une relation fréquentielle ne prouve pas l'origine
   physique d'une composante.
4. Rapporter les distributions par groupe figé, par fold et par nombre de notes
   attendues retrouvées. Les catégories peuvent se chevaucher ; ne pas additionner
   leurs comptes comme une partition causale.
5. Contrôle synthétique : reconstruire un mélange exact des colonnes de la
   meilleure combinaison vraie et vérifier que la recherche globale la couvre.

Cette phase établit où le critère préfère une autre combinaison et l'ampleur du
coût qui sépare les choix. Elle ne mesure aucun gain d'Exact-K.

## Passage éventuel à une correction

N'ajouter une phase B que si la phase A confirme un défaut précis représentable
sans annotations. Écrire alors le contrôle et ses paramètres avant d'en mesurer
les résultats Exact-K. Conserver les mêmes 1 666 lignes, les 845 lignes VAL,
la base, B_low, la petite LR et la sélection exclusivement sur FIT avec
abstention. Rapporter tous les autres vrais K et ne promouvoir aucun bras choisi
sur VAL. Les résultats restent ceux du chemin résiduel normal ; aucune conclusion
sur le chemin compressé n'en découle.

## Phase B — contrôle à pondération logarithmique

Décidé après la phase A et avant son évaluation Exact-K. Sur les 267 K3 dont le
pool est complet, le gabarit historique produit 508 composantes non appariées,
dont 166 sont proches d'une sous-harmonique attendue. La pente `1/h²` réduit ce
compte à 23, mais 157 de ses 400 composantes non appariées sont proches d'un
harmonique attendu. Les catégories se chevauchent et restent diagnostiques.

Le critère actuel additionne des erreurs par bins uniformes en Hz. Tester un
unique changement défini par la mesure uniforme en log-fréquence :

`w(f) = 65 Hz / f`, puis minimiser `||sqrt(w) * (x - D a)||²` sous `a >= 0`.

Ce poids est fixé par le changement de variable `d(log f) = df/f`; son facteur
constant n'a aucun effet sur le choix. Aucun exposant ni seuil n'est ajusté.
Garder le pool-64, la saillance historique, le gabarit gaussien `1/h²`, les dix
harmoniques et tout le reste figés. Le contrôle `w=1` doit reproduire exactement
le bras pool64 / pente 2 déjà archivé.

Recalculer les deux résidus sur les 1 666 lignes audio uniques. Pour chaque fold
externe, comparer `w=1` et `w=65/f` par la même rotation FIT, le même départage
et l'abstention si le meilleur net FIT n'est pas strictement positif. Réajuster
ensuite la LR sur tout FIT et appliquer à ses 845 lignes VAL au total. Archiver
tous les comptes fixes, y compris les actions sur les autres vrais K. La
couverture annotée des 488 cas sert uniquement à comprendre la modification ;
elle ne sélectionne ni le poids ni un modèle.

## Phase C — forme exacte de la fenêtre avec pool élargi

Décidé après le rejet de la phase B et avant l'évaluation de ce bras. Le poids
log-fréquence donne 146 corrections et 151 régressions, net −5 ; il améliore le
nombre de fondamentales choisies dans 19 K3 et le dégrade dans 26. Il est négatif
sur FIT dans les quatre rotations et reste rejeté.

L'audit initial de gabarits a testé la réponse de puissance exacte de la fenêtre
Hann uniquement avec les fréquences annotées injectées. Le pipeline sans
annotation a ensuite varié les pentes avec une gaussienne fixe. Tester maintenant
la combinaison encore manquante : même pool-64 produit par la saillance
historique, même pente `1/h²`, mais colonnes formées par la réponse de puissance
phase-moyennée exacte de la fenêtre Hann de 2 048 échantillons. Le contrôle
gaussien `1/h²` doit reproduire exactement le bras déjà archivé.

Aucun paramètre de forme n'est ajusté. Exécuter le contrôle gaussien et le bras
Hann sur les mêmes 1 666 lignes, puis la même sélection tournante sur FIT avec
abstention. Sauvegarder également les 1 666 spectres vérifiés afin que les audits
de reconstruction suivants ne redécodent plus l'audio. L'évaluation annotée des
488 cas décrit les fréquences choisies mais ne décide pas du bras.
