# Stabilité temporelle — protocole figé

Préenregistré le 6 octobre 2026 à partir de `d9f75cb23e14204615bf5a374629ba1f06c720f9`,
avant le calcul des nouvelles fenêtres. Chemin audio **normal uniquement**.

## Sources et population

Conserver les exports du run `37356100423`, les folds internes 0, 1, 2 et 4,
les 1 666 lignes uniques / 122 enregistrements, les 845 lignes VAL d'action,
la base neuronale, B_low et tous les vrais K. Ne charger ni audio, ni annotation,
ni export du fold 3. Vérifier les archives, les manifests et l'affectation des
enregistrements avant décodage. Aucun nouvel entraînement neuronal.

## Fenêtres et extraction sans annotations

À partir du début de candidat figé `s`, à 44 100 Hz :

- pré : `[s-2048, s)` ; première fenêtre : `[s, s+2048)` ;
- seconde fenêtre : `[s+2048, s+4096)`.

Même Hann 2 048 points, FFT 8 192, bande 65–6 000 Hz. Pour chaque fenêtre
postérieure `Qt`, calculer `Xt = max(Qt-P, 0) / (sum(max(Qt-P,0))+1e-12)`.
La référence pré `P` reste commune : ne pas soustraire la première fenêtre à
la seconde. Le padding historique par zéro reste inchangé ; le tracer sans
exclure de ligne. La seconde fenêtre ajoute 2 048/44 100 = **46,44 ms** de
contexte futur, soit **92,88 ms** après `s` au total.

Rejouer `X1` contre les 1 666 spectres archivés. La saillance historique,
le NMS stable et le pool-64 sont produits uniquement depuis `X1` et réutilisés
dans les deux fenêtres. Gabarit gaussien `1/h²`, dix harmoniques, grille et
solveur NNLS figés. Aucune fréquence annotée n'entre dans l'extraction.

## Phase A : stabilité et contrôle de reconstruction

Pour chaque couple/triplet du même pool, calculer les deux résidus NNLS
normalisés séparément (amplitudes non négatives libres dans chaque fenêtre).
Comparer les choix libres de chaque fenêtre à 55 cents, univoquement. Tracer
également le choix minimisant la moyenne des deux résidus. Le sous-ensemble
de fréquences est commun, mais les amplitudes peuvent changer : cette moyenne
ne garantit pas que toutes les composantes restent actives dans chaque fenêtre.

Contrôles : fenêtre répétée => mêmes choix/coûts ; solveur exhaustif de référence
=> mêmes minima ; mélange synthétique de trois colonnes à amplitudes variables
=> triplet retrouvé. Un contrôle synthétique avec composante transitoire
documente ce que le critère fait et ne fait pas.

Après l'extraction, utiliser les seules 488 annotations de la cohorte figée
pour compter les composantes attendues/non appariées qui persistent, et la
couverture des couples/triplets. Décrire les notes attendues qui s'éteignent et
les notes étrangères qui apparaissent dans la seconde fenêtre. Ces annotations
restent diagnostiques ; pas de filtrage de population ni de sélection sur ces
sous-groupes. Un changement entre fenêtres ne prouve pas la source physique.

## Phase B : unique bras temporel prévu

Si la phase A confirme que les choix changent entre fenêtres, tester le bras
conjoint ci-dessus contre le contrôle première-fenêtre. Chaque bras garde deux
features : résidu du meilleur couple et résidu du meilleur triplet. Aucune
grille de durées, pondérations ou seuils. La petite LR reste identique, entraînée
sur les K2/K3 de FIT ; le comptage inclut tous les vrais K.

Sélection par rotation des seuls folds de FIT, départage historique (net,
régressions, actions, ordre contrôle puis conjoint), abstention si net <= 0.
Archiver les modèles internes et finaux, scores, fréquences et résidus.
VAL décrit chaque bras fixe et la politique réellement sélectionnée. Le contrôle
première-fenêtre doit reproduire les 137 corrections / 135 régressions antérieures
et les scores figés. Aucun résultat VAL ne choisit un paramètre.

Ne promouvoir aucune variante sur un gain descriptif ou une régression. Ces
folds ont déjà été inspectés ; ce travail n'est pas une validation finale intacte.
Ne tirer aucune conclusion pour le chemin compressé. Publier même un résultat
négatif, avec un rejeu sans audio ni réajustement et une décision documentée.
