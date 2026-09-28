# Audit acoustique des surcomptages restants sans pondération

Périmètre fixé : modèle natif à 31 trames, sans pondération, fold 3, vrais
K=0 à 3. Les 874 surcomptages proviennent des sorties déjà vérifiées de
l'expérience 36351028493. Aucun modèle réentraîné ni correcteur ajouté.

Les données GuitarSet sont vérifiées contre leurs empreintes historiques.
Pour chaque surcomptage et des contrôles corrects de même piste et même K,
recalculer depuis le PCM original la carte spectrale effectivement reçue
par le réseau et exiger l'égalité exacte du cache float16.

Reproduire indépendamment l'affectation globale des annotations : une
attaque est attribuée au groupe contenant le candidat le plus proche,
à au plus 882 échantillons. En cas d'égalité, le premier groupe gagne.
Compter séparément les attaques qui seraient admissibles au groupe courant
si les groupes concurrents étaient absents. Ce contrefactuel décrit la règle
d'affectation ; il ne remplace jamais les étiquettes pour annoncer un score.

Mesurer le nombre maximal de notes annotées actives dans la fenêtre,
les attaques d'autres groupes, l'énergie absolue avant et après le début,
le niveau du signal brut, sa platitude spectrale et les canaux spectraux natifs.
Comparer les erreurs aux comptes corrects, séparément pour chaque vrai K.
Les familles peuvent se chevaucher ; leurs effectifs ne sont pas additionnés
pour prétendre expliquer toutes les erreurs.

Vérifier spécifiquement les K=0 sans note annotée dans la fenêtre : absence
d'annotation ne signifie ni silence numérique ni absence certaine de signal
musical. Conserver une trace des annotations, distances aux candidats et
propriétaires pour chaque surcomptage, ainsi que deux exemples par K choisis
par confiance décroissante. Ces exemples ne sont pas représentatifs.

Source des courts extraits illustratifs : GuitarSet, enregistrements mono
pickup-mix, https://zenodo.org/records/3371780. Les extraits servent à vérifier
les formes d'onde et le contexte temporel. Aucune reconnaissance certaine
d'un bruit ou d'une harmonique n'est déduite de la seule énergie du signal.

Cet audit est descriptif et vérifie une dépendance de la supervision aux
groupes voisins. Il ne démontre pas à lui seul un mécanisme causal universel
du réseau. Les poids, seuils, sorties et référence V27.3 restent inchangés.
