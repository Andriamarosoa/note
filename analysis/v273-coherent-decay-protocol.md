# Audit exploratoire : exponentielles et phase des oscillations

L'ancien indice ajuste une droite à `ln(puissance)` sur neuf fenêtres FFT,
puis reconstruit la puissance attendue par `exp`. Des interférences entre
sons continus produisent déjà un indice positif sans nouvelle note.

Une somme d'oscillations amorties est une somme d'exponentielles complexes.
Sa puissance inclut des termes d'interférence dépendant des phases ; elle
n'est généralement pas une seule exponentielle décroissante. Le nouveau
prédicteur ajuste une récurrence linéaire à la forme d'onde passée. Ni les
fréquences connues du générateur, ni les annotations, ni le futur ne lui sont fournis.

Ce travail est un **prototype synthétique de développement**, pas une expérience
Exact K. Des essais exploratoires sur les familles de sons déjà connues ont
servi à choisir les configurations ; les scores ne sont pas une validation indépendante.

| Méthode | Historique passé | Ordre | Pas entre retards |
|---|---:|---:|---:|
| Ancien indice logarithmique | neuf trames de la fenêtre originale | — | — |
| Forme d'onde courte | 1 308 échantillons, 29,66 ms | 16 | 32 |
| Forme d'onde longue, même récurrence | 8 820 échantillons, 200 ms | 16 | 32 |
| Forme d'onde longue, récurrence élargie | 8 820 échantillons, 200 ms | 32 | 64 |

Les trois nouvelles méthodes prédisent 1 764 échantillons, soit 40 ms, sans
réajustement sur le futur. Toutes utilisent la même normalisation RMS sur
les derniers 1 308 échantillons passés. Le modèle long élargi change plusieurs
facteurs ; seul le contrôle intermédiaire isole l'allongement de l'historique.
Le solveur est un moindre carré tronqué, `rcond=1e-7`, sans fréquence oracle.
Les méthodes PCM disposent de la phase, absente des spectres V100. Les unités
des scores diffèrent : seul leur classement est comparé par AUC descriptive.
Aucun seuil de classification ou de comptage n'est sélectionné.

Le banc comprend six familles (sons simples, paires proches, huit harmoniques
et partiels légèrement inharmoniques), seize phases, trois niveaux de bruit
(sans bruit, SNR 40 et 20 dB). Les cas positifs ajoutent une attaque faible,
forte, de même hauteur, ou deux notes simultanées après 5 ms, avec rampe 2 ms.
Les cas négatifs de base sont tenus ou amortis ; les difficultés supplémentaires
comprennent vibrato continu, glissando continu et bruit transitoire sans nouvelle
note musicale. Un transitoire imprévisible peut donc être un faux indice de note.

Les résultats doivent rapporter toutes ces difficultés. Une bonne prévision
sur des oscillateurs ne prouve ni un gain sur GuitarSet, ni un nombre correct
d'attaques, ni leur appartenance au groupe. Aucun modèle entraîné n'est modifié,
aucun nouvel entraînement GuitarSet n'est lancé par cet audit, V27.3 reste la référence.

Référence de principe : Duxbury, C., Bello, J.P., Davies, M. et Sandler, M. (2003),
*Complex domain onset detection for musical signals*, DAFx-03.
https://www.dafx.de/paper-archive/2003/pdfs/dafx81.pdf
Le prototype de récurrence ici n'est pas une reproduction de leur algorithme.
