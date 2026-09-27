# Bilan complet : fenêtre de 23 contre 31 trames, fold 3

Les **quatre entraînements et l'audit complet sont terminés**. Le réseau natif
à 31 trames gagne en Exact K polyphonique dans les deux comparaisons prévues :
**+1,7268 point sans pondération**, **+1,9299 point avec pondération**.
Ce progrès est mesuré sur le fold 3 ; le problème de surcomptage reste présent
et s'aggrave dans la comparaison avec pondération.

## Résultats externes, sur les mêmes groupes

| Pondération | 23 trames : Exact K polyphonique | 31 trames : Exact K polyphonique | Gain net |
|---|---:|---:|---:|
| Uniforme — comparaison principale | 636/1 969 = 32,3007 % | 670/1 969 = 34,0274 % | +34 groupes ; +1,7268 point |
| Pondérée — comparaison secondaire | 728/1 969 = 36,9731 % | 766/1 969 = 38,9030 % | +38 groupes ; +1,9299 point |

La pondération secondaire est l'inverse de la racine de la fréquence des
classes, calculée sur les données d'apprentissage. Chaque ligne compare deux
fenêtres avec les mêmes poids de classes. La comparaison principale et la
comparaison secondaire ont été fixées avant l'observation des scores ; aucune
variante n'est promue à partir du fold externe.

| Bilan des 1 969 groupes polyphoniques | Uniforme, 23 → 31 | Pondérée, 23 → 31 |
|---|---:|---:|
| Erreurs devenues correctes | 122 | 163 |
| Groupes corrects devenus erronés | 88 | 125 |
| Erreurs devenues une autre erreur | 160 | 130 |
| Surcomptages | 288 → 288 | 412 → 460 |
| Sous-comptages | 1 045 → 1 011 | 829 → 743 |
| Erreurs restantes à 31 trames | 1 299 | 1 203 |

**Le gain pondéré ne signifie pas une baisse de toutes les erreurs.** Il
résulte de 86 sous-comptages de moins, avec 48 surcomptages de plus : le bilan
net reste +38 comptes exacts. L'augmentation du surcomptage est auditée et
n'est pas masquée par la hausse du score.

Sur les 15 279 groupes, silences inclus, l'Exact K global passe de 83,2777 %
à 83,5657 % sans pondération (+44 comptes exacts), et de 82,1651 % à
82,1912 % avec pondération (+4 comptes exacts). Dans la paire pondérée, les
fausses détections sur les 10 285 groupes K=0 passent de 457 à 525 ; cela
explique pourquoi le gain global est beaucoup plus faible que le gain
polyphonique. Les surcomptages totaux passent de 883 à 895 sans pondération,
et de 1 329 à 1 465 avec pondération.

## Où se trouvent les gains et les régressions ?

| Vrai K | Groupes | Corrects sans pondération, 23 → 31 | Corrects avec pondération, 23 → 31 |
|---|---:|---:|---:|
| 2 | 806 | 217 → 243 | 324 → 311 |
| 3 | 628 | 297 → 312 | 223 → 216 |
| 4 | 405 | 116 → 105 | 159 → 217 |
| 5 | 109 | 6 → 10 | 22 → 22 |
| 6 | 21 | 0 → 0 | 0 → 0 |

Dans la paire pondérée, le gain provient de K=4 (+58), tandis que K=2 et
K=3 régressent de 13 et 7 groupes. Les surcomptages de K=2 augmentent de 21,
ceux de K=3 de 42 ; ceux de K=4 diminuent de 15, soit +48 au total.
Ces résultats décrivent précisément le déplacement des erreurs. Ils ne
prouvent pas, à eux seuls, un mécanisme acoustique unique ni une cause
universelle du surcomptage.

Parmi les 42 groupes polyphoniques dont une attaque était auparavant hors
fenêtre, les comptes exacts passent de 8 à 9 sans pondération et de 12 à 14
avec pondération. Les gains totaux de 34 et 38 groupes ne peuvent donc pas
être attribués uniquement à la récupération directe de ces attaques. La
couverture audio est corrigée, mais l'expérience mesure aussi l'effet du
contexte supplémentaire pendant tout le réentraînement.

## Validation interne et variabilité

| Validation interne, fold 0 | 23 trames | 31 trames | Écart |
|---|---:|---:|---:|
| Exact K polyphonique uniforme | 399/2 111 = 18,9010 % | 403/2 111 = 19,0905 % | +0,1895 point |
| Exact K polyphonique pondéré | 637/2 111 = 30,1753 % | 647/2 111 = 30,6490 % | +0,4737 point |

Les gains internes sont plus faibles que les gains externes. Sans pondération,
les cinq compositions externes gagnent respectivement 2, 4, 5, 12 et 11 groupes.
Avec pondération, elles gagnent 2, 12, 8 et 25 groupes, mais `SS2-88-F` en perd 9.

L'intervalle descriptif par rééchantillonnage des cinq compositions est de
+0,996 à +2,496 points sans pondération et de -0,725 à +3,532 points avec
pondération. La seconde plage inclut zéro. Une seule graine et un seul fold
déjà utilisé pour le diagnostic ne constituent pas une preuve de généralisation.

## Audit et portée du verdict

L'audit a vérifié les huit époques internes et huit finales de chaque modèle,
les données et partitions communes, les paramètres initiaux, les graines,
les poids de classes, les permutations de lots observées et l'alignement
des prédictions. Les archives et leurs inventaires ont été contrôlés par
SHA-256. L'argmax et les métriques ont été recalculés depuis les probabilités
sauvegardées. L'audit complet rejoué localement produit un rapport identique,
octet pour octet, à celui de GitHub Actions. L'inférence TensorFlow depuis
les poids n'a pas été rejouée localement.

Il s'agit du **composant natif de comptage à sept classes**, sans correcteur
ajouté. La chaîne V27.3 complète, son ancre, ses fusions et ses transitions
ne sont pas évaluées par ce test. **V27.3 reste la référence officielle à
42,6019 %**, sur une autre population et avec une autre chaîne ; une
comparaison directe à 38,9030 % serait incorrecte.

Seul le fold 3 a servi de test externe. Le proposant historique commun n'a
pas été réentraîné séparément dans chaque partition. Aucune époque n'a été
choisie sur le test externe, aucun correcteur de sortie n'a été ajouté et
aucune promotion de modèle n'a été effectuée.

Le résultat justifie un **gain limité de comptage dans cette expérience**.
Il ne justifie pas de déclarer le surcomptage résolu ou de remplacer V27.3.
L'audit situe les limites restantes : fausses détections à K=0, confusion
entre K=2, K=3 et K=4, et absence de comptes exacts à K=6. Leur cause
acoustique ou d'apprentissage reste à établir par une expérience dédiée.

- [Rapport d'audit complet](v273-window-pair-audit.json).
- [Provenance, empreintes et vérifications](v273-window-pair-audit-sources.json).
- [Protocole fixé avant résultats](v273-window-pair-protocol.md).
- Exécution terminée : https://github.com/Andriamarosoa/note/actions/runs/36351028493
- Archives des quatre modèles : https://github.com/Andriamarosoa/note/releases/tag/v273-window-pair-36351028493
- Source de l'expérience : `cdb11b6cfe50beac86194fc6a2ddca06149e8467`.

Ce bilan complet remplace, pour l'état courant, le rapport intermédiaire
de la comparaison principale. Celui-ci reste conservé comme historique.
