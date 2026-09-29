# Exponentielle : conserver les oscillations, pas seulement leur puissance

**Verdict : une piste de représentation est étayée sur des sons synthétiques,
mais aucun gain d'Exact K sur GuitarSet n'est démontré.** Le prototype ne constitue
ni un nouveau modèle de comptage, ni un correcteur de sortie. V27.3 reste la référence.

## Défaut ciblé

L'ancienne branche logarithmique utilise déjà l'exponentielle. Elle ajuste
`ln(puissance) = a + bt` par bande puis prédit une décroissance. Le problème
n'est donc pas l'absence de `exp` : la puissance d'un mélange peut remonter
par interférence sans nouvelle attaque.

Pour deux coefficients spectraux complexes, la relation exacte est :

\[
|z_1+z_2|^2=|z_1|^2+|z_2|^2+2\operatorname{Re}(z_1\overline{z_2}).
\]

Le dernier terme dépend des phases relatives. L'ancien modèle de décroissance
par bande ne le représente pas explicitement. Le test publié antérieurement
montrait déjà des indices positifs sur deux sons continus ; cela ne démontrait
pas que cette ambiguïté causait toutes les erreurs de GuitarSet.

La nouvelle sonde conserve la forme d'onde. Une oscillation amortie correspond
à la partie réelle de `A exp((-lambda + i*2*pi*f)*t)` ; une somme finie de ces
oscillations obéit à une récurrence linéaire. Le prototype ajuste cette récurrence
sur le **PCM passé uniquement**, sans connaître les fréquences du générateur.
Il prédit le futur sans le réinjecter dans l'ajustement, puis mesure l'erreur
entre cette prévision et le PCM observé. Une grande erreur indique une évolution
mal expliquée par le modèle ; elle n'est pas automatiquement une note.

## Expérience réellement exécutée

- **1 944 signaux synthétiques** : six familles, seize phases, trois niveaux
  de bruit. Tous les cas et scores sont publiés dans le CSV associé.
- Par niveau de bruit : 192 sons tenus ou amortis sans nouvelle attaque,
  384 ajouts d'attaques (faibles, fortes, de même hauteur, deux notes),
  72 difficultés sans nouvelle note (vibrato, glissando, bruit transitoire).
- Ancien indice calculé avec le vrai prétraitement V100 et le cache float16.
- Trois prévisions PCM : historique 29,66 ms ; historique 200 ms à récurrence
  identique ; historique 200 ms et récurrence élargie. La dernière configuration
  change donc plusieurs facteurs, et n'isole pas l'effet de la durée seule.
- Même support futur de 40 ms pour les quatre méthodes. Davantage de passé
  n'ajoute pas de regard vers le futur ; il nécessite un tampon et une période
  d'initialisation. Aucune latence réelle de calcul n'a été mesurée.
- **Cinq tests réussis** : sons proches, harmoniques amorties, absence de
  contamination par le futur, silence/entrées invalides, bruit non musical.

Ces configurations ont été développées sur des oscillateurs apparentés :
ce banc est exploratoire, sans partition indépendante ni intervalle statistique.
La sonde PCM dispose de la phase, absente des cartes de puissance originales.

## Résultats sans masquer les cas difficiles

L'AUC mesure ici le classement des attaques ajoutées au-dessus des cas sans
nouvelle note. **Ce n'est ni une précision, ni Exact K, ni un score du réseau.**
Les unités des quatre indices diffèrent : leurs valeurs brutes ne sont pas
comparées entre méthodes. Aucun seuil de détection n'est sélectionné.

AUC en incluant **toutes** les difficultés sans nouvelle note :

| Méthode | Sans bruit ajouté | Bruit SNR 40 dB | Bruit SNR 20 dB |
|---|---:|---:|---:|
| Ancien indice de puissance logarithmique | 0,5453 | 0,5804 | 0,5084 |
| PCM, historique court | 0,9173 | 0,8888 | 0,6955 |
| PCM, historique long, même récurrence | 0,8966 | 0,8427 | 0,7940 |
| PCM, historique long, récurrence élargie | 0,9125 | 0,8717 | 0,8483 |

L'historique long élargi améliore nettement le classement à SNR 20 dB sur ce
banc, mais le modèle court est meilleur sans bruit et à SNR 40 dB lorsque
tous les cas difficiles sont inclus. Ce résultat ne justifie pas un remplacement
universel par la plus longue fenêtre.

Si l'on retire les difficultés et ne garde que les sons tenus ou amortis,
les AUC à SNR 20 dB sont respectivement **0,5056 ; 0,7494 ; 0,9183 ; 0,9998**.
Ce dernier chiffre, beaucoup plus favorable, serait trompeur s'il était présenté
seul. Les bruits transitoires, en particulier, produisent une erreur de prévision
élevée sans introduire de nouvelle note musicale.

L'ancien indice vaut exactement zéro sur **216 des 384 ajouts d'attaque** à
SNR 20 dB. Son filtrage des pentes jugées fiables peut donc aussi laisser passer
des attaques sans fournir d'indice. Ce sont les sorties de la caractéristique,
pas des erreurs du modèle de comptage.

## Décision

1. Ne pas relancer l'ancienne branche en remplaçant simplement `ln` par `exp`.
2. Retenir comme hypothèse de représentation la prévision de la forme d'onde,
   avec sa phase, plutôt qu'une seule décroissance de puissance par bande.
3. Avant un entraînement de comptage, vérifier la séparation entre attaques,
   sons tenus et bruits sur le PCM réel d'une seule partition interne, avec
   protocole fixé et conservation des cibles d'appartenance aux groupes.

Cette dernière vérification réelle **n'a pas été exécutée ici**. Le prototype
n'est pas intégré au réseau, ne regroupe pas les harmoniques en notes et ne
résout pas l'attribution d'une attaque au bon groupe. Aucun nouvel entraînement
GuitarSet n'est lancé par cet audit, et aucun poids historique n'est modifié.

## Sources et reproduction

- [Protocole](v273-coherent-decay-protocol.md).
- [Résultats et empreintes](v273-coherent-decay-results.json).
- [Tous les cas](v273-coherent-decay-results-cases.csv).
- [Source du prototype](../scripts/probe_v273_coherent_decay.py).
- [Limite antérieure de l'indice](v273-native-decay-diagnosis.md).
- Duxbury, C., Bello, J.P., Davies, M. et Sandler, M. (2003), *Complex domain
  onset detection for musical signals*, DAFx-03 :
  https://www.dafx.de/paper-archive/2003/pdfs/dafx81.pdf.
  Cette référence motive l'intérêt de la phase ; notre récurrence n'est pas
  une reproduction de leur détecteur, et leurs performances ne sont pas les nôtres.

```sh
OPENBLAS_NUM_THREADS=1 python -B -m unittest -v test.test_v273_coherent_decay
OPENBLAS_NUM_THREADS=1 python -B -m scripts.probe_v273_coherent_decay \
  --output analysis/v273-coherent-decay-results.json
```
