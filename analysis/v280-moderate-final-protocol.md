# V28.0-I — dernier essai de pondération intermédiaire

Protocole du 16 septembre 2026, figé avant l'apprentissage. L'utilisateur a
demandé de revenir à V27.3 si cette piste n'apporte pas de bénéfice convaincant.
G a montré un compromis défavorable avec une masse polyphonique de 50 % ;
H n'a apporté qu'un petit gain incertain au prix de deux réseaux. I teste
**une seule hypothèse supplémentaire**, puis applique la règle d'arrêt.

## Hypothèse et budget

Un seul modèle harmonique, 110 402 paramètres, est entraîné à neuf pendant
**12 époques complètes**, en deux blocs de six avec reprise des derniers
poids et de toutes les variables Adam. Les poids du meilleur état servent
à la comparaison ; ils ne servent pas à reprendre l'apprentissage.

La masse de la perte principale allouée aux lignes K≥2 est fixée à **35 %**,
contre 65 % aux lignes K<2. Il s'agit d'une hypothèse intermédiaire choisie
avant I, pas d'une valeur optimisée ni d'une recherche sur plusieurs poids.

\[
w_{K\ge2}=0,35\,N_{fit}/N_{poly},\qquad
w_{K<2}=0,65\,N_{fit}/N_{nonpoly}.
\]

Seules les 46 921 lignes d'ajustement des folds 2, 3, 4 déterminent ces poids :
5 514 polyphoniques et 41 407 autres. La moyenne des poids vaut 1 avant
l'arrondi float32. Il ne s'agit pas d'un suréchantillonnage : chaque ligne
est visitée une fois par époque. Les cibles et masques auxiliaires, leurs
coefficients, le sampler, l'architecture, l'initialisation par couche,
Adam 2e-4, seed 28035, batch 32 et l'absence d'augmentation restent ceux de G.
Le budget atteint **17 604 mises à jour**.

Validation interne : fold 1, 14 001 lignes dont 1 776 polyphoniques.
Fold 0 exclu de l'apprentissage et de l'inférence. La validation reste non
pondérée. Les métadonnées hors ajustement sont lues pour vérifier les
partitions, sans déterminer les poids. Le fichier événementiel externe est
vérifié par son empreinte, sans désérialisation pour cet essai.

La sélection conserve exactement la règle de G : nombre maximal de corrects
polyphoniques, puis NLL polyphonique minimale, puis époque la plus ancienne.
Les résultats du meilleur et du dernier état sont audités. Toutes les courbes
sont conservées. Aucun autre checkpoint n'est choisi après le verdict pour
contourner un échec. Le décodeur reste argmax, sans fusion ni recalibration.

## Comparateur figé et sources

Le témoin uniforme est celui de G : run `34684806334`, tentative 1, commit
`bff41aec998a798055687c94705629eb98626aee`, artefact `v280-g-control-2`
ID `10299622000`. Il a déjà effectué les mêmes 12 époques ; **il n'est pas
réentraîné**. L'expérience ajoute donc un seul entraînement, comparé à ce
témoin figé, et non à V27.3 sur des partitions incompatibles.

Le JSON de comparaison G est fixé par SHA-256
`680e54d783bae6894d42058fbe0f3f9845a70b2c64914c4124910ae78edbe2a9`.
Il contient l'état exact attendu du témoin. Ses artefacts, poids, prédictions
du meilleur et du dernier état, sorties intermédiaires et métriques sont
revérifiés. La comparaison I exige les mêmes initialisations complètes,
paramètres, identités, ordres de lots et 17 604 mises à jour dans les deux bras.
Les pertes d'entraînement des deux objectifs ne sont pas directement comparables.

Le cache provient de F : run `34462604665`, artefact `v280-f-prepared-all`,
ID `10146298232`, manifeste SHA-256
`e43567e3bd0945911915b4b11d88f960e4d95c0519809045dc9c7b056f0e5226`.
L'original expire le 17 septembre ; I conserve une copie vérifiée pendant
90 jours. Les fichiers scientifiques d'origine ne sont pas réécrits.
La préparation ajoute seulement `i-preflight.json`. Tous les workers
revérifient les fichiers selon le manifeste brut original et les empreintes
des implémentations F/E/modèle/G. Les IDs et SHA des sources sont aussi
vérifiés par l'API GitHub avant tout entraînement.

## Verdict défini avant I

Le gain doit satisfaire **tous** les critères suivants face au témoin G,
sur exactement les mêmes lignes et avec la règle de sélection commune :

1. Au moins **+5 points** d'Exact-K polyphonique. Cette exigence de taille
   d'effet reprend l'ordre de grandeur du critère matériel de la recherche V28 ;
   ici le comparateur est le témoin interne G, pas V27.3 externe.
2. Borne inférieure du bootstrap apparié par piste à 95 % **strictement > 0**.
3. Nombre total de corrects **au moins égal** à celui du témoin.
4. Nombre de lignes réellement K<2 prédites K≥2 **au plus égal** au témoin.

Le bootstrap emploie 10 000 réplications, seed 28035. Il reste descriptif et
conditionnel à la sélection des checkpoints ; cette partition a déjà servi
au développement. Le passage de ces critères n'est pas une confirmation
indépendante ni une preuve de supériorité face à V27.3.

Si un seul critère échoue, `decision.json` conclut
**`stop_v28_return_to_v273`** : cette piste V28 est close et la prochaine
amélioration doit repartir de V27.3. Aucun nouvel ajustement de pondération,
ensemble ou seuil n'est enchaîné. Un gain minime ne suffit pas.

Si tous passent, la conclusion est seulement
`material_internal_candidate_requires_confirmation`. V27.3 reste la référence
officielle ; aucune promotion, évaluation externe ou nouvelle expérience
n'est automatiquement déclenchée. Le protocole externe complet reste nécessaire
avant toute éventuelle promotion.

Un échec technique ou un entraînement incomplet ne vaut pas verdict scientifique :
le calcul de décision exige les deux blocs complets et tous les audits valides.
Les historiques, matrices de confusion, NLL, Brier, sous-comptages et sorties
intermédiaires sont publiés même lorsque le résultat scientifique est négatif.

## Vérification et conservation

Les tests couvrent les masses 35/65 et gradients Keras, l'indépendance des
poids vis-à-vis des labels hors ajustement, les masques auxiliaires, les
12 époques et la reprise Adam, l'exclusion des features du fold 0, la
comparaison à un témoin d'un autre run strictement figé et la règle de retour
à V27.3. Les tests TensorFlow doivent réussir dans Actions avant GuitarSet.

Artefacts, tous conservés 90 jours : `v280-i-prepared`,
`v280-i-preparation-audit`, `v280-i-reference`, `v280-i-moderate-1/2`,
`v280-i-final-verdict`. La durée d'apprentissage attendue est de plusieurs heures.
