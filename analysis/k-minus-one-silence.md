# K = −1 : silence ; K = 0 : aucune nouvelle attaque attribuée

Décision utilisateur du 30 septembre 2026 : « on va metre k=-1 pour le silence ».
Cette version introduit huit **états**. −1 est un symbole de silence et ne
représente jamais un nombre négatif de notes. Les classes du réseau sont
encodées 0…7 ; leur signification est −1…6.

| État | Cible |
|---|---|
| −1 | Silence vérifié sur la fenêtre audio originale, aucune attaque attribuée et aucune note annotée chevauchante. |
| 0 | Aucune attaque attribuée, avec activité audio ou une note annotée présente, y compris une note tenue ou attribuée au voisin. |
| 1…6 | Nombre d'attaques annotées attribuées au groupe selon la règle existante. |

Une réattaque à la même hauteur compte comme une nouvelle attaque. Une note
déjà tenue ne compte pas comme un nouveau départ. Une attaque au voisin ne
doit pas être comptée deux fois. Une attaque très faible conserve son K positif.
La règle d'attribution historique (candidat le plus proche à 20 ms maximum,
égalité départagée par ID) ne change pas.

## Définition opérationnelle du silence

On mesure le PCM original normalisé à [−1,1], avant toute compression de
hauteur, sur le support déjà lu par les cartes :
`[origine−3100, origine+2788[`, soit 133,51 ms à 44 100 Hz.

La première politique, explicite dans `v273-silence-policy.json`, exige :

- toutes les tranches de 256 échantillons ont un RMS ≤ −60 dBFS ;
- la crête de toute la fenêtre est ≤ −50 dBFS ;
- aucune note annotée ne chevauche la fenêtre et K historique vaut zéro ;
- tous les échantillons sont réellement disponibles : aucun zéro de padding
  ne constitue une preuve de silence.

Ces seuils conservateurs sont un choix d'ingénierie préalable à l'audit de
population, pas un seuil universel d'audibilité ni un réglage optimisé sur
les scores de validation. Un bruit supérieur à ces limites reste K=0 ; le
bruit de fond inférieur fait partie du silence opérationnel. Une annotation
de note tenue empêche −1 même si cette note est très faible. Les fenêtres
calmes incomplètes et sans annotation sont exclues comme inconnues ; −2 est
uniquement une sentinelle interne d'absence de cible, jamais une classe.

Les annotations servent à construire les cibles hors ligne. Elles ne sont
pas des entrées du modèle et ne servent jamais à corriger sa prédiction.

## Modèle et évaluation

Le nouvel entraînement emploie le même encodeur natif à quatre canaux et une
tête softmax à huit sorties. `encode_states` traduit −1…6 en 0…7 avant la
cross-entropy ; le décodage final est directement `argmax(P)−1`. Le réseau
apprend lui-même le silence. Le seuil acoustique construit les étiquettes,
il n'est pas un correcteur ajouté après le réseau.

Rapporter séparément :

- exactitude des huit états et matrice de confusion 8×8 ;
- rappel/précision de −1 et confusion −1↔0 ;
- Exact K polyphonique sur les mêmes groupes K≥2 ;
- faux départs lorsque la cible est −1 ou 0 ;
- exactitude du nombre d'attaques, en traduisant −1 et 0 en zéro attaque.

Une confusion −1↔0 est une erreur d'état mais pas une erreur du nombre
d'attaques. Prédire −1 pour K=2 reste un sous-comptage. Le score global à
huit états n'est pas directement comparable à l'ancien score à sept classes.

## Intégration et reproduction

Les nouveaux fichiers sont activés explicitement. Le run high pitch
`36703546041`, son manifeste et les constructeurs historiques restent figés
avec leurs sept classes. Ses résultats ne seront pas renommés rétroactivement.

```bash
PYTHONPATH=.:src python -B -m unittest -v test.test_count_states
PYTHONPATH=.:src python -B scripts/preflight_v273_silence_states.py --output model/silence-preflight
PYTHONPATH=.:src python -B scripts/prepare_v273_silence_states.py \
  --dataset data/GuitarSet --geometry model/source/ownership-inputs \
  --output model/silence-targets
PYTHONPATH=.:src python -B scripts/train_v273_silence_states.py \
  --bundle model/source/native-window-bundle --geometry model/source/ownership-inputs \
  --maps model/high-pitch-inputs --targets model/silence-targets \
  --output model/silence-training --arm observed_only --epochs 12
```

L'audit ne traite que les 190 pistes internes et publie la population −1/0/1…6
de chaque partition. Le lanceur refuse un entraînement à huit états si
l'apprentissage ou la validation ne contient aucun silence vérifié. Les
groupes issus de propositions d'attaque peuvent naturellement exclure les
zones silencieuses : dans ce cas il faut préparer des fenêtres de fond
réelles avec un contrat d'entrée compatible, avant tout entraînement.
Les résultats d'audit et de vérification sont conservés dans
`analysis/evidence/k-minus-one/`.

## Résultat de l'audit du 30 septembre 2026

L'audio original des 190 pistes internes a été relu et vérifié par empreinte.
Avec la politique ci-dessus, **aucun groupe candidat ne contient un silence
complet vérifié**. Ce résultat concerne cette population et ce seuil ; il ne
signifie pas que tous les fichiers audio sont dépourvus de plages silencieuses.

| Partition | Groupes audités | K = −1 | K = 0 retenus | Fenêtres calmes incomplètes exclues |
|---|---:|---:|---:|---:|
| Apprentissage | 43 357 | 0 | 28 863 | 26 |
| Validation | 15 952 | 0 | 10 753 | 10 |

Tous les K positifs sont conservés. Parmi les K=0 complets sans note annotée,
763 fenêtres d'apprentissage et 142 de validation dépassent les limites
acoustiques ; elles ne sont donc pas renommées « silence ». Les minimums du
maximum RMS par trame, pour ces fenêtres complètes sans note, sont respectivement
0,00886 et 0,01278 en amplitude normalisée (environ −41 et −38 dBFS).

La classe et le chemin d'entraînement sont implémentés ; **aucun nouveau modèle
n'a été entraîné sur GuitarSet**. Il faut d'abord échantillonner des plages de
fond réelles, avec des entrées et une politique de propositions compatibles,
et distinguer les silences des sons sans nouvelle attaque. On ne fabrique pas
une classe −1 en changeant les seuils après observation des scores ou en
renommant les K=0 actuels. Le lanceur refuse cette population sans exemple −1.

Vérification locale : 33 tests unitaires passent ; le graphe TensorFlow 2.15.1
à 249 379 paramètres effectue une mise à jour finie sur les huit classes
synthétiques. Les poids initiaux de l'encodeur commun et les constructeurs
historiques à sept classes sont inchangés. Ce contrôle valide le chemin de
calcul, pas la qualité d'une détection de silence appris.
