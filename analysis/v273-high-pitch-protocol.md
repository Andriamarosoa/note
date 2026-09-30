# High pitch rescue — protocole avant entraînement

Autorisation : « vas y » le 30 septembre 2026, après proposition d'une vue
normale et d'une vue comprimée vers l'aigu, avec un seul comptage natif K.

## Question et population

La compression ×2 (+12 demi-tons) améliore-t-elle le comptage des groupes,
notamment K=3 avec une note grave, sans dégrader les autres registres ?

Partition interne inchangée : 43 357 groupes d'apprentissage, 15 952 de
validation, 140 et 50 pistes, 14 et cinq compositions disjointes. Les mêmes
cinq guitaristes sont présents des deux côtés. Les compositions de validation
ont déjà été examinées ; ce résultat sera exploratoire. Aucun calcul sur le
fold externe 3 ni Locked12. Une seule graine ne démontre pas la généralisation.

L'audit préalable des annotations a retrouvé 9 127 attaques attribuées aux
groupes de validation : 1 434 sous C3, 7 427 de C3 à B4, 266 à partir de C5.
Ce sont des catégories d'audit explicites, pas des classes du réseau. Les
hauteurs sont arrondies au demi-ton MIDI le plus proche. K compte les nouvelles
attaques attribuées au groupe, pas toutes les notes encore audibles.

## Intervention et causalité

Les vues partagent exactement les fenêtres source de 256 et 2 048 échantillons,
les 31 fins de trame, les 64 fréquences de 55 à 6 000 Hz et les mêmes candidats.
Les observations normales sont identiques à celles du contrôle du run
36688979041. Aucun résidu acoustique n'entre dans cette expérience.

| Bras | Canaux |
|---|---|
| observed_only | normal256, normal256, normal2048, normal2048 |
| with_high_pitch | normal256, compressé256→128, normal2048, compressé2048→1024 |

Le filtre passe-bas FIR de 127 coefficients, fenêtre Kaiser bêta 8,6, coupure
0,225 cycle/échantillon précède la décimation. Chaque fenêtre passée est filtrée
séparément, avec extension synthétique par zéros aux deux bords ; aucun véritable
échantillon futur n'est lu. La phase impaire de décimation conserve l'emplacement
du dernier échantillon source. Le résultat est interprété à 44,1 kHz : la hauteur
double, la durée de la vue est divisée par deux. La FFT reste de taille 2 048 et
les puissances sont calibrées par la somme de la fenêtre de Hann, puis log1p.

Les horodatages, masques d'appartenance et K restent sur l'horloge d'origine.
La vue comprimée apporte **zéro historique et zéro anticipation audio de plus**.
Toutes les données source restent dans [origine−3100, origine+2788[. L'attente
audio existante, jusqu'à environ 80,11 ms selon les propositions, est inchangée.
Le surcoût CPU sera mesuré et publié ; aucune accélération réelle n'est promise.

Il s'agit d'un test de représentation à quantité de signal égale. Une vue de
taille constante après compression exigerait davantage de passé ; ce serait
une expérience différente. La compression ne crée pas de périodes encore
inobservées. Le filtrage peut atténuer les harmoniques élevées et introduire des
effets de bord ; la vue normale reste présente pour conserver son information.

## Conditions d'apprentissage

- Même architecture à quatre canaux, même nombre de paramètres et mêmes poids
  initiaux ; les deux modèles sont entraînés de zéro.
- Graine 16164, dropout de comptage 46164, même ordre des exemples par époque.
- Python 3.11, TensorFlow 2.15.1, NumPy 1.26.4, opérations déterministes.
- Adam 0,0002, batch 128, cross-entropy non pondérée, normalisation fixe /12.
- **12 époques**, aucune sélection de checkpoint. Époque 12 primaire ; 4 et 8
  sont conservées à titre descriptif. Décodage argmax P(K=0..6).
- Un seul K appris à partir des vues ; aucun correcteur de sortie ni addition
  de comptages. Les annotations de hauteur servent exclusivement à l'audit.

## Contrôles et décision

Avant entraînement : doublement de fréquence sur sinus, rejet d'aliasing,
égalité des canaux normaux avec le contrôle antérieur, invariance à toute
modification du futur et du passé hors support, bords/silence, poids et dropout
appariés, gradients non nuls sur les canaux transposés, mise à jour finie.
Les archives et fichiers sont épinglés par empreintes. Les étiquettes K de
validation sont reconstruites depuis les annotations et l'affectation historique
au candidat le plus proche, rayon 882 échantillons, départage par ID de groupe.

Publier Exact K global, K>=2, chaque K, sur/sous-comptages, matrice de confusion,
comparaisons appariées, cinq compositions séparées et intervalles descriptifs
par bootstrap de composition (2 000 tirages, graine 9302026). Publier aussi les
groupes avec grave, avec aigu, registres purs/mixtes, doublures à l'octave et
K=3 avec/sans grave. Ces strates peuvent se chevaucher ; ne pas les additionner.

Critère fixé favorable à une réplication : amélioration d'Exact K polyphonique
et de K=3 avec grave ; absence de régression globale, pour K=1/2/3 et pour les
groupes contenant un aigu ; baisse des surcomptages K<4. Tous les résultats,
même défavorables, sont publiés automatiquement. Aucune promotion automatique
ni remplacement du score officiel V27.3. La rareté des aigus et les cinq seules
compositions limitent la portée des conclusions.
