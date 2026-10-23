# S838 — Audit du premier sélecteur appris par fragment PCM256

## Preuves vérifiées
- [Run #40000089112](https://github.com/Andriamarosoa/note/actions/runs/40000089112) : SUCCESS, 36 tests unitaires et cinq vrais WAV GuitarSet signés, 20 062 frontières 256.
- Artifact `S838_REAL5_PRIVATE_K_FRAGMENT_GATES_TRAIN_CAL_OUTER`, ID `12521936945`, rapport JSON source.
- Sélection apprise **par fragment**, deux modèles d'axe indépendants : Exact-K et Active Exact-K. Chaque OUTER est exclu du TRAIN et du CAL : 3 compositions TRAIN, 1 CAL, 1 OUTER. Le MIDI public et les autres axes n'ont pas changé.

## Résultat signé
| Axe | Référence avant | Gate OUTER | Écart | Blocs corrects net |
|---|---:|---:|---:|---:|
| Exact-K | 56,599541 % | **55,926627 %** | **−0,672914 point** | −135 sur 20 062 |
| Active Exact-K | 46,772514 % | **47,241020 %** | **+0,468506 point** | +90 sur 19 210 blocs actifs |

### Exact-K : explication de la régression majeure SS1
BN1, Funk2, Jazz1 et Rock3 : **tête gelée** (aucune alternative retenue), donc aucun impact.

Seul OUTER **SS1** active le sélecteur, confiance `threshold=0.50`, appris sur Funk2/Jazz1/Rock3, seuil calibré sur BN1. Sur CAL BN1 : Exact-K 55,40 % référence et 57,32 % candidat. Sur **OUTER SS1**, ce modèle change **1 251 fragments** :
- **231 vrais K récupérés**, **366 vrais K perdus**, **135 blocs nets perdus**.
- Par vrai K : K0 −5 ; K1 : 22 récupérés, 83 perdus (**−61**) ; K2 : 73 récupérés, 116 perdus (**−43**) ; K3 : 110 récupérés, 0 perdu (**+110**) ; K4 : 1 récupéré, **119 perdu (−118)** ; K5 : 10 récupérés, 42 perdus (−32) ; K6 : 15 récupérés, 1 perdu (+14).
- Des fragments peuvent être déjà mal comptés par les deux experts ; 108 décisions modifiées de cette sorte.
- Cause structurelle **prouvée** : seuil global appris/validé sur d'autres compositions ne couvre pas les risques conditionnels selon la vraie polyphonie, et une amélioration CAL BN1 ne se transfère pas à SS1.
- Cause acoustique **non prouvée** : l'origine spectrale des faux K4/K1 reste à examiner. Interdit de prétendre qu'il s'agit forcément d'harmoniques.

**Hypothèse suivante H2**, à apprendre sur TRAIN et régler seulement sur CAL : seuil de risque conditionnel sur les probabilités K du réseau (et autres caractéristiques audio causales) plutôt qu'un simple seuil global. L'interdiction des mauvais K4 après avoir vu SS1 ne peut pas être revendiquée comme amélioration HELD sur **le même SS1** ; la prochaine boucle doit répéter sur d'autres sources et indiquer le caractère exploratoire FIT80.

### Active Exact-K : conserver le petit gain comme sélection distincte
- **BN1** : 21 récupérés, 1 perdu = +20, 102 fragments commutés.
- **Funk2** : 214 récupérés, 147 perdus = +67, mais **214 fragments changés restent faux dans les deux politiques**. Calibration insuffisante pour classer tous les cas.
- **Jazz1** : 119 récupérés, 121 perdus = **−2**, régression locale à bloquer.
- **Rock3** : 4 récupérés, 2 perdus = +2.
- **SS1** : 25 récupérés, 22 perdus = +3, avec 616 fragments commutés et **14 toujours faux**.

Le gain de +90 blocs est validé uniquement sur les 5 OUTER FIT déjà explorés, et **ne justifie pas une promotion**. Le gate est très actif dans certains cas où il ne produit pas de correction nette ; la prochaine tête de risque doit se concentrer sur le filtrage des transferts inutiles et des échecs par registre/K.

## Règles d'audit retenues
1. Aucun candidat inférieur au témoin comparable n'est promu, quelle que soit la moyenne d'une autre composition.
2. Une amélioration même faible est conservée comme candidate SHADOW propre à son axe, **sans interférence avec les cinq autres**.
3. Toute nouvelle solution doit montrer, par K0–K6 et piste, les TP/blocs corrects récupérés et perdus, et séparer causes prouvées de simples hypothèses acoustiques.
4. Aucun apprentissage à partir du contexte OUTER, aucun futur PCM, aucun délai artificiel, aucun remplacement du moteur NOTE Live avant six métriques appariées et p99 physique.
