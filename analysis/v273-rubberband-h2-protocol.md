# Protocole — H2 Rubber Band vs Librosa

Date : 6 octobre 2026.

## Question

Le signal H2 vient-il de la structure musicale ou d'artefacts propres au moteur
de pitch-shift Librosa/soxr ?

## Règle figée

Aucun changement de décision :

- vues : -2, -1, 0, +1, +2 demi-tons ;
- margin = P(K2) - P(K3) ;
- action 3->2 si mean(margin) > 0.

## Seule variable modifiée

Les vues ±1/±2 sont produites par **Rubber Band** via `pyrubberband.pitch_shift`
au lieu de `librosa.effects.pitch_shift`.

La vue 0 reste exactement la probabilité figée de référence.

## Mesure interne

Même cohorte B_low + base K3 K2/K3, 488 lignes, folds 0/1/2/4, fold 3 exclu.
Rapporter corrections, régressions et net par fold et total.

Comparaison de référence Librosa H2 fixe (run 37404522178) :
- fold nets : 0:+5, 1:+6, 2:-4, 4:+3 ;
- total : 26 corrections / 16 régressions = +10.

## Mesure outer globale

Même population que le run 37421205179 :
- outer fold 3, 15,279 lignes ;
- B_low & base K3 uniquement ;
- aucune sélection/tuning sur outer.

Référence Librosa outer H2 :
- 21 actions ;
- 5 corrections / 5 régressions / 11 neutres ;
- net 0 ;
- Exact-K global 83.533% -> 83.533% ;
- Poly Exact-K 39.817% -> 39.817%.

## Interprétation

Le signal H2 sera considéré plus crédible s'il garde le même signe avec Rubber
Band, particulièrement sur >=3/4 folds internes. Un gain outer >0 serait
exploratoire car fold 3 a déjà été exposé.

Aucune promotion automatique.
