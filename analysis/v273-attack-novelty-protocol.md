# Protocole — nouvelle attaque vs composante déjà présente

Date : 6 octobre 2026.

## Question

Sur la cohorte acoustique figée des 488 cas K2/K3, les fréquences que le
reconstructeur choisit à la place des notes attendues sont-elles plus souvent
**déjà présentes avant l'événement**, tandis que les fréquences annotées comme
attaques du groupe montrent une augmentation plus nette entre PRE et POST1 ?

Ce protocole est diagnostique. Il ne modifie aucun correcteur Exact-K, aucun
seuil et aucun réseau.

## Population figée

- mêmes 488 cas de l'audit acoustique publié ;
- folds internes 0, 1, 2 et 4 uniquement ;
- aucun fichier du fold 3 ;
- mêmes groupes figés :
  - K3_regressed : 125 ;
  - K3_preserved : 147 ;
  - K2_corrected : 108 ;
  - K2_missed : 108 ;
- audio normal uniquement.

Les annotations servent uniquement à identifier les fréquences attendues et à
classer les composantes choisies en appariées / non appariées. Elles ne sont
jamais une entrée d'inférence.

## Fenêtres

Pour chaque début de groupe s, avec WINDOW=2048 à 44,1 kHz :

- PRE    : [s-2048, s)
- POST1  : [s, s+2048)
- POST2  : [s+2048, s+4096)

Même fenêtre Hann et même FFT 8192 que les audits précédents.

## Support harmonique

Pour une fréquence f, le support est la somme de puissance dans l'union des
bandes de largeur +/- KERNEL_HZ autour de h*f pour h=1..MAX_HARMONICS, limitée
à la bande d'analyse existante.

Pour chaque f, enregistrer :

- support brut PRE, POST1, POST2 ;
- support normalisé par la puissance totale de la fenêtre ;
- gain brut POST1-PRE ;
- contraste d'attaque (POST1-PRE)/(POST1+PRE+eps) ;
- ratio log log((POST1+eps)/(PRE+eps)) ;
- maintien tardif POST2/(POST1+eps) ;
- support positif conservé max(POST1-PRE,0) dans les mêmes bandes.

Aucun epsilon n'est ajusté sur les folds ; eps=1e-12.

## Comparaisons principales

### A. Vraies notes K3 vs composantes choisies non appariées

Sur les 272 vrais K3 :

1. mesurer les trois fréquences annotées ;
2. prendre le triplet choisi par le reconstructeur de première fenêtre déjà
   archivé ;
3. apparier à 55 cents de façon univoque ;
4. séparer les composantes choisies en :
   - selected_matched ;
   - selected_unmatched.

Comparer expected vs selected_unmatched pour :

- support PRE normalisé ;
- contraste PRE->POST1 ;
- log-ratio PRE->POST1 ;
- maintien POST2/POST1.

Rapporter aussi des différences appariées au niveau du cas lorsqu'au moins une
composante non appariée existe.

### B. K3 dégradés vs K3 préservés

Pour les fréquences annotées, comparer les résumés par cas entre K3_regressed et
K3_preserved afin de savoir si les attaques réellement attendues sont
acoustiquement plus faibles ou déjà davantage présentes en PRE dans les cas
dégradés.

### C. K2 corrigés / K2 manqués

Même diagnostic descriptif pour les fréquences attendues et la composante
supplémentaire du triplet lorsqu'elle est non appariée.

## Critères d'interprétation

Cette étape ne sélectionne aucun bras d'inférence.

Une piste "onset-aware" n'est autorisée pour une expérience suivante que si le
même sens descriptif apparaît de manière cohérente :

- expected plus neuf que selected_unmatched dans les K3 ;
- effet visible séparément dans plusieurs folds, pas seulement en agrégé ;
- pas d'explication triviale par une seule poignée de lignes extrêmes.

Sinon, la piste nouvelle-attaque / déjà-présente est rejetée sans nouvel
entraînement.

## Sorties exigées

- cases.jsonl : mesures par cas et composante ;
- report.json : agrégats globaux, par groupe et par fold ;
- aucune prédiction modifiée ;
- aucun gain Exact-K annoncé ;
- fold 3 explicitement absent ;
- empreintes SHA-256 des sources et du script.

## Limites

Le support dans des bandes harmoniques ne sépare pas physiquement les sources
lorsque plusieurs notes partagent des partiels. Une fréquence choisie peut
représenter un harmonique, sous-harmonique, voisin ou composant étranger. Les
résultats restent un diagnostic du chemin normal.
