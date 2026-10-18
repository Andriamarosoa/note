# S710–S712 : nouvelles boucles pour six caractéristiques au-dessus de 80 %

**Résultat vérifié : objectif non atteint.** Les nouvelles expériences ont
terminé 7 séries et 386 comparaisons CAL : 146 pour S710, 216 pour S711,
24 pour S712. S711 a effectué 16 époques supplémentaires. S710 et S711 sont
évalués et recomptés sur HELD/DEV ; S712 est écarté sur CAL. Aucun candidat
n’est promu et le LIVE officiel reste inchangé.

Avec les expériences S704–S709 déjà documentées, le total est de **26 séries
et 3 743 comparaisons CAL**. Ce total compte des configurations évaluées,
pas des extraits indépendants ni des modèles tous entraînés séparément.

L'objectif est **strictement plus de 80 % sur chacune des six mesures**, pour
un même système et un même flux de notes émises. Une moyenne de 80 %, un bon
sous-ensemble solo ou une combinaison des meilleurs chiffres de plusieurs
modèles ne satisfait pas l'objectif.

## S710 : diagnostic puis 146 comparaisons de décodage

S710 compare les sorties acoustiques gelées de S706 et S709. Pour chacune,
72 filtres d'état communs aux hauteurs et au nombre de notes sont comparés
au décodeur de contrôle original. Les 32 extraits CAL déterminent le choix ;
ce choix est enregistré avant l'évaluation des 32 extraits HELD et des
12 extraits DEV communs.

Le nouveau filtre utilise la probabilité acoustique de présence, l'attaque,
la relâche, l'état précédent et le nombre de notes prédit. Il impose une
preuve acoustique minimale pour ouvrir une note, une capacité de six notes
et une durée minimale ; les étiquettes ne sont pas une entrée du décodage.

Le classement exige d'abord les six seuils, puis maximise le plus faible des
six scores, avec leur moyenne pour départager. **Le contrôle S709 reste le
meilleur choix sur CAL**. Les nouveaux filtres améliorent certains scores
de présence et de comptage, mais dégradent suffisamment les attaques ou les
notes complètes pour perdre ce classement. S710 n'apporte donc aucun gain
validé : son contrôle reproduit exactement les comptes S709 sur HELD et DEV.

| Mesure du même flux S710/S709 | CAL, 32 extraits | HELD, 32 extraits | DEV, 12 extraits |
|---|---:|---:|---:|
| F1 onset, bonne hauteur et ±80 ms | 73,69 % | 71,50 % | 66,86 % |
| Nombre exact de notes actives | 63,33 % | 64,67 % | 62,50 % |
| Nombre exact, trames actives seulement | 57,03 % | 60,63 % | 57,23 % |
| Offset correct ±50 ms, conditionnel | 76,09 % | 66,21 % | 73,34 % |
| F1 des notes complètes | 55,77 % | 46,49 % | 48,62 % |
| F1 des hauteurs actives par trame | 77,24 % | 82,83 % | 71,47 % |

Les six caractéristiques ne dépassent pas toutes 80 %. Le contrôle échoue aussi
au critère strict de non-régression : par rapport au système natif + S675,
3 attaques de notes longues sont perdues sur HELD et 1 sur DEV ; respectivement
294 et 129 fins auparavant correctes sont perdues. Les gains agrégés ne
suppriment pas ces régressions individuelles.

## Ce que le diagnostic CAL permet de conclure

Ces nombres décrivent les probabilités internes ; **ce ne sont pas les
performances du flux de notes émis**.

Pour S706, 1 046 des 1 088 notes manquées n'atteignent pas conjointement
les seuils d'attaque et de présence dans la fenêtre d'appariement. Parmi les
fins des notes appariées et évaluables, 504 sont trop précoces, 248 trop
tardives, 2 sont des clôtures forcées et 2 223 sont correctes. L'erreur de fin
aux centiles 10/50/90 vaut −134,92 / 6,17 / 45,78 ms.

Le nombre de notes prédit par la tête S706 est correct sur 69,61 % des
31 568 blocs CAL, avant application du décodeur. La difficulté augmente
nettement avec les notes simultanées :

| Nombre de notes de référence | Blocs CAL | Comptage interne correct, S706 |
|---|---:|---:|
| 0 | 6 499 | 84,64 % |
| 1 | 13 267 | 84,89 % |
| 2 | 3 716 | 45,96 % |
| 3 | 2 967 | 39,53 % |
| 4 | 4 180 | 51,82 % |
| 5 | 750 | 21,73 % |
| 6 | 189 | 0,00 % |

Le comptage des notes et celui des hauteurs distinctes diffèrent sur 97 blocs
CAL sur 31 568 (0,31 %). Il existe 12 paires d'attaques successives de même
hauteur distantes de moins de 65 ms parmi 4 083 notes de référence. Aucune
note CAL ne dure moins de 35 ms. Ces contrôles ne donnent pas un plafond de
performance ; ils quantifient quelques cas où les contraintes du décodeur
peuvent intervenir.

Le profil par nombre de notes est reproductible avec
`scripts/s710_calibration_error_profile.py`. Ses totaux de comptages justes,
sous-estimés et surestimés concordent avec le diagnostic initial S710.

## S711 : apprentissage d'un contexte acoustique commun

S711 part du modèle S706 sélectionné à l'époque 10. Il ajoute une attention
entre les 49 hauteurs du bloc courant et un contexte acoustique commun aux
têtes d'attaque, de présence et de fin. L'historique temporel reste limité
aux 30 blocs passés ; aucun bloc futur n'est ajouté. Les résidus sont
initialisés à zéro pour préserver les prédictions S706 au démarrage.

La fonction de coût renforce le comptage et ajoute une cohérence de
transition entre présence précédente, attaque, relâche et présence courante.
Les cibles de transition servent uniquement à l'apprentissage FIT.

L'apprentissage a été évalué après 4, 8 puis 16 époques supplémentaires, avec 72 décodeurs
CAL complets par budget : 24 variantes du décodeur à seuils et 48 variantes
du filtre d'état S710. Le choix est gelé à partir de CAL. Les tests couvrent
la conservation des sorties au démarrage, l'isolation des trames futures,
la concordance entre calcul incrémental et par blocs, et l'objectif de
transition. Les 13 tests de la suite CI S711 passent.

| Budget supplémentaire | Onset CAL | Notes complètes CAL | Plus faible des six scores CAL |
|---|---:|---:|---:|
| 4 époques | 74,01 % | 55,11 % | 55,11 % |
| 8 époques | 73,34 % | 54,72 % | 54,72 % |
| 16 époques — retenu | 73,60 % | 55,19 % | 55,19 % |

Le point à 16 époques est retenu : son plus faible score CAL dépasse celui
du point à 4 époques de seulement 0,09 point. Le choix vérifié parmi les
216 politiques est `joint_on0.75_active0.5_off0.3_mix0.7`. Le gain CAL est
faible et ne franchit pas le seuil demandé.

| Mesure du même modèle S711 retenu | CAL, 32 extraits | HELD, 32 extraits | DEV, 12 extraits |
|---|---:|---:|---:|
| F1 onset, ±80 ms | 73,60 % | 73,84 % | 67,54 % |
| Nombre exact de notes | 61,97 % | 63,98 % | 58,90 % |
| Nombre exact, trames actives | 55,30 % | 59,92 % | 52,17 % |
| Offset conditionnel, ±50 ms | 75,45 % | 67,26 % | 72,96 % |
| F1 des notes complètes | 55,19 % | 48,78 % | 48,89 % |
| F1 des hauteurs par trame | 76,51 % | 82,31 % | 69,80 % |

**Aucun des deux groupes de validation ne passe les six seuils.** Le meilleur
onset DEV mesuré dans les expériences précédentes reste S708 à 68,75 % ;
S711 obtient 67,54 % sur les mêmes 12 extraits. Ces deux valeurs désignent
des modèles différents et ne sont pas combinées avec les autres scores.

Le critère strict de non-régression échoue également. Par rapport au natif
+ S675, S711 perd 3 attaques de notes longues et 369 fins auparavant
correctes sur HELD ; sur DEV il perd 0 attaque longue et 144 fins auparavant
correctes. Les entrées audio, les références et les événements des deux
baselines ont été contrôlés identiques à ceux de S710.

La durée d’inférence des têtes seules, mesurée sur le runner CPU, est de
7,76 ms à la médiane et 8,67 ms au centile 95. Elle exclut l’extraction
acoustique et ne valide pas une latence complète sur microphone.

## S712 : mémoire causale courte des attaques

Le détail du diagnostic S706 sépare les 1 088 notes manquées sur CAL en
581 cas où attaque et présence restent toutes deux sous leur seuil,
195 où seule l'attaque reste faible, 168 où seule la présence reste faible,
102 où les deux pics ne coïncident pas, et 42 autres échecs malgré un seuil
conjoint disponible. S709 comporte 89 cas de pics non simultanés.

S712 teste donc une mémoire de 0 à 3 blocs d'attaque et de 0 à 2 blocs de
présence, uniquement pour les notes inactives, avec une présence courante
minimale de 0,25. Les sorties de fin des notes actives restent celles du bloc
courant. Il n'y a ni accès au futur ni modification rétroactive des dates.

Les 12 combinaisons sont testées sur chacune des deux sources acoustiques,
soit **24 comparaisons CAL**. Quatre tests locaux passent : émission au bloc
courant après alignement, exclusion d'une présence ancienne lorsque la
présence actuelle est trop faible, fin courante, et isolation du futur.
Les contrôles sans mémoire reproduisent les comptes CAL de S710.

Le meilleur point utilise S709 et un seul bloc d'historique d'attaque
(23,22 ms), sans historique de présence. Son flux CAL est également
recompté indépendamment :

| Mesure CAL du même flux S712 | Résultat |
|---|---:|
| F1 onset | 74,03 % |
| Nombre exact | 65,04 % |
| Nombre exact, trames actives | 59,28 % |
| Offset conditionnel | 76,03 % |
| F1 des notes complètes | 55,97 % |
| F1 des hauteurs par trame | 78,61 % |

Le gain d'onset sur son contrôle S709 est de 0,34 point. Le plus faible des
six scores reste à 55,97 %, contre 56,28 % pour l'ancien S707-706 sur les
**mêmes 32 extraits CAL** (126 304 trames, 4 083 notes de référence). Ce
candidat est donc écarté avant HELD/DEV ; il n'a aucun résultat HELD/DEV
à présenter et n'est pas promu. Le rapport CAL complet conserve les 24
résultats, le contrôle, le choix et le recomptage indépendant.

Les CAL de S704, S705 et S707-705 utilisent 16 extraits ; leurs valeurs CAL
ne sont pas directement comparées ici aux 32 extraits des expériences récentes.

## Protocole et limites d'interprétation

- FIT : 80 extraits, compositions BN1, Funk2, Jazz1, Rock3 et SS1.
- CAL : 32 extraits, compositions BN2 et Jazz2.
- HELD : 32 extraits, compositions Jazz3 et SS3.
- DEV : les mêmes 12 extraits diagnostiques que les boucles précédentes.
- HELD et DEV ont déjà été exposés lors d'expériences précédentes. Ces
  résultats ne constituent pas une nouvelle estimation sur un test vierge.
  L'historique d'apprentissage des têtes natives reste également à considérer.
- TEST reste fermé. Le fonctionnement sur microphone réel n'est pas validé
  par ces relectures de fichiers.

Toutes les mesures proviennent du même flux d'événements. Une attaque est
appariée une seule fois, à hauteur correcte et dans ±80 ms. Le score de fin
est conditionnel aux attaques appariées dont la fin de référence est
évaluable ; le F1 de note complète inclut aussi les faux positifs et les
notes manquées. Avec les comptes TP, FP et FN d'attaque, il vaut
`2 × fins_correctes / (2 × TP + FP + FN)`. Un onset à 80 % et un offset à
80 % ne produisent donc pas 80 % de notes complètes.

Le comptage est plafonné à six pour le score, comme dans le protocole
historique. Les candidats émettent au plus six notes simultanées. La
référence native + S675 peut temporairement dépasser six (jusqu'à huit
dans les extraits HELD/DEV de S710) : cette différence reste visible dans
les preuves et n'est pas masquée par le score plafonné.

Les événements réguliers sont datés à la fin du bloc reçu de 1 024
échantillons (4 sauts de 256, à 44,1 kHz), sans antidatage. Les clôtures EOF
ne reçoivent aucun crédit d'offset. Aucun résultat n'est promu en LIVE.

## Preuves et reproduction

Les comptes HELD/DEV ont été recomputés à partir des JSONL par
`scripts/s704_verify_emitted_artifact.py`, sans charger un modèle. Le contrôle
local de l'archive S710 confirme aussi le SHA du modèle et les dates exactes
des émissions : 10 226 événements réguliers HELD et 3 188 DEV, plus 64 et
12 clôtures EOF respectivement. Le contrôle S711 confirme 8 831 événements
réguliers HELD et 2 770 DEV, plus 63 et 10 clôtures EOF, la limite de six
notes actives et le choix CAL parmi les 216 politiques. Les deux archives
finales téléchargées concordent avec les SHA publiés par GitHub.

- [Exécution S710](https://github.com/Andriamarosoa/note/actions/runs/39098582673)
  — commit `de619b5fd9ef75a13a83f083d8afa57e48d256d3`.
- [Exécution S711](https://github.com/Andriamarosoa/note/actions/runs/39099109305)
  — commit `352ac6f3780826f7defaa8cad92d5f06091e914a`.
- Artefact final S710 : `S710_JOINT_STATE_CAUSAL_RECOVERY`, ID `12125961589`,
  SHA256 `6be7b5db982a8123b53e9bfe820f82a86c9869b3d280c66b9a9d8a1649953167`.
- Artefact final S711 : `S711_GLOBAL_CONTEXT_CAUSAL_LOOPS`, ID `12126357427`,
  SHA256 `e164082096583d9d8d796d7337da4188885076b52dc0f5a283d1ea0c5d68b612`.
- Modèle S711 sélectionné : SHA256
  `5b7e5174812510b4b77a4e82e568e4961bcb08286caf972c2e3f4dae794e71ba`.
- Cache CAL S710 : `S710_CAL_DIAGNOSIS_AND_PROBABILITIES`, ID `12125556863`,
  SHA256 `9d02179aa2d20dce4edce32cdbfcc6238fd726e80f98f340502f094c5355112f`.

Le SHA imprimé par le recomptage CI S710, `6908896002f95fa2f896a56e9d941ba312bc2f9061649e799bc9c1f9db875762`,
désigne le ZIP intérieur contenant rapport et JSONL. Le SHA de l'archive
finale téléchargée est celui donné ci-dessus ; les deux ne désignent pas
le même fichier. Pour S711, le ZIP intérieur porte le SHA
`6fb328148e2a97b13adc8a66c15e2953cb5b03232f0973d7328613dad7942204`,
distinct lui aussi du ZIP final téléchargé.

Les preuves JSON persistantes sont sous
`analysis/evidence/s710-s712-loop-audit/`. Pour recomputer les comptes après
téléchargement de l'archive complète :

```bash
python scripts/s704_verify_emitted_artifact.py /chemin/s710-final.zip --out /chemin/s710-verified.json
python scripts/s704_verify_emitted_artifact.py /chemin/s711-final.zip --out /chemin/s711-verified.json
python scripts/s710_calibration_error_profile.py /chemin/cache-cal-verifie --out /chemin/cal-profile.json
PYTHONPATH=.:src python -m scripts.s712_calibrate_attack_memory /chemin/cache-cal-verifie --out /chemin/nouveau-cal-s712
PYTHONPATH=.:src python -m unittest -v test.test_s712_attack_memory
```

Le cache joblib est un artefact de cette expérience : vérifier sa provenance
et son SHA avant lecture. Le recomptage des événements n'en a pas besoin.
