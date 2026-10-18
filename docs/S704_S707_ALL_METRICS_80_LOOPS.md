# NOTE — boucles S704 à S707, cible supérieure à 80 %

État intermédiaire : **S704 et S705 terminés et recomptés ; cible non atteinte**.
S706 et les deux affinages S707 poursuivent leurs calculs. Aucune promotion LIVE.

La demande porte sur tous les scores d'une même chaîne : Onset avec bonne
hauteur (±80 ms), Exact-K décodé, Offset conditionnel (±50 ms) et F1 des notes
complètes. Le contrôle comprend aussi Exact-K sur les seules trames actives
et le F1 micro des hauteurs par trame. Le critère est strictement `> 0.80`
pour chacun ; il ne repose ni sur une moyenne entre scores ni sur les silences.

Tous ces scores sont reconstruits à partir des mêmes événements sérialisés.
Les émissions sont datées à la réception des blocs de quatre hops (1 024
échantillons, 23,22 ms). Ce sont des relectures causales de WAV, pas une mesure
de la latence totale d'un microphone : le temps de calcul et les délais du
périphérique ne sont pas ajoutés aux horodatages audio du benchmark.

## Résultats DEV déjà vérifiés

12 prises GuitarSet, 1 580 notes de référence, 48 970 trames. Les variantes
sont figées sur CAL avant DEV. DEV est historiquement exposé à la recherche.

| Même flux causal, émission par blocs de 4 hops | Onset F1 | Exact-K | K actif | Offset ±50 ms | Notes complètes F1 | Hauteurs par trame F1 |
|---|---:|---:|---:|---:|---:|---:|
| Natif + S675 avec secours causal différé | 62,73 % | 52,84 % | 42,91 % | 74,09 % | 46,10 % | 60,32 % |
| S704, budget 220 choisi sur CAL | 66,60 % | 60,47 % | 54,62 % | 67,97 % | 44,88 % | 70,36 % |
| S705, époque 32 choisie sur CAL | 68,29 % | 61,09 % | 54,42 % | 72,51 % | 49,10 % | 70,34 % |

S705 récupère 132 vraies attaques DEV et en perd 42, soit **+90 TP nets** ;
les FP passent de 591 à 507. Les bonnes fins appariées passent de 729 à 778,
mais le taux conditionnel passe de 74,09 à 72,51 %, car le dénominateur change.
Ce gain DEV ne constitue pas une validation extérieure.

## Contrôle HELD déjà vérifié

S704/S705 : 64 prises, quatre familles de compositions séparées du nouvel
entraînement et de CAL, 9 464 notes, 264 576 trames.

| Même flux causal | Onset F1 | Exact-K | K actif | Offset ±50 ms | Notes complètes F1 | Hauteurs par trame F1 |
|---|---:|---:|---:|---:|---:|---:|
| Natif + S675 causal | 67,30 % | 50,18 % | 43,10 % | 61,17 % | 40,59 % | 68,49 % |
| S704 | 67,80 % | 59,85 % | 55,62 % | 61,14 % | 40,87 % | 78,67 % |
| S705 | 66,94 % | 59,58 % | 54,55 % | 61,92 % | 40,83 % | 79,07 % |

S704 perd 11 Onset de notes longues précédemment reconnues sur HELD ; S705
en perd 18. S705 augmente les TP HELD (+113) mais aussi les FP (+338), ce qui
diminue son F1 Onset. Les protections de non-régression ne passent pas.

## Boucles exécutées et en cours

- **S704** : 192 caractéristiques par hauteur, acoustique native complète et
  historiques strictement passés ; nouvelles têtes Onset/maintien/release et K.
  Budgets 100, 220 et 400 arbres, avec 96 sélections par budget. CAL retient 220 :
  400 augmente légèrement l'Onset mais réduit le F1 des notes complètes.
- **S705** : réseau temporel causal, interactions entre harmoniques, têtes K
  et notes conditionnées par K prédit. Contrôles après 10, 20 et 32 époques,
  96 sélections chacun. La mémoire temporelle du réseau vaut 30 blocs passés.
  Temps des seules têtes en mode incrémental sur le runner : médiane 2,87 ms,
  p95 3,89 ms ; ce n'est pas le temps total du frontend et du microphone.
- **S706** : 96 entrées supplémentaires de spectres interpolés, contraste
  fréquentiel et évolution de phase, soit 288 entrées. Sur une sinusoïde E2,
  le maximum FFT natif donne la même valeur à E2 et F2 ; l'interpolation les
  sépare. Trois budgets 10/20/40 époques ; entraînement couvrant les cinq styles.
- **S707** : le meilleur Onset CAL atteint la borne `.60` de la première grille.
  Extension à `.90`, recherche des seuils de maintien/release, puis intégration
  S675 sur des chaînes présélectionnées. Les horloges, secours différés et le
  plafond de six notes sont causaux. Tous les choix utilisent CAL.

S704/S705 apprennent sur BN1, BN2, Funk2, Jazz1 (64 prises) ; CAL = Jazz2
(16 prises) ; HELD = Jazz3, Rock3, SS1, SS3 (64 prises). S706 apprend sur
BN1, Funk2, Jazz1, Rock3, SS1 (80 prises) ; CAL = BN2, Jazz2 (32 prises) ;
HELD = Jazz3, SS3 (32 prises). **Le HELD S706 ne se compare donc pas directement
à l'agrégat HELD S704/S705.** DEV reste le même pour toutes les familles.

Les nouvelles têtes ne sont entraînées sur aucune composition CAL/HELD/DEV.
Les têtes natives réutilisées et S675 ont une exposition TRAIN historique plus
large ; les groupes HELD et DEV ont aussi déjà été examinés par la recherche.
Ces résultats ne prétendent donc pas constituer un test externe inédit.
Les labels TEST ne sont pas ouverts.

## Différences avec les anciens pourcentages

Le 63,10 % Onset historique utilisait des émissions au hop individuel. La
référence comparable ici est 62,73 %, après disponibilité effective du bloc
de quatre hops. La tolérance Onset reste ±80 ms ; aucun événement n'est antidaté.

Le 78,08 % S675 historique provient d'une relecture de fins séparée. Son
sélecteur pouvait choisir une ancienne fin native après avoir constaté
l'absence d'un franchissement futur. Un flux causal doit au contraire émettre
un secours au moment où son délai expire. Ici le secours standard arrive
24 hops après la fin native connue, avec un plafond d'âge de 720 hops repris
du mécanisme d'essai causal. Le taux de ce contrôle DEV est 74,09 % ; il ne
doit pas être présenté comme une répétition à l'identique du 78,08 % historique.

Exact-K est toujours compté sur les notes effectivement ouvertes par le flux
évalué. Le 66,66 % du spécialiste S699, calculé indépendamment en shadow,
n'est pas réutilisé comme résultat de ces chaînes.

Le F1 des notes complètes est le F1 Onset multiplié par le taux Offset
conditionnel, avec correction des fins de référence censurées au bord du clip.
80 % d'Onset et 80 % d'Offset donnent donc environ 64 % de notes complètes,
et non 80 %. Atteindre 80 % sur la transcription complète exige davantage
sur les deux premières composantes.

## Preuves et reproduction

Les ZIP sont téléchargés avec taille et SHA-256 contrôlés. Le script
`scripts/s704_verify_emitted_artifact.py` recalcule tous les TP/FP/FN,
appariements de fins, intervalles de hauteurs et Exact-K depuis les JSONL ;
il ne désérialise aucun modèle. Tous les comptes S704/S705 coïncident avec
les rapports. Les résumés vérifiés sont conservés dans `analysis/evidence/`.

Les tests couvrent les préfixes acoustiques et temporels, l'équivalence entre
inférence séquentielle et par blocs, l'absence de crédit artificiel en fin de
flux, les délais de secours et le budget des notes réellement émises.

- [S704, exécution 39092626757](https://github.com/Andriamarosoa/note/actions/runs/39092626757)
- [S705, exécution 39092939353](https://github.com/Andriamarosoa/note/actions/runs/39092939353)
- [S706, exécution 39093239181](https://github.com/Andriamarosoa/note/actions/runs/39093239181)
- [S707, exécution 39093963095](https://github.com/Andriamarosoa/note/actions/runs/39093963095)

```bash
python scripts/s704_verify_emitted_artifact.py /chemin/resultats.zip --out /chemin/verification.json
PYTHONPATH=.:src python -m unittest test.test_s704_joint_loops test.test_s705_temporal_heads test.test_s706_sharp_spectra test.test_s707_causal_release
```
