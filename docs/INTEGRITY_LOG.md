# Journal d’intégrité expérimentale

## 2026-08-27 — accès technique involontaire au joueur 05

Pendant l’inventaire initial, deux contrôles exploratoires ont été lancés sur
l’archive complète avant l’application du filtre de split :

- les 44 octets d’en-tête de chaque WAV ont été lus, y compris pour le joueur
  05 ; aucun échantillon PCM n’a été lu ;
- un contrôle de schéma a désérialisé les JAMS globaux, y compris ceux du
  joueur 05.

Aucune annotation individuelle, métrique de modèle, performance, prédiction ou
comparaison propre au joueur 05 n’a été affichée ou utilisée pour une décision.
Les agrégats globaux issus de ces contrôles sont écartés du développement.

Conséquences appliquées immédiatement :

1. toute inspection suivante est limitée aux joueurs 00 à 04 ;
2. le code de split refuse désormais le joueur 05 sans option de contournement ;
3. le joueur 05 ne sera pas présenté comme « jamais ouvert » ;
4. avant l’évaluation finale, le protocole devra décider explicitement de la
   remédiation scientifique (évaluateur propre et audité, ou nouveau test
   réellement tenu à l’écart).

