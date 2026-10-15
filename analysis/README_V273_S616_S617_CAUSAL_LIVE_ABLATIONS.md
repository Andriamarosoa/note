# S616–S617 — Boucles LIVE NOTE natives, preuves et décision

Date : 2026-10-15. Statut : **expérimental, aucune promotion**.

## Objectif et intégrité

Conserver **strictement** le vrai S80/S397, les têtes de notes historiques,
les modèles S578/S585/S601 et le checkpoint strict-Live S614. Améliorer la
chaîne causale PCM → têtes de comptage / hauteur / attaque / maintien /
relâchement → sélections → NoteOn / NoteOff sans futur audio.

- Référence S614 (test **déjà publié**) : **69,03 % F1 NoteOn ±80 ms**, sur
  22 enregistrements, 2 375 attaques, 1 647 TP, 750 FP, 728 FN.
- Checkpoint S614 inchangé SHA256 :
  `7ccf046e3a1870eaca3a108e6f9565298aeeadf43732a2f83559542c03f5f17f`.
- Les expérimentations suivantes ont uniquement ouvert les groupes
  **train/dev existants** ; **aucun réglage n'a été fait sur le test S614**.
- Aucun code ou modèle de reconnaissance du dépôt `midi` n'a été importé.

## S616 A — Quinze ablations sur les 5 URMP de développement

[Tous les tests et métriques](https://github.com/Andriamarosoa/note/actions/runs/38781924089)
(rapport JSON dans l'artefact `S616-live-dev-ablation-only`).

La chaîne **exacte S614** a été reproduite sans divergence d'événements, et
le résultat historique a été retrouvé : **243 TP, 149 FP, 256 FN,
54,5455 % F1**. Tous les tests de causalité et de préservation sont réussis.

| Variante | F1 | TP | FP | Écart en points |
|---|---:|---:|---:|---:|
| S614 inchangé | 54,5455 % | 243 | 149 | 0 |
| Pénalité harmonique 0,6 | 54,6680 % | 243 | 147 | +0,1227 |
| Récupération d'attaque 0,6 | **55,0725 %** | **247** | 151 | **+0,5270** |
| Fusion pénalité 0,5 / récupération 0,4 | 54,9719 % | 246 | 150 | +0,4266 |
| Libération après quatre trames | 54,5455 % | 243 | 149 | 0 |

La libération à **quatre trames** améliore la proportion conditionnelle de
NoteOff dans la tolérance, **64,20 % → 66,26 %**, mais ces notes sont un
ensemble déjà reconnu par le modèle. Aucune revendication de score MIDI complet.

## S616 B — Vérification sur les 12 GuitarSet de développement

Les quatre variantes sont figées **avant** l'ouverture de ces résultats. Ce
jeu a déjà servi à la sélection de S614, ce qui **n'en fait pas une nouvelle
validation indépendante**.

[CI réussie / artefact détaillé](https://github.com/Andriamarosoa/note/actions/runs/38782639881).

La référence est reproduite : **998 TP, 585 FP, 582 FN, 63,1046 % F1**.

| Variante | F1 | TP | FP | Écart en points |
|---|---:|---:|---:|---:|
| S614 original | 63,1046 % | 998 | 585 | 0 |
| Pénalité harmonique 0,6 | 63,1379 % | 996 | 579 | +0,0332 |
| Récupération d'attaque 0,6 | 63,1612 % | 1 003 | 593 | +0,0566 |
| Fusion pénalité 0,5 / récupération 0,4 | **63,2543 %** | 1 001 | 584 | +0,1497 |
| NoteOff après quatre trames | 63,0847 % | 998 | 586 | −0,0199 |

Les améliorations F1 ne sont ni suffisamment grandes ni uniformes pour
promouvoir le sélecteur. Une réduction brute des FP peut aussi perdre de
vraies notes, par exemple la pénalité 0,6 : −6 FP, **−2 TP** sur GuitarSet.

## S617 — Deuxième branche d'attaque, sans apprentissage

[Tests et ablations réels](https://github.com/Andriamarosoa/note/actions/runs/38783079409).

19 politiques ont été comparées sur les 5 URMP dev : reproduction du contrôle
S614, reproduction de la meilleure version S616, différents seuils d'attaque,
de maintien et de variation de fondamentale, et variantes combinées.

- **Meilleure option** : reste **S616 récupération 0,6**, à **55,0725 %**.
- Nouveau pont très sélectif : **54,7085 %** (244 TP, 149 FP).
- Pont permissif (`bridge_0.60_0.10_0.05`) :
  **53,2508 %**, 258 TP et **212 FP**.
  Il gagne **15 TP** mais crée **63 FP** supplémentaires ;
  une régression de **−1,2947 point**.
- Le test de causalité par mutation du futur est réussi. La chaîne historique
  S614/S80 n'a pas changé. Aucun score nouveau sur le test S614 n'est revendiqué.

Le pont confirme la limite d'une sélection reposant seulement sur les
probabilités déjà produites : lorsqu'une fondamentale réelle manque en
amont, relâcher les conditions de sélection récupère aussi des harmoniques
et des faux déclenchements.

## Contrôle complémentaire audio réel de l'utilisateur

Rejeu local **du vrai S614** sur les 30 premières secondes du MP3 fingerstyle
`Until I Found You` transmis dans le chat : **325 NoteOn**, 325 NoteOff,
7,44 s de calcul pour 30 s d'audio sur cette machine, aucun fichier de
vérité terrain. L'expérience S601 hors ligne antérieure produit **166 NoteOn**
sur la même durée. Leurs pipelines étant différents, **ce contraste n'est
pas une mesure F1 ou une preuve de supériorité**.

## Verdict et prochaine expérience S618

**REJET DES NOUVELLES SÉLECTIONS COMME RÉFÉRENCE**, conservation permanente
des variantes positives et négatives dans les scripts, tests et artefacts.

Le prochain essai significatif nécessite une **nouvelle tête acoustique
apprise**, plutôt qu'un nouveau seuil :

1. Sur les 53 enregistrements historiques `train`, entraîner une tête
   distincte qui classe de vraies **attaques de fondamentale**, avec
   **négatifs difficiles** : énergie harmonique partagée, attaques
   trompeuses et tenues sans nouvelle note.
2. Tester la nouvelle tête en **union de candidats** avec le modèle S614
   plutôt qu'en remplaçant ses têtes, et sélectionner via les 17 groupes
   `dev` avec audit de chaque TP/FP/FN et par source.
3. Préserver le mode strict causal : caractéristiques à bord droit,
   aucun centrage futur, état/mémoire bornés, latence et budget par bloc
   mesurés.
4. Geler modèle et règles avant d'ouvrir des **enregistrements réellement
   nouveaux** hors du test S613/S614 déjà publié.
5. Promouvoir seulement si le F1, les régressions, les NoteOff et la
   latence satisfont explicitement les garde-fous. **80 % LIVE non atteint**.

## Reproduction

```bash
export PYTHONPATH=.:src
python -m unittest -v test.test_strict_live test.test_s616_live_guard test.test_s617_live_attack_bridge
python -m scripts.s616_live_dev_ablation --out /tmp/S616
python -m scripts.s616_live_guitar_transfer --out /tmp/S616_Guitar
python -m scripts.s617_live_dev_scan --out /tmp/S617
```

Les dépendances et les identifiants de corpus sont figés dans les fichiers
de workflow et dans `analysis/evidence/v273-s613-strict-live/protocol.json`.
L'intégrité de l'ancien checkpoint et la concordance des références sont
vérifiées pendant chaque audit. Aucune commande ci-dessus ne remplace S80.
