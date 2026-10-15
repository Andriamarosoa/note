# S613–S614 — NOTE autonome, causal, audio → notes

Le nouveau mode `strict-live` fonctionne depuis le PCM continu, avec les caractéristiques, têtes apprises et sélections de NOTE. Il émet des NoteOn et des NoteOff sans événements V88, K ou observations BP précalculés. Il accepte WAV, stdin PCM16 et microphone via PortAudio, avec sortie MIDI optionnelle.

**Le test réservé atteint 69,03 % de F1 NoteOn ±80 ms. L’objectif de 80 % n’est pas atteint.** Les 84 comparaisons ont été faites avant l’ouverture des annotations du test. Aucun ancien score hors ligne à 80–85 % n’est présenté comme un score de cette version Live.

## Architecture et conservation de S80

La branche améliore les composants natifs avec un nouvel apprentissage causal :

1. Fenêtres FFT à bord droit 2048/4096/8192, pas de 256 échantillons, 44,1 kHz. Spectre harmonique à huit partiels, compétition avec les fondamentales voisines et différences temporelles, dans la continuité des idées S511/S562/S569 et du frontal causal V280.
2. 126 caractéristiques par hauteur MIDI40–88 ; historique limité à 32 trames. Aucun indicateur BP. Aucune normalisation calculée sur le morceau entier.
3. Une tête MLP de comptage actif K0..6 fournit sept probabilités en entrée des têtes de hauteur, dans la logique de conditionnement S402. Les entrées des têtes de hauteur ont donc 133 dimensions. **Ce K actif diffère du nombre de nouvelles notes de S80.**
4. Trois sorties par hauteur : attaque, maintien, relâchement. Une famille neuronale et une famille d’arbres sont entraînées uniquement sur le groupe d’apprentissage. La sélection retenue combine leurs probabilités géométriquement : poids neuronal 0,75, poids arbres 0,25.
5. Sélection causale : seuil attaque 0,60, maintien 0,50, réarmement 0,20, fermeture après cinq trames de relâchement, réfractaire par hauteur de dix trames (58,05 ms), maximum six notes actives. Pas de confirmation additionnelle retenue. Le veto dur de K n’est pas retenu, mais le conditionnement appris par K reste actif.
6. Suivi borné des notes, NoteOff avant réattaque, fermeture de toutes les notes à l’arrêt. La fermeture de contrôle en fin de flux est distinguée d’une fin acoustique.

**Aucun code ni poids du moteur `midi` n’est repris. S80/S397 et les 44 fichiers historiques protégés restent identiques.** S611 et S612 restent accessibles. La chaîne complète S80 ne devient pas automatiquement causale : cette branche est un complément expérimental, pas une nouvelle certification des sélections S69–S80. Les anciens checkpoints S578/S585 ne sont pas réutilisés avec des entrées artificiellement mises à zéro.

## Données et séparation

- Apprentissage : **53 enregistrements**, 36 GuitarSet MIC et 17 ensembles URMP, au plus les 24 premières secondes de chacun.
- Développement : **17 autres enregistrements**, 12 GuitarSet et 5 URMP, **2 079 notes** évaluées. Tous les choix de modèle, de combinaison et de seuil sont faits ici.
- Test réservé aux nouveaux modèles : **22 enregistrements, 478,62 secondes**, 12 GuitarSet du joueur 05, 5 URMP et les 5 morceaux IDMT Dataset3 du registre S606.
- GuitarSet : séparation par famille de grille (par exemple BN1), avec tous les tempi, tonalités et rôles solo/accompagnement dans le même groupe ; joueurs 00–03 à l’apprentissage, 04 au développement et 05 uniquement au test. URMP : séparation par titre, regroupant les arrangements d’un même titre. Les identités et sources sont dans le protocole versionné.
- Les deux jeux de prédictions, modèle sélectionné et référence neuronale, sont enregistrés et hachés **avant** la lecture des annotations test. Une implémentation indépendante par chemins augmentants confirme les 88 comptages TP du scoreur hongrois.

Les corpus ont déjà été examinés dans les travaux historiques du dépôt. Il s’agit d’un test tenu hors du nouvel apprentissage et des nouveaux réglages, pas d’une prétention à des sources universellement inconnues. IDMT Dataset3 couvre un instrument ; cela ne valide pas les sept guitares de la collection. URMP provient du miroir préparé à 16 kHz, rééchantillonné pour l’entrée 44,1 kHz ; IDMT stéréo est réduit en mono quand nécessaire. Le microphone n’effectue pas ces préparations de corpus.

Périmètre : 2 411 NoteOn annotés dans les extraits ; 28 hors MIDI40–88 sont exclus et 8 unissons exactement simultanés sont regroupés. **2 375 NoteOn uniques** sont évalués. Les faux événements prédits dans ce périmètre restent tous comptés.

## Boucles de développement

| Famille | Comparaisons | Meilleur F1 ±80 ms |
|---|---:|---:|
| Neuronal, états d’apprentissage 3/8/16/28 + sélections affinées | 48 | 58,90 % |
| Arbres seuls, mêmes entrées causales et mêmes sources | 12 | 56,38 % |
| Combinaison des têtes, poids et seuils sélectionnés sur développement | 24 | **61,22 %** |

Le choix final et ses empreintes précèdent les annotations réservées. Aucun réglage supplémentaire n’est effectué après les résultats ci-dessous. Les arbres sont exportés vers ONNX pour réduire le coût d’inférence. Un écart de conversion sur les seuils doubles a été corrigé par un arrondi vers le bas garantissant la même branche pour les entrées float32 ; les tests de parité couvrent les 512 premières trames de chaque source de développement (50,53 s au total), puis la sélection combinée est évaluée sur toutes les trames via le runtime exporté. Les sources, scripts, poids et toutes les variantes sont conservés.

## Test réservé : NoteOn, hauteur exacte et appariement 1:1

Les temps scorés sont les instants de disponibilité du PCM, sans antidatation des notes vers une attaque reconstruite. Ils n’incluent pas le temps du pilote ou du synthétiseur. Le temps de calcul est mesuré séparément.

| Corpus | Enregistrements | Notes | Référence neuronale ±80 ms | Sélection ±80 ms | Sélection ±50 ms |
|---|---:|---:|---:|---:|---:|
| GuitarSet | 12 | 1512 | 73.23 % | **73.72 %** | 70.01 % |
| URMP | 5 | 775 | 56.22 % | **60.53 %** | 44.00 % |
| IDMT | 5 | 88 | 59.53 % | **60.70 %** | 43.78 % |
| **Total** | **22** | **2 375** | **67,33 %** | **69,03 %** | **60,73 %** |

À ±80 ms : **TP 1 647, FP 750, FN 728**, contre TP 1 525, FP 630, FN 850 pour la référence neuronale. La sélection récupère **150 vraies notes** et en perd **28** : gain net de 122 TP, avec 120 FP supplémentaires. **13 enregistrements progressent, 9 régressent**. Les détails de chaque perte sont conservés ; il n’y a pas de prétention à une absence de régression musicale.

Les résultats concernent des extraits de durée bornée et ce protocole précis. Ils ne sont pas directement comparables aux cohortes S578/S585/S601, qui utilisent d’autres extraits et des entrées différentes.

## Latence, NoteOff et limites

Sur les NoteOn correctement appariés à ±80 ms, le délai de disponibilité médian est **25,40 ms**, p95 **60,57 ms**. Sur GuitarSet : médiane **21,02 ms**, p95 **49,77 ms**. Les notes manquées, fausses ou trop tardives ne sont pas incluses dans cette statistique conditionnelle.

Sur le contrôle réel GuitarSet de 10 s, 1 715 blocs mesurés après huit blocs de chauffe : calcul complet frontal + têtes + sélection **1,20 ms médian, 1,59 ms p95, 2,17 ms p99, 5,35 ms maximum**, pour un bloc de **5,805 ms** ; aucun dépassement dans ce contrôle. D’autres mesures sous charge ont connu des pointes plus élevées : ce contrôle ne garantit pas le pire cas d’un ordinateur ou d’un pilote.

**Zéro lookahead est vérifié ; la latence de reconnaissance n’est pas nulle.** La grande fenêtre contient du passé, pas une attente imposée après une attaque. Les fenêtres, les réarmements et le délai de fermeture influencent toujours la réaction acoustique.

Les NoteOff sont opérationnels mais encore imprécis : parmi **1 622 NoteOn correctement appariés dont la fin de référence est observable**, **944 fins** respectent la tolérance max(50 ms, 20 % de durée), soit **58,20 % conditionnellement**. Cinq fins restent forcées ou manquantes. L’erreur acoustique de fin a une médiane de +16,63 ms et des queues larges (p10 −202,40 ms, p95 +177,12 ms). **Ce n’est pas un score F1 MIDI complet.**

Les modules microphone et sortie MIDI sont raccordés, mais aucun matériel n’a été mesuré ici : PortAudio et le séquenceur ALSA ne sont pas disponibles dans l’environnement. La latence acoustique microphone → synthétiseur reste à mesurer sur la machine cible. La file est limitée à deux blocs et un dépassement arrête les notes ; le programme ne garantit pas l’ordonnancement d’un système temps réel matériel.

## Vérifications et reproduction

- 42 tests : contrats natifs historiques et récents, causalité par mutation du futur, blocs de tailles différentes, PCM fragmenté, erreurs de continuité, mémoire bornée, réattaques, fermeture et silence.
- Audio réel : événements identiques avec 256 et 997 échantillons par bloc ; trois mutations du futur laissent les préfixes de décisions inchangés ; CLI WAV identique au moteur sur 140 événements.
- 44 fichiers protégés inchangés ; modèles récents, S80/S397 et Live historique conservés. La branche principale n’est pas modifiée.
- 88 vérifications indépendantes des appariements, sans désaccord. Modèles et émissions test figés avant annotations.

[Commandes microphone, MIDI, stdin et WAV](../LIVE.md). Les dépendances d’inférence sont dans `requirements-live-strict.txt`, les périphériques optionnels dans `requirements-live-devices.txt`. L’entraînement/export utilise `requirements-live-training.txt` et PyTorch CPU 2.5.1.

```bash
export PYTHONPATH=.:src
python -m unittest -v test.test_strict_live
python -m scripts.s611_fetch_native_fixture --out data/S613/real_guitar.wav
python -m scripts.s613_validate_strict_live --wav data/S613/real_guitar.wav --out logs/S613
```

Pour rejouer l’expérience, copier `analysis/evidence/v273-s613-strict-live/protocol.json` dans un nouveau répertoire de travail `data/S613/protocol.json`, puis préparer les groupes séparément avec `scripts.s613_prepare_strict_live`. `scripts.s613_train_strict_live --stage train`, puis `--stage refine`, reproduit les 48 boucles neuronales. `scripts.s614_train_causal_tree_heads`, `scripts.s614_export_causal_trees` et `scripts.s614_mix_causal_heads` produisent les autres variantes. Les arguments `--help` nomment les répertoires attendus. Geler les deux prédictions par `--stage freeze-test` avant `--stage score-test`. Le test historique étant désormais publié, ses prochains rejeux seront des contrôles de régression et non de nouvelles validations aveugles.

Les preuves machine sont dans [evidence/v273-s613-strict-live](evidence/v273-s613-strict-live), dont `protocol.json`, les trois journaux de boucles, les deux jeux d’émissions, les audits individuels, le bilan des régressions et les empreintes des checkpoints. Les anciens états d’apprentissage sont gardés dans `model/experimental/strict_note` ; le modèle Live retenu est dans `model/live/strict_note`.

Les émissions et audits individuels volumineux sont conservés en `.json.gz`, sans modification des octets décompressés. `archived_ledgers.json` fournit les SHA256 avant et après compression ; les empreintes originales restent celles des gels et rapports.
