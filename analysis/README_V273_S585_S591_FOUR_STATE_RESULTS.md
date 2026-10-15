# S585–S591 — Tête attaque/maintien : gain confirmé sur deux nouvelles sources, réattaque rapide non résolue

**Verdict du 15 octobre 2026 :** le candidat **S585 quatre états + dédup100** atteint **75,31 % de F1 NoteOn ±80 ms**, contre **72,46 % pour S578 Global28 sur les mêmes deux nouveaux morceaux** : **+2,84 points**, 8 vrais NoteOn récupérés et 4 perdus, 14 faux positifs retirés et 5 ajoutés (net −9 FP). Les deux morceaux progressent à ±80 ms.

La [CI 38679314946 est terminée avec succès](https://github.com/Andriamarosoa/note/actions/runs/38679314946). **Candidat de recherche retenu, sans promotion officielle.** L'objectif global de 80 % n'est pas atteint. Les **59,34 % de S578 sur S577** concernent cinq autres morceaux : ce n'est pas une progression de 59,34 à 75,31 sur un test constant.

## Ce qui a été entraîné et réellement vérifié

- Exactement les mêmes 28 sources et les mêmes cinq folds de compositeurs/co-compositeurs que S578. Le témoin OOF S578 reproduit exactement **TP 1676 / FP 713 / FN 1309**.
- Modèle ExtraTrees 52D à quatre états : inactive/résonance, touche ou pédale maintenue, nouvelle attaque, réattaque. Les annotations MIDI ne servent qu'aux cibles d'apprentissage et au score après gel ; aucune vérité MIDI en entrée du modèle.
- 203 757 cibles inactives/résonance, 14 297 maintenues, 1 843 nouvelles attaques, 1 020 réattaques ; 4 630 cibles voisines ambiguës exclues. **Seulement 10 vraies réattaques à moins de 100 ms** dans ces 28 sources ; la classe générale de réattaque couvre aussi des répétitions plus lentes et des notes encore soutenues par la pédale.
- Poids d'origine réutilisés après la réussite du job S585 ; pas de nouvel entraînement sur les deux morceaux cibles. Empreinte : `555cb092accb6ea0b42394280f646c90ccac0b48746040811ac57da382837d36`.
- Trois politiques décidées avant les annotations : S578, quatre états avec dédup100, quatre états autorisant la répétition sous 100 ms si P(réattaque)≥0,5 et P(attaque totale)≥0,7. Basic Pitch original conservé.

| Développement sur les 28 sources, validation par blocs de compositeurs | TP | FP | FN | F1 ±80 ms |
|---|---:|---:|---:|---:|
| S578 Global28 | 1 676 | 713 | 1 309 | 62,37 % |
| S585 quatre états + dédup100 | 1 641 | 382 | 1 344 | **65,54 %** |
| S585 + autorisation de réattaque | 1 642 | 402 | 1 343 | 65,30 % |

Le gain de développement n'est pas uniforme : 19 morceaux progressent, 7 régressent, 2 sont à égalité. Le médium gagne 13 TP et perd 97 FP (+4,09 points F1) ; l'aigu perd 18 TP et 15 FP (−1,17 point). Les nouvelles cibles et les négatifs d'entraînement changent ensemble : cette expérience ne démontre pas que la seule séparation en quatre classes explique tout le gain.

## Pourquoi seulement deux nouvelles sources ?

Le premier run a arrêté la sélection initiale à cinq compositeurs **avant tout nouveau WAV ou MIDI**. Après exclusion des 33 fichiers précédents et de leurs 37 composantes de compositeurs, seuls Enescu et Moussorgski restaient dans les métadonnées MAESTRO, tous les splits confondus. Leurs fichiers dépassaient la limite technique initiale de durée.

L'[amendement préalable](README_V273_S585_S591_FOUR_STATE_PREREGISTRATION.md) conserve toutes les exclusions : un fichier par compositeur, ordre SHA inchangé, plafond WAV 500 Mo et durée maximale 2 300 s. Extraits inchangés : **12 secondes à partir de 15 secondes**, soit 24 secondes au total.

| Nouveau morceau, split officiel MAESTRO train | Vraies notes MIDI40..88 | S578 F1 ±80 ms | S585 dédup100 F1 ±80 ms |
|---|---:|---:|---:|
| Moussorgski — *Pictures at an Exhibition*, enregistrement 2004 | 47 | 79,55 % | **85,39 %** |
| Enescu — *Suite Op. 10*, enregistrement 2006 | 180 | 70,55 % | **72,50 %** |

Ce sont deux nouvelles sources pour les modèles entraînés ; aucun audit exhaustif des usages dans toutes les anciennes branches n'est revendiqué. Ce n'est pas un benchmark officiel MAESTRO TEST, ni une validation multi-instruments. Deux sources sont insuffisantes pour certifier une généralisation. Le bootstrap à deux sources reste descriptif et n'est pas une preuve de significativité générale.

## Test indépendant complet

Le vrai V88 original, Basic Pitch et les 49×11 caractéristiques acoustiques ont traité les WAV. Les trois politiques ont ensuite été figées dans l'artefact **S590**, avant que le job distinct **S591** ouvre les MIDI. Les **340 sites V88, dont K0**, sont conservés. Tous les faux événements et toutes les notes manquées comptent ; appariement 1:1 à hauteur exacte.

| Politique sur les mêmes 227 vraies notes | TP | FP | FN | Précision | Rappel | F1 ±80 ms |
|---|---:|---:|---:|---:|---:|---:|
| S578 Global28 | 150 | 37 | 77 | 80,21 % | 66,08 % | 72,46 % |
| **S585 quatre états + dédup100** | **154** | **28** | **73** | **84,62 %** | **67,84 %** | **75,31 %** |
| S585 + autorisation de réattaque | 154 | 29 | 73 | 84,15 % | 67,84 % | 75,12 % |

| Tolérance NoteOn, hauteur exacte | S578 | S585 dédup100 | Gain |
|---|---:|---:|---:|
| ±30 ms | 62,32 % | 63,57 % | +1,25 point |
| ±50 ms | 68,60 % | 70,90 % | +2,31 points |
| ±80 ms | 72,46 % | 75,31 % | +2,84 points |

La hausse globale existe aux trois tolérances, mais **Enescu régresse légèrement à ±30 ms** (58,28 → 58,13 %). Ne pas présenter tous les temps d'attaque comme améliorés.

| Registre | Δ TP | Δ FP | S578 F1 | S585 dédup100 F1 |
|---|---:|---:|---:|---:|
| Grave, MIDI40–59 | −2 | −2 | 72,99 % | **72,18 % : régression** |
| Médium, MIDI60–71 | +2 | −5 | 71,79 % | 75,00 % |
| Aigu, MIDI72–88 | +4 | −2 | 73,17 % | 80,95 % |

## Audit des régressions et des corrections

Les huit notes récupérées sont MIDI65/68/80 chez Moussorgski et MIDI61/70/73/76/78 chez Enescu. Les quatre notes perdues sont toutes chez Enescu. Au site V88 qui permettait à S578 de les reconnaître, le nouveau modèle les fait passer sous le seuil fixe 0,7 :

| Note perdue | Temps de vérité dans l'extrait | P attaque S578 | P attaque S585 | P maintien S585 |
|---|---:|---:|---:|---:|
| MIDI45 / A2 | 6,393 s | 0,766 | 0,496 | 0,493 |
| MIDI54 / F♯3 | 1,523 s | 0,716 | 0,567 | 0,418 |
| MIDI68 / G♯4 | 3,289 s | 0,713 | 0,650 | 0,317 |
| MIDI69 / A4 | 7,513 s | 0,758 | 0,669 | 0,310 |

**Cause immédiate vérifiée :** les probabilités d'attaque baissent sous 0,7 ; une part importante est attribuée au maintien. Ces quatre pertes ne viennent pas d'une suppression automatique à 100 ms, puisque le candidat est déjà rejeté par la tête. Ce constat ne prouve pas à lui seul la cause acoustique profonde. Aucun seuil n'a été ajusté pour récupérer ces cas après lecture des annotations.

L'audit séparé réconcilie **14 faux positifs retirés, 5 ajoutés, 23 conservés**, ainsi que **8 TP récupérés et 4 TP perdus**. Les événements et probabilités sont conservés [dans les preuves](evidence/v273-s585-s591/postlabel_probability_diagnosis.json). Le diagnostic est reproductible par `scripts/audit_s591_lost_note_probabilities.py` après récupération des artefacts S589/S590/S591 ; il est postérieur au score, pas un nouveau test indépendant.

## Décision et suite justifiée

- **Conserver S585 quatre états + dédup100 comme challenger expérimental**, avec S578 comme contrôle et référence de comparaison. Les nouveaux états améliorent ici le filtrage des notes sans empêcher toutes les pertes.
- **Ne pas retenir l'autorisation apprise de réattaque** : développement +1 TP / +20 FP face à la même tête stricte ; nouveau test +0 TP / +1 FP. **Aucune vraie réattaque sous 100 ms n'est présente dans les deux nouveaux extraits.** Ce test ne valide donc pas la reconnaissance des répétitions rapides.
- Le prochain apprentissage doit traiter les vraies attaques masquées par le maintien, en particulier les notes graves, avec davantage de vraies réattaques et un corpus externe à ces cohortes MAESTRO désormais exposées. Ne pas retoucher les seuils sur Enescu/Moussorgski puis présenter leur score comme indépendant.
- **Exact-K/S80, poids officiels et Live inchangés dans le code. Aucun rejeu complet Exact-K/GuitarSet ou test de latence Live n'a été réalisé dans S585–S591.** Ces conditions restent nécessaires avant toute intégration ou promotion.

## Preuves conservées

[Manifeste des runs, modèles et fichiers SHA](evidence/v273-s585-s591/manifest.json) ; [rapport d'entraînement](evidence/v273-s585-s591/training_report.json) ; [gel avant MIDI](evidence/v273-s585-s591/prediction_freeze_report.json) ; [score indépendant](evidence/v273-s585-s591/independent_score_report.json) ; [audit note par note](evidence/v273-s585-s591/individual_note_audit.json).

Le premier run est globalement en échec à cause de la sélection initiale impossible, mais son **job d'entraînement S585 est réussi**. Le second run est **entièrement réussi**, réutilise les mêmes poids et exécute réellement le gel puis le score final. Il n'y a pas de second entraînement ni de fusion dans la branche principale.
