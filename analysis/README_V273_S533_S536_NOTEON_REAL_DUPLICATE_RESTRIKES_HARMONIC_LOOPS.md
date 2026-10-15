# S533–S536 — Boucles NoteOn : doublons temporels, harmoniques et vraies réattaques

**15 octobre 2026 · recherche seulement · aucune promotion ni modification de S80, Exact-K, V88, freeze_local_combo ou Live.**

## Contrat de mesure
Même cohorte historique : trois extraits MAESTRO de 12 secondes, 330 groupes MIDI NOTE_ON et 416 notes (hauteurs) annotées, 575 candidats temporels V88 originaux. Les modèles par hauteur S528 ont été entraînés sur deux morceaux et évalués sur le troisième, mais ces trois morceaux étaient déjà exposés aux développements précédents. Pour chaque vraie note, exactitude de hauteur ET attaque à plus ou moins 80 ms avec appariement un-à-un par hauteur. Chaque note prédite sans appariement est FP et chaque note MIDI non détectée est FN. Garder toutes les notes déjà nées dans Basic Pitch ; agir uniquement sur les ajouts.

S530 baseline expérimental : **213 TP / 206 FP / 203 FN = 51,02 % F1 NOTE_ON**. Ce n'est PAS Exact-K GuitarSet et PAS une validation aveugle.

## S533 — neuf variantes physiques sans réentraînement

[Run réussi S533 #38652938992](https://github.com/Andriamarosoa/note/actions/runs/38652938992). Hypothèses : dédupliquer une même nouvelle hauteur déjà émise sous 40/80 ms ; inhiber une hauteur suspectée d'être l'harmonique +12/+19/+24 demi-tons d'une fondamentale inférieure lorsque l'attaque et la nouveauté de l'harmonique sont toutes deux beaucoup plus faibles. Les notes nées dans BP restent toujours intactes.

| Politique | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|
| Sans ajout (référence S530) | 213 | 206 | 203 | 51,02 % |
| Doublon même hauteur à 40 ms | 213 | 206 | 203 | 51,02 % |
| Doublon même hauteur à 80 ms | 213 | 150 | 203 | **54,69 %** |
| Inhibition harmonique modérée seule | 213 | 204 | 203 | 51,14 % |
| Inhibition harmonique forte seule | 212 | 198 | 204 | 51,33 % |
| Doublon 80 ms + inhibition modérée | 213 | 148 | 203 | 54,83 % |
| Doublon 80 ms + inhibition forte | 212 | 143 | 204 | **54,99 %** |

Conclusion : l'erreur la plus importante dans cette cohorte est la génération répétée de la **même hauteur**, plutôt qu'une simple erreur de nombre de notes K. La déduplication 80 ms élimine 56 FP sans perdre de TP sur ces trois extraits. Le contrôle harmonique apporte moins et une suppression forte perd une vraie note.

## S534 — audit temporel indépendant et attaques répétées MIDI

[Run réussi S534 #38653138575](https://github.com/Andriamarosoa/note/actions/runs/38653138575). Réévaluation SHA-identique de S533 sur tolérances 30, 50 et 80 ms, ledger des notes individuelles supprimées et contrôle des vraies attaques successives de même hauteur.

| Politique | F1 ±30 ms | F1 ±50 ms | F1 ±80 ms | Vraies notes de même hauteur rejouées dans la fenêtre |
|---|---:|---:|---:|---:|
| Sans dedup | 40,72 % | 47,43 % | 51,02 % | — |
| Dedup 60 ms | 43,17 % | 50,57 % | 54,41 % | 0 |
| Dedup 80 ms | 42,88 % | 50,58 % | 54,69 % | **2** |
| Dedup 100 ms | 43,01 % | 50,78 % | 55,18 % | **3** |
| Dedup 80 ms + harmonique fort | 43,32 % | 50,84 % | 54,99 % | 2 |

Avertissement : ces 2 ou 3 attaques MIDI authentiques très rapprochées pourraient être supprimées par une politique de déduplication aveugle sur d'autres morceaux. Une simple fenêtre de 100 ms n'est pas considérée sûre.

## S535 — six experts de hauteurs réellement entraînés sur tous les candidats, dont les non-appariés

[Run réussi S535 #38653432182](https://github.com/Andriamarosoa/note/actions/runs/38653432182) après correction de l'erreur de somme NumPy du premier run. Deux familles ExtraTrees, chacune avec trois entraînements distincts (deux morceaux train, troisième test), visant toutes les 49 notes sur chaque candidat V88 ; négatifs issus des 253 candidats temporels non appariés à un groupe MIDI vrai à ±80 ms. Ces négatifs relèvent de la **convention de scoring NoteOn** et peuvent correspondre physiquement à des doublons ou à des sons soutenus.

| Variante | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|
| Raw audio 11 indices p≥0,50 | 251 | 599 | 165 | 39,65 % |
| Raw audio 11 indices p≥0,70 | 185 | 202 | 231 | 46,08 % |
| Compétition 38 indices harmoniques p≥0,50 | 242 | 510 | 174 | 41,44 % |
| Compétition 38 indices harmoniques p≥0,70 | 175 | 188 | 241 | 44,93 % |

**Rejet : aucune amélioration sur S530 ni S533.** Ce test prouve seulement que le nouveau classificateur n'est pas un candidat viable sur ce protocole.

## S536 — dix politiques de protection des vraies réattaques rapides

[Run réussi S536 #38653634175](https://github.com/Andriamarosoa/note/actions/runs/38653634175). Une seconde attaque d'une note déjà émise peut être acceptée dans la fenêtre de déduplication si elle présente une montée et une nouveauté acoustiques nettement plus fortes que le dernier événement de cette hauteur. Les politiques intègrent aussi le veto harmonique faible-fondamentale. Tests synthétiques : la réattaque redevient autorisée ; **sur ces trois vrais extraits aucune réattaque candidate n'a été libérée par les conditions acoustiques testées**, donc aucun gain empirique de ce garde-fou.

| Politique | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|
| Dedup 80 ms | 213 | 150 | 203 | 54,69 % |
| Dedup 100 ms | 213 | 143 | 203 | 55,18 % |
| Dedup 80 ms + harmonique fort | 212 | 143 | 204 | 54,99 % |
| **Dedup 100 ms + harmonique fort** | **212** | **136** | **204** | **55,50 %** |
| Fenêtres 80/100 avec réattaque conditionnelle | Identiques aux règles strictes correspondantes | | | 54,69 à 55,50 % |

Le maximum **55,50 %** gagne +4,48 points vs S530, mais perd une note vraie et a été choisi après de multiples essais sur des morceaux historiquement connus. Ne jamais l'appeler nouvelle référence ou résultat général GuitarSet.

## Décision et prochaine étape obligatoire

- **Candidat architectural prudent :** la suppression de faux NOTE_ON par déduplication à 80 ms, avec le statut d'expérience. Score historique 54,69 %, zéro TP perdu sur cette petite cohorte.
- **Candidat exploratoire de score :** S536 avec 100 ms et veto harmonique fort, 55,50 % seulement sur cette cohorte. Il reste un risque d'effacer de véritables notes rejouées.
- **S535 :** rejeté ; aucun nouveau modèle déployé.
- **Référence officielle S80 / V27.3 / Exact-K / Live : inchangée.**

La suite scientifiquement utile nécessite une **cohorte MAESTRO / GuitarSet / guitare à morceaux et interprètes réellement inédits**, choix gelé avant la lecture des annotations, et une validation complète des doublons, vraies réattaques rapides, accords polyphoniques, notes NOTE_ON et latence Live. Nouvel entraînement d'une tête temporelle de réattaque réellement discriminante requis. **80 % F1 NOTE_ON non atteint.**


## S537 — nouvelle cohorte indépendante de développement préenregistrée sans étiquettes

[CI S537 SUCCESS #38654030814](https://github.com/Andriamarosoa/note/actions/runs/38654030814). Après les boucles rétrospectives, cinq morceaux du split MAESTRO officiel test ont été sélectionnés par **ordre SHA fixe des noms de fichiers**, avec cinq compositeurs différents et sans chevauchement des trois compositeurs ou fichiers S469. **Seul le CSV public de métadonnées** a été consulté. Aucun WAV, MIDI, note réelle, prédiction, poids de modèle ou score n'a été ouvert pour sélectionner ces cinq morceaux.

Le premier essai S537 avec la fenêtre de durée historique 55–145 secondes a échoué : pas assez de compositeurs distincts admissibles. Sans examiner l'audio ni les annotations, le filtre de durée **40–240 s** a été défini pour rendre la sélection possible. Le nouveau run a réussi avec **36 fichiers candidats admissibles** après exclusion, dont cinq sélectionnés sous SHA de métadonnées officielles **cb88a64a12971a995dd41852aa3a33879501cf70d01b9a6feb637f4ad8d8ddff**.

La preuve de non-chevauchement porte sur S469 seulement ; elle **ne constitue pas une preuve que ces cinq fichiers n'ont jamais été utilisés ailleurs dans tout le dépôt**. Il faut auditer cette provenance avant de qualifier S537 de jeu vraiment aveugle. S537 ne contient pas d'évaluation de note ni de score et ne change aucune référence. **Les annotations MIDI de ces nouveaux morceaux doivent rester scellées jusqu'au gel complet des décisions WAV/NOTE_ON de la prochaine chaîne.**
