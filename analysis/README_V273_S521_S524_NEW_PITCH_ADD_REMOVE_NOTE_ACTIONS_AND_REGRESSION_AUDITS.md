# S521–S524 — Trois boucles de sélections neuronales pour les notes nouvelles : bilan mesuré, aucune promotion

**15 octobre 2026.** L'utilisateur demande la poursuite du développement, puis une promotion de référence seulement après une validation solide. Aucun des modèles S521–S524 ne modifie `S80`, les décisions historiques K0–K6, les onsets/offsets, `freeze_local_combo` ou le moteur Live.

La question n'est pas « comment augmenter le nombre de têtes », mais **comment apprendre l'action ajouter une VRAIE hauteur absente de Basic Pitch sans effacer un vrai accord polyphonique ou générer des harmoniques imaginaires**.

## Architecture source et garanties

Toutes ces boucles emploient les vraies entrées d'audio originales :
- S505 : 45/45 attaques XML IDMT réelles appariées au groupe canonique et 220/322 attaques MIDI MAESTRO appariées ; 102 attaques MAESTRO ne possèdent pas de groupe canonique et sont **toujours comptées comme manquées**.
- S515 : 49 hauteurs MIDI40–88 disponibles avec 11 indices sonores de fondamentale/attaque/sustain/harmoniques depuis huit WAV sources ; caractéristiques figées avant lecture XML/MIDI.
- S516 : producteur acoustique de hauteurs entraîné via ExtraTrees sur chaque ensemble d'entraînement.
- S509 : attention neuronale par hauteur sur les 13 têtes existantes H0–H12.
- S519 : vrai réseau S509 à **14 têtes**, ajouté à un générateur acoustique S516. Validation *imbriquée* : 8 partitions principales par enregistrement ; dans chacune, les probabilités acoustiques des 7 enregistrements train sont produites par 7 entraînements sur 6 **autres** enregistrements train. Seul le modèle acoustique entraîné sur les 7 enregistrements du train prédit le morceau tenu à part.
- S520 : contrôle indépendant des 16 modèles originaux, déjà validé.

**Tous les corpus ont été exposés pendant les études historiques.** Le découpage par enregistrement rend cette fusion plus crédible mais **n'autorise pas** à parler de test prospectif ou de généralisation prouvée. Un F1 par hauteur calculé sur des attaques positives annotées n'est PAS un F1 événementiel NoteOn sur le flux audio complet.

## S519, référence expérimentale de fusion

[CI S519 success](https://github.com/Andriamarosoa/note/actions/runs/38643896675) et [rejeu S520 success](https://github.com/Andriamarosoa/note/actions/runs/38644090354).

| Politique de sélection | Guitare /45 accords exacts | Piano /220 accords exacts | Guitare fixes/regressions | Piano fixes/regressions |
|---|---:|---:|---:|---:|
| BP original | 27 | 125 | — | — |
| S519 p≥0,50 | 26 | 121 | 0/1 | 0/4 |
| S519 p≥0,70 | 26 | 125 | 0/1 | 0/0 |
| S519 KEEP/ADD/REMOVE prudent | **28** | 125 | **1/0** | 0/0 |

La protection conservatrice ne rétablit toujours pas S516 seul (30/45 guitare) ni l'ancienne fusion fixe S518 (130/220 piano). **Ne pas promouvoir.**

## S521 — Essai rééquilibrant la rareté des vraies notes

[CI S521 success](https://github.com/Andriamarosoa/note/actions/runs/38644423162).

S521 ne change **que la pondération de la classe « note réellement présente » dans la fonction de perte de S519**, passant du plafonnement 2,5 à la racine carrée du nombre de faux candidats sur vrais candidats du TRAIN, plafonnée à 12. Toutes les partitions et décisions restent identiques ; pas de seuil appris sur le test.

| Politique | Guitare /45 | Piano /220 | Fixes/regressions guitare | Fixes/regressions piano |
|---|---:|---:|---:|---:|
| S521 p≥0,50 | 28 | 123 | 2/1 | 3/5 |
| S521 p≥0,70 | 24 | 128 | **1/4** | **8/5** |
| S521 protégé | 27 | 125 | 0/0 | 0/0 |

L'augmentation du poids de classe **ne résout pas le problème de fond** : manque de nouveaux vrais pitches et dégradation des accords polyphoniques.

### S522 — Audit indépendant complet

[CI S522 success](https://github.com/Andriamarosoa/note/actions/runs/38644603472) : 8 générateurs + 8 réseaux S519 et les mêmes 16 checkpoints S521, soit 32 empreintes de modèles, chaque note et K revérifiés, aucune optimisation. Constat clé : sur les 45 attaques IDMT et les 220 groupes MAESTRO, **S519 et S521 n'ont ajouté AUCUNE vraie note jusque-là absente de BP, quel que soit le seuil proposé**. Le réseau a essentiellement appris à supprimer des notes (dont certaines étaient correctes). C'est un défaut de conception de l'objectif, pas simplement un nombre insuffisant d'époques.

## S523 — Séparer explicitement « ADD_NEW_NOTE » et « KEEP_OR_REMOVE_NOTE »

[CI S523 success](https://github.com/Andriamarosoa/note/actions/runs/38645106253).

Le script `scripts/s523_train_independent_ADD_and_REMOVE_music_note_heads.py` crée **deux nouvelles têtes d'action entraînées** :
- `ADD_NEW_NOTE` : uniquement sur des hauteurs **absentes du jeu BP original** ; entraînée avec des preuves acoustiques réelles S515/S516, les votes des 13 têtes et les étiquettes des autres morceaux. Son label est « ce pitch absent était-il vraiment joué ? ».
- `KEEP_OR_REMOVE_NOTE` : uniquement sur les hauteurs **déjà proposées par BP** ; entraîne la distinction entre vraie voix et harmonique fausse.

Chaque validation externe sur un enregistrement entraîne 7 générateurs internes sur 6 autres enregistrements, puis 1 générateur sur 7 train, et chaque tête de sélection sur les exemples du TRAIN uniquement. 24 modèles sauvegardés SHA.

| Politique S523 | Guitare accords exacts /45 | Piano /220 | Corrections et régressions | Vraies nouvelles notes ajoutées / faux ajouts |
|---|---:|---:|---|---|
| Ajouter seul (p≥0,90, S516≥0,50) | 27 | 119 | Guitare 0/0 ; piano 0/6 | Guitare 0/0 ; piano **1/10** |
| Supprimer seul (KEEP≤0,15, S516≤0,35) | 27 | **127** | Guitare **1/1** ; piano **2/0** | Aucun ajout |
| Ajouter + supprimer | 27 | 121 | Guitare 1/1 ; piano 2/6 | Piano 1/10 |
| Variante couplée plus permissive | 27 | 124 | Guitare 1/1 ; piano 1/2 | Piano 1/4 |

Ce tableau est une **évaluation exploratoire**, et non un critère d'éligibilité pour figer une nouvelle référence : le même ancien corpus est utilisé depuis les itérations S509–S521. Le gain modeste de la branche de suppression piano ne prouve aucun taux de faux déclenchement sur un flux continu.

### S524 — Relecture indépendante des 24 modèles

`scripts/s524_independent_audit_learned_ADD_REMOVE_new_pitch_heads.py` vérifie les 24 fichiers de poids originaux, les 8 splits de composition source, les sorties réellement gelées et les vrais pitches ajoutés/supprimés, avec un bilan note par note et par vrai K. Les annotations servent **uniquement au calcul des erreurs**, jamais à sélectionner une action.

## Conclusion et prochaine architecture à construire

Cette boucle prouve qu'il ne suffit ni d'ajouter une 14ᵉ tête de probabilités ni d'augmenter le poids du label positif. Le déficit vient des **preuves acoustiques peu discriminantes sur les faibles fondamentales et de la détection d'attaques absentes du graphe BP**. Priorité :
1. Entraîner séparément des têtes `frame`, `onset`, `sustain` et `fundamental_vs_harmonic` sur un jeu de nouveaux WAV **avec véritables négatifs d'attaques**, en tenant compte du registre, de l'instrument et de l'ordre des notes ;
2. permettre aux têtes d'émettre de nouveaux événements canoniques NOTE_ON, y compris pour les 102 attaques MAESTRO sans groupe BP ;
3. faire coopérer les têtes Exact-K S80, ADD/REMOVE, protection poly et suivi de durée, en gardant le K historique comme prior souple ;
4. comparer à S516, S518 et YourMT3+ sur **de nouveaux morceaux indépendants** avec précision/rappel/F1 événement note hauteur+attaque, Exact-K par classe, accords exacts, offsets et latence Live ;
5. promouvoir une nouvelle référence **uniquement** après un gain vérifié sur ce test réellement disjoint, sans masquer les régressions.

**Décision : conserver les références existantes ; archiver ces modèles comme expériences vérifiées et les variantes rejetées comme garde-fous de conception.**
