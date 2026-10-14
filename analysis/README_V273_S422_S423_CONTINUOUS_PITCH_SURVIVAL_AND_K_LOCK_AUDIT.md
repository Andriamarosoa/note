# S422–S423 — Durée des notes par survie acoustique continue, **sans régression Exact-K**

**13 octobre 2026 — expérience et audit réussis.** S422 teste une vraie probabilité de persistance de chaque hauteur MIDI, sur des trames acoustiques complètes MIC GuitarSet à pas de **10 ms** jusqu'à **5 secondes** après le NoteOn. Contrairement à un correcteur statique de durée, le modèle reconsidère les harmoniques à plusieurs âges successifs (25, 50, ... 5000ms) et estime le moment où la probabilité que la note soit encore active devient trop faible. K0 n'est **pas** le silence, et des NoteOff peuvent survenir pendant K0.

## Périmètre et références préservées

- **S80/S397 référence Exact-K active** : 59 309 événements et 33 475 nouvelles notes, 53 529/59 309 = 90,25443 % Exact-K global et 6 054/7 385 = 81,97698 % Exact-K poly.
- **S402 référence hauteur MIDI active** et **S408 référence Offset explicitement choisie** restent inchangées.
- **S420 meilleur décodeur complet natif précédent** : 16 140 notes complètes correctes à ±50 ms sur 34 954 annotations GuitarSet, soit 47,17298 % F1 complet.
- S420 garde ses 33 475 hauteurs MIDI et 33 475 Onsets **identiques** lors de S422. Seul le NoteOff est autorisé à varier. Il reste **un NoteOn par unité du K original** (K0..K6).
- Comme S80 historique connaît déjà une partie de GuitarSet, tous ces chiffres sont un **essai de développement avec recoupement d'entraînement**, et non une validation indépendante complète. MIC GuitarSet est une source différente du pickup mix S80 ; S422 est **hors ligne** car il exploite jusqu'à 5 s de signal futur.

## Architecture S422 (tranches temporelles + survie)

1. Pour chaque note `(source_row_id, MIDI, Onset S407, K S80)`, inspecter le **WAV complet** et suivre l'amplitude de sa fondamentale et de ses partiels, leur pente, leur variation et les composantes concurrentes.
2. Construire 15 caractéristiques physiques/structurelles à 60 horizons échelonnés entre 25 ms et 5 secondes, comprenant la prédiction précédente S420, les notes suivantes et les réattaques de même hauteur. Les trames **sans nouvelle attaque** restent incluses.
3. Entraîner sur les **notes appariées** à leur note JAMS par hauteur MIDI et début à ±50ms, uniquement dans les **autres compositions** du fold. Le label est `note toujours active à l'horizon`, non une durée médiane statique.
4. Apprendre `P(note active | hauteur, K, âge, trajectoire continue, concurrence)`, monotoniser la courbe de survie, en déduire un temps de fin interpolé.
5. Choisir le seuil de probabilité et la politique (remplacement complet, prolongement seulement, prolongement suffisamment probable) **uniquement sur des compositions d'entraînement internes**. En l'absence d'un gain interne suffisant, **conserver intégralement S420** : fold 0.
6. Sur la composition tenue à l'écart, prédire le nouvel Offset, exporter les 33 475 vrais événements MIDI en JSONL avec `source_row_id`, `pitch`, `onset_sample`, `offset_sample`, puis **recompter indépendamment** toutes les notes et vérifier le K des 59 309 événements.

Code : [S422](https://github.com/Andriamarosoa/note/blob/codex/s276-idmt-song-disjoint-candidate-audit/scripts/s422_train_full_WAV_pitch_survival_noteoffs_S80_Klocked.py).

Workflow : [#38394264593](https://github.com/Andriamarosoa/note/actions/runs/38394264593) — **success** ; modèles et prédictions en artefact `11811400063`.

## Résultats S422 vs S420, 190 pistes GuitarSet

| Mesure | S420 (intouché) | **S422** |
|---|---:|---:|
| 59 309 K0..K6 inchangés | Oui | **Oui, 0 régression** |
| N nouvelles notes exportées | 33 475 | **33 475** |
| N vraies notes `note_midi` GuitarSet | 34 954 | 34 954 |
| Hauteur+Onset correctement appariés à ±50ms | 25 395 | **25 395 — inchangé** |
| Offsets corrects ±50ms sur notes appariées | 16 140 | **16 340** |
| Précision Offset conditionnelle | 63,5558 % | **64,3434 %** |
| F1 note complète `pitch+onset+offset` ±50ms | 47,17298 % | **47,75753 %** |
| Erreur absolue moyenne Offset, notes appariées | 91,0318 ms | **88,3999 ms** |
| F1 note complète sous tolérance Offset `max(50ms,20% durée)` | 50,5195 % | **51,4139 %** |

Le F1 progresse de **+0,58455 point** et le nombre de notes complètes exactes de **+200**.

### Vraies durées : toutes les notes, y compris celles non appariées

| Durée de vérité | Total de vraies notes | S420 notes complètes correctes | S422 | Gain net |
|---|---:|---:|---:|---:|
| <100 ms | 2 263 | 564 | **670** | **+106** |
| 100–499 ms | 28 212 | **15 043** | 15 024 | **−19** |
| 500–999 ms | 3 579 | 504 | **577** | **+73** |
| >=1 000 ms | 900 | 29 | **69** | **+40** |
| **TOTAL** | **34 954** | **16 140** | **16 340** | **+200** |

Attention : parmi les 900 vraies notes longues >=1s, **seules 736 sont appariées par MIDI et Onset** ; S422 termine **69 de ces 736** à ±50ms. La performance des notes longues est donc encore **insuffisante malgré le gain**.

## Audit indépendant S423 : vraies corrections/régressions appariées

Script [S423](https://github.com/Andriamarosoa/note/blob/codex/s276-idmt-song-disjoint-candidate-audit/scripts/s423_independent_S422_survival_vs_S420_offset_same_notes_Klocked.py) ; [run #38394546659](https://github.com/Andriamarosoa/note/actions/runs/38394546659) **success**, artefact `11810918948`.

- **1 228 corrections**, **1 028 régressions**, gain **+200 notes complètes** contre S420 sur les **mêmes 34 954 notes** (et non des sélections d'ensemble de notes différentes).
- Fold composition 0 : **0**, 1 : **+107**, 2 : **−65**, 3 : **+158**. Fold 0 a conservé les anciens Offsets par abstention après contrôle interne.
- Le gain global positif ne doit **pas masquer** les régressions du fold 2 ni les 19 notes 100–499ms perdues nettes.
- **Vérification de l'identité** : même `(source_row_id, hauteur MIDI, Onset)` entre S420 et S422. Le système n'a modifié aucune naissance de note et ne peut pas créer une régression Exact-K.
- **Relecture physique du JSONL** : exactement K nouvelles notes par `source_row_id`, incluant tous les événements K0. Total 33 475, et tous les K S80 de 59 309 lignes sont inchangés.

## Décision et prochain verrou

- **S80/S397** reste notre référence de comptage : 90,25 % global et 81,98 % poly sur développement, **zéro régression pour toutes les sorties S422**.
- **S402 et S408** restent les références des hauteurs et des Offsets explicitement conservées ; **S422 devient le meilleur challenger natif complet mesuré** avec 47,76 % F1, **sans promotion officielle**.
- L'état `note active` et la probabilité de survie ont apporté **+40 vraies fins longues exactes** mais seulement 69/900 de l'ensemble des notes longues sont complètes. Priorité : rendre la survie robuste aux notes de plus d'une seconde et aux réattaques de même hauteur, au lieu de pousser un seuil unique.
- Les *regressions par composition* (fold 2) requièrent un audit ciblé et des politiques d'abstention mieux calibrées sur musique étrangère, puis une validation réellement indépendante de l'ensemble S80+notes.
