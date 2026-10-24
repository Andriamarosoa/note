---
serie: S844
statut: termine
type: experience
reference: S708
branche: codex/note-s844-real-hex-rare-note-coverage256
run_id: 40087751049
commit_sha: f5b82d620e72db706e7ed64bae9d5d567a8c82f7
promotion: false
---

# S844 — Apprentissage renforcé de sons de cordes réellement enregistrés, E2 / E3, PCM256

## Objectif
Réparer les lacunes **mesurées** de S843 sans inventer de performances : S843 a vérifié trois archives GuitarSet originales, entraîné un CNN+GRU reconstructeur de 6 vrais canaux hex, mais sa transposition latente ne détectait **aucun E2/E3** sur son OUTER ; le témoin sans transpose utilisait le seuil CAL minimal .05 et ajoutait près de 9000 faux positifs. S843 avait 10 pistes TRAIN, 10 CAL, 40 OUTER, 28 séquences de huit décisions 256 par piste. Son OUTER sélectionné n'avait **aucun** E2+E3 actif simultanément, donc le F1 du chevauchement y est **non défini**, pas égal à 0 %.

## Hypothèse
La forte asymétrie positif/négatif des GT rendait la réponse « aucune note » favorable aux pertes BCE. Nous testons une meilleure couverture des vraies notes graves et un apprentissage **pondéré par les fréquences TRAIN uniquement**, ainsi qu'une calibration à seuil fin appris exclusivement sur CAL. Un son hex reconstruisible n'est pas une preuve de fondamentale ni d'Onset correct.

## Modifications
- `scripts/s843_train_real_hex_transpose256.py` : nouveau protocole S844 **dans une branche indépendante**, 96 séquences maximum par prise de huit blocs de 256, priorité aux véritables NoteOn E2, E3 et aux durées simultanément actives validées par JAMS ; autres attaques et contextes aléatoires conservés. Échantillonnage hors ligne uniquement, aucune annotation ni délai dans la tête Live.
- Nombre maximal de prises par joueur augmenté à 16, choisies sur BN1/Funk2/Jazz1/Rock3/SS1, sans changer TRAIN joueur 04, CAL joueur 05, OUTER joueurs 00–03.
- Compteurs par split : nombre exact de blocs positifs E2, E3 et E2+E3 ; les valeurs seront réellement mesurées dans CI après téléchargement. Un split OUTER dépourvu de simultanéité ne peut pas prouver un gain E2+E3.
- `src/onset256/real_audio_transpose_head256.py` : fonction de perte note pondérée par un vecteur de 49 poids plafonnés à 20 **calculés sur TRAIN seulement**, plus grande importance donnée à la preuve des notes réelles, tout en conservant la perte de reconstruction sur les vraies prises hex.
- `scripts/s843_evaluate_real_hex_reconstruction256.py` : calibration seuil depuis les quantiles et scores positifs de CAL (plus de plancher .05), puis gel avant OUTER ; aucune métrique F1 si 0 GT positifs OUTER. MAE de reconstruction comparée à une prédiction silencieuse pour éviter de promouvoir un générateur audio insignifiant.
- `tests/test_s844_real_hex_rare_note_audit256.py` : tests négatifs faux overlap, vrai contexte PCM256, BCE pondérée, calibration à seuil sous .05 et score absent si classe absente.

## Protocole
1. Télécharger les trois archives GuitarSet **vraies** et vérifier exactement les empreintes officielles : micro SHA256 237cdc... ; hex débleedé MD5 c31d... ; annotations SHA256 8daa... . Ne jamais générer une note cible par oscillateur ou effet DSP.
2. Constituer les données originales avec un seul nouvel échantillon PCM256 par frontière. Le modèle peut conserver un état de réseau récurrent issu de blocs passés, **aucun futur, aucun ajout d'attente**.
3. TRAIN/CAL/OUTER disjoints par identifiant de guitariste 04/05/00..03 ; ne jamais apprendre les poids des classes, les seuils ou le checkpoint sur OUTER.
4. Entraîner CNN+GRU reconstructeur sur les vraies cibles hex pickup, avec transport ±12 latent appris ; comparer la même architecture avec vs sans transposition. Évaluer E2, E3, E2+E3 avec TP/FP/FN par bloc, MAE sur vrais canaux hex et MAE prédiction zéro, le CPU Linux p99 et signaler le p99 Windows/iRig **non mesuré**.
5. Avant promotion MIDI, requérir un générateur NoteOn et replay des 80 WAV originaux (306 536 trames), vrai Onset F1 ±80ms, gain >69,2584331 %, aucune régression des cinq autres axes, aucun ralentissement physique.

## Audits obligatoires
| Source | Vérification |
|---|---|
| Données | GuitarSet originaux signés et couples mic/hex de la même vraie performance, distribution de GT E2/E3/joint, aucun WAV DSP |
| Acoustique | 256 nouveaux échantillons / décision, GRU passé uniquement, pas de FFT longue ni lookahead, cibles audio vraies |
| Architecture | branche nouvelle et séparée, reconstruction de 6 formes d'onde et transport ±12 appris, autres six têtes inchangées |
| Apprentissage | poids E2/E3 provenant de TRAIN seulement, CAL sélection seuil+checkpoint, OUTER jamais pour optimisation |
| Métriques | F1 de présence != Onset F1, rapporter les cas non évaluables, compter TP/FP/FN et comparaison silence |
| LIVE | p95/p99 appareil Windows/iRig inconnu; code sans attente, pas d'intégration au MIDI |

## Statut
**Terminé, refusé :** [GitHub Actions #40087751049](https://github.com/Andriamarosoa/note/actions/runs/40087751049) SUCCESS technique avec 16 TRAIN, 16 CAL, 64 OUTER et 49 152 pas. La reconstruction réelle hex s'est révélée moins bonne que le silence (MAE réseau 0,029785 vs 0,029205) et les vues latentes de transposition n'apportent aucun gain net E2/E3/joint sur les mêmes exemples. Aucun Onset F1 nouveau ; aucune promotion.  La référence Onset PCM256 à préserver est **69,2584331 % F1** (7 607 TP / 3 093 FP / 3 660 FN). Le résultat S843 prouve le fonctionnement technique de l'IA et l'échec de sa calibration/détection, pas l'atteinte de notre objectif musical.
