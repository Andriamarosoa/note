# S612 — les têtes et sélections récentes de NOTE en flux

S612 porte **S511/S515 → S569 → S578/S585 → sélections S536/S585** dans des objets persistants. Les poids entraînés, les seuils et les règles originales sont conservés. S80/S397, ses familles de têtes et ses sélections restent protégés : **44 fichiers d'origine vérifiés inchangés**, dont le Live historique et S601.

Ce portage est un adaptateur en aval des producteurs d'événements V88 et des observations Basic Pitch déjà présents dans NOTE. Il ne constitue pas encore un microphone → S80 → notes autonome. Les observations BP utilisées ici sont celles des archives de NOTE ; leur décodeur hors ligne n'est pas devenu causal par ce portage.

## Ce qui est exécuté

| Élément | Portage S612 | Contrat conservé |
|---|---|---|
| S511/S515 | FFT 8192, pas 441 à 44,1 kHz ; calcul continu, historique de 512 trames | 49 hauteurs × 11 caractéristiques ; aucune FFT de silence ajoutée dans le futur |
| S562/S569 | Caractéristiques physiques 34D et mémoire des deux événements précédents | 52D par hauteur, délai maximal entre voisins 350 ms |
| S578 Global28 | Chargement unique du checkpoint SHA `21b23479…` | Probabilités des 49 hauteurs, modèle entraîné sur 28 sources |
| S585 quatre états | Chargement unique du checkpoint SHA `555cb092…` | Classes absence/résonance, activité, attaque, réattaque ; mêmes 28 sources |
| Sélections | Accords BP préservés, seuil 0,7, veto harmonique 0,8, déduplication 100 ms | Trois politiques séparées : Global28, S585 strict, S585 réattaque |
| Ordonnancement | Identités, chronologie, disponibilité audio et disponibilité du producteur distinctes | Refus des trous audio, producteurs trop tardifs, mélange de sources et files pleines |

Le nouveau module n'applique pas un quota K artificiel : les politiques récentes S578/S585 acceptent historiquement des sites V88 K0 et peuvent y récupérer une note. Le K original reste dans la sortie. Le chemin **Exact-K strict S80/S397 → S402/S416/S428** de S611 reste disponible séparément, avec ses sélections originales. La tête médium S578, rejetée lors de la validation antérieure, n'est pas promue.

S585 « activité » n'est pas une tête de NoteOff acoustique validée : son apprentissage utilise notamment un proxy d'activité avec pédale. S612 ne fabrique donc pas de fins de notes.

## Vérification sur les archives authentiques

| Sources déjà évaluées | Événements | Vérification |
|---|---:|---|
| Deux morceaux MAESTRO de S589/S590 | 340 | 52D identiques bit à bit ; trois politiques identiques à S590 |
| Cinq ensembles URMP de S595/S596 | 1 358 | 52D identiques bit à bit ; trois politiques identiques à S596 |
| Total | **1 698** | **Aucun groupe de notes modifié** |

Les probabilités diffèrent au maximum de **3,33 × 10⁻¹⁶** du calcul archivé, en raison de l'ordre de sommation des arbres. Les décisions restent identiques. Les modèles ne sont ni réentraînés ni convertis.

Les **120 secondes de véritable audio URMP** ont aussi été rejouées par blocs de **256 et 997 échantillons**. Les cinq extraits sont Rondeau (cordes), Dance (flûte/clarinette), Slavonic (cuivres), Pirates (cordes/saxophone), King (cordes). Les fichiers du miroir préparé à 16 kHz et les extraits rééchantillonnés à 44,1 kHz sont vérifiés par SHA. Ce n'est pas la distribution URMP originale à 48 kHz.

Le mode de référence conserve les notes archivées sur ces cinq extraits. Les caractéristiques acoustiques sont indépendantes de la taille des blocs ; l'écart maximal avec le calcul local d'origine est de **9,99 × 10⁻⁷**, et avec l'archive d'un autre environnement de **1,20 × 10⁻⁶**. Les six nouveaux tests, les tests S611 et les contrats Live historiques passent : **36 tests**.

C'est une vérification de portage sur des enregistrements connus. Ce n'est ni une nouvelle validation sur des guitares/IDMT, ni un score microphone Live. Les validations GuitarSet/IDMT/MAESTRO de S601 restent inchangées.

## Dépendance non causale identifiée et adaptation mesurée

S511 normalise chaque hauteur par la médiane et le percentile 95 de **tout l'extrait**, et le flux spectral par une médiane/MAD globale. Une fenêtre FFT bornée ne suffit donc pas à rendre la chaîne causale.

Deux modes explicitement nommés sont fournis :

- `reference` : profil calculé sur l'extrait entier, utilisé pour reproduire la référence ; **il lit le futur**.
- `rolling3s` : profil calculé avec les 300 dernières trames disponibles au bord droit fixe de l'événement. L'historique est borné, les sorties déjà émises ne changent pas quand le futur change. Les poids et seuils restent ceux de NOTE ; cette variante est **expérimentale**.

Les deux sorties ont été écrites avant la lecture des annotations déjà connues de S597. Aucun réglage n'a été choisi sur ces scores. Avec les mêmes producteurs V88/BP précalculés, sur les mêmes **1 119 notes MIDI40–88**, F1 NoteOn à ±80 ms :

| Politique | Référence sur extrait entier | Normalisation sur 3 s passées | Écart en points |
|---|---:|---:|---:|
| S578 Global28 | 62,23 % | 60,57 % | −1,66 |
| S585 strict | 64,78 % | 64,25 % | −0,53 |
| S585 réattaque | 64,64 % | 64,25 % | −0,39 |

La normalisation causale n'est pas promue : elle régresse. Les deux derniers scores ne sont pas exactement égaux avant arrondi (64,2470 % et 64,2549 %). Ce résultat mesure l'effet de la normalisation, pas la précision d'une chaîne entièrement causale : **les observations BP et événements V88 restent précalculés**.

S601 reste également une référence hors ligne : ses caractéristiques utilisent des observations futures, des durées de notes et la distance au candidat suivant ; sa suppression par probabilité classe les candidats sur l'extrait entier. Lui ajouter un tampon de 24 trames ne préserverait pas ses décisions. Ses scores supérieurs à 80 % ne sont pas attribués au portage S612.

## Temps et mémoire

La FFT centrée et les deux trames suivantes imposent environ **108 à 123 ms de disponibilité audio** en régime établi avec des blocs de 256. Les événements au début de l'extrait peuvent attendre **209 ms** ; une fin de source explicitement observée utilise les dernières trames réelles, comme l'original. Ces chiffres excluent le calcul du producteur V88/S80 et du décodeur BP, encore inconnu en Live.

Calcul local des deux têtes et des trois sélections par événement : médiane **15,75 ms**, p95 **19,32 ms**, p99 **22,88 ms**, maximum observé **82,06 ms**. Le rejeu n'est pas cadencé en temps réel et ne mesure pas la latence physique microphone/MIDI. L'inférence doit rester hors du callback audio ; une taille de bloc de 5,80 ms n'est pas une latence globale de 5,80 ms.

Le frontal conserve moins de 8192 échantillons de reliquat, au plus 512 trames spectrales résumées et 128 événements en attente. Les caractéristiques temporelles conservent deux événements ; chaque sélection conserve au plus 49 dernières attaques. Les CLI de validation accumulent des résultats pour écrire les preuves, contrairement au cœur d'inférence qui reste borné.

## Reproduction

Python 3.11/3.12, `PYTHONPATH=.:src` (`.;src` sous Windows), dépendances de `requirements-live.txt`. Pour les scripts originaux de comparaison : `mido==1.3.3` et `torch==2.5.1` CPU, comme dans la CI.

```text
python -m scripts.s612_restore_artifacts --out data/S612
python -m unittest -v test.test_recent_native test.test_native_stream test.test_live_k_midi_bridge test.test_note_runtime_modes test.test_v273_harmonic_trajectory
python -m scripts.s612_validate_recent_stream --assets data/S612 --fetch-urmp --out logs/S612
python -m scripts.note_runtime --mode recent-notes --status
```

La restauration utilise `gh`, les runs originaux S578/S585/S590/S596/S597 et vérifie les poids par SHA. `--models-only` restaure seulement les têtes. Les archives Actions ont une durée de rétention : ce n'est pas un nouveau paquet de poids autonome.

Rejeu d'un WAV mono 44,1 kHz PCM16/float32 avec ses événements originaux et son registre BP :

```text
python -m scripts.live_recent_notes --wav source.wav --events original_V88_source.jsonl --bp original_BP_notes.jsonl --models data/S612 --normalization rolling3s --out logs/recent_notes
```

`--events` conserve les champs d'identité, de temps et de K des registres S589/S595 ; les caractéristiques précalculées éventuelles sont ignorées et recalculées sur le WAV. `--bp` conserve les identités `WAV_filename_SHA256` ; seules les notes de la source sont utilisées. Les temps de disponibilité de ce rejeu ne mesurent pas le délai de production BP/V88.

[Rapport et scores par morceau/registre](evidence/v273-s612-recent-native-stream/report.json) · [Prédictions de référence](evidence/v273-s612-recent-native-stream/reference_predictions.json) · [Prédictions rolling3s](evidence/v273-s612-recent-native-stream/rolling3s_predictions.json) · [44 fichiers protégés](evidence/v273-s612-recent-native-stream/protected_recent_sources_SHA256.json).
