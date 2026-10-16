# NOTE LIVE — sortie audio de monitoring (prototype isolé)

Cette branche ajoute au chemin `scripts.live_strict_notes` un monitoring optionnel **du signal brut du microphone**, distinct de la synthèse de notes MIDI. La reconnaissance, les têtes, les sélections et les poids NOTE sont inchangés.

Sous Windows PowerShell, depuis la racine du dépôt et après installation des dépendances décrites dans LIVE.md :

```powershell
$env:PYTHONPATH = ".;src"
python -m scripts.live_strict_notes --list-devices
python -m scripts.live_strict_notes --mic --monitor-input --monitor-gain 0.3
```

Pour cibler une sortie WASAPI précise, relever l'index du périphérique WASAPI affiché par `--list-devices`, puis ajouter `--output-device INDEX`. L'entrée se choisit séparément par `--device INDEX`.

**Attention : utiliser un casque.** La boucle microphone → enceintes peut créer du Larsen. Le monitoring est désactivé par défaut. Une file bornée indépendante évite que la sortie audio ne bloque la reconnaissance : lorsqu'elle est vide, le programme émet du silence. Le son entendu est l'entrée microphone, **pas** une note resynthétisée.

Mise à jour : la sélection automatique du dépôt `midi` est maintenant portée dans `scripts/live_audio_io.py` : test des périphériques, préférence pour la plus faible latence négociée stable, entrée sans WDM-KS automatique, sortie avec WDM-KS/WASAPI. L'usage du périphérique explicitement sélectionné reste prioritaire. Cette adaptation n'a pas encore été testée sur le matériel Windows de l'utilisateur.

Validation à faire : tests matériels entrée/sortie, arrêt propre, niveaux et clipping, choix WASAPI vs MME, latence entrée→sortie (médiane/p95), overflows sous charge, vérification d'identité des prédictions avec/sans monitoring.

## Sortie piano (nouveau)

La demande principale est d'entendre **les notes prédites jouées au piano**, pas l'audio brut du microphone. Une sortie `--soundfont` optionnelle synthétise les événements NOTE `note_on` et `note_off` en stéréo via FluidSynth et PortAudio ; le programme sélectionné est General MIDI 0 (Acoustic Grand Piano). Les prédictions du moteur NOTE sont inchangées.

Dans un venv Windows PowerShell, installer les dépendances Python :

```powershell
python -m pip install pyfluidsynth==1.3.6
$env:PYTHONPATH = ".;src"
python -m scripts.live_strict_notes --list-devices
python -m scripts.live_strict_notes --mic --soundfont "C:\\SoundFonts\\piano.sf2"
```

Le paquet Python `pyfluidsynth` dépend également de la bibliothèque native FluidSynth installée sur Windows et disponible dans PATH. La SoundFont indiquée doit réellement exister ; il ne s'agit pas d'un fichier livré par NOTE. On peut cibler une sortie WASAPI, si disponible sur le PC, en ajoutant `--piano-output-device INDEX` où INDEX provient de `--list-devices`. Sans cet argument, NOTE teste automatiquement WDM-KS, WASAPI et le périphérique système et choisit le candidat stable ayant la latence négociée la plus faible.

`--soundfont` et `--monitor-input` s'excluent volontairement afin d'éviter le mélange involontaire entre piano synthétisé et microphone. La sortie piano est facultative ; la détection reste utilisable en JSONL et en MIDI. **Aucune validation sonore réelle sous Windows ni mesure de latence microphone→piano n'a encore été effectuée.**

## Mode recommandé : piano automatiquement routé

Le venv de NOTE doit être actif dans PowerShell. Depuis la racine du dépôt :

```powershell
git fetch origin
git switch codex/note-live-audio-monitor-wasapi
git pull --ff-only
python -m pip install -r requirements-live-strict.txt -r requirements-live-devices.txt
$env:PYTHONPATH = ".;src"
python -m unittest -v test.test_live_audio_selection
python -m scripts.live_strict_notes --list-devices
python -m scripts.live_strict_notes --mic --soundfont "C:\Users\user\Documents\Codex\guitarRig\.venv309\Lib\site-packages\pretty_midi\TimGM6mb.sf2"
```

Si `python -m pip install` n'a installé que le package Python FluidSynth, la bibliothèque Windows native FluidSynth doit également être installée et accessible dans PATH, sinon le lancement renverra une erreur explicite.

À l'ouverture, le programme affiche maintenant sur stderr la sélection automatique des entrées/sorties, le nom de l'API et les latences négociées, en conservant la sortie JSONL sur stdout. En cas de panne de synthèse répétée ou d'arrêt du flux de sortie, le LIVE remonte une erreur plutôt que de rester silencieux.

Cette sortie piano est une **synthèse des notes prédites** par le moteur NOTE, non un monitoring de la guitare d'origine, non une modification des poids ou des têtes, et non un nouveau score de F1. Le port MIDI externe reste optionnel et s'active séparément avec `--midi-port`. Les tests unitaires du sélecteur sont fournis dans `test/test_live_audio_selection.py`, mais le test matériel Windows et la mesure de latence acoustique restent à effectuer.

## Diagnostic ciblé iRig : cordes graves absentes

Comparer le même signal *avant tout gain et tout modèle*. Le script `scripts/diagnose_low_strings.py` enregistre le premier canal du périphérique effectivement ouvert par le sélecteur audio NOTE, sans le normaliser, et affiche le nom de l'entrée, RMS dBFS, crêtes, clipping, énergie à la fondamentale et aux harmoniques et les pics spectraux principaux. C'est un diagnostic de capture et non une validation de note reconnue.

Depuis le venv actif sous PowerShell :

```powershell
$env:PYTHONPATH = ".;src"
python -m unittest -v test.test_low_string_diagnostics
python -m scripts.live_strict_notes --list-devices
python -m scripts.diagnose_low_strings --mic --note E2 --duration-s 4 --record-wav "irig_E2.wav"
python -m scripts.diagnose_low_strings --wav "irig_E2.wav" --note E2
python -m scripts.live_strict_notes --wav "irig_E2.wav" --out "irig_E2_notes.jsonl"
```

Au début du diagnostic, jouer uniquement la corde Mi grave E2, puis enregistrer la corde La A2 en remplaçant E2 par A2 et le nom de fichier. Si l'entrée choisie n'est pas l'iRig, passer `--device INDEX` d'après la liste; la sélection automatique suit normalement l'entrée Windows par défaut. D2/Drop D (MIDI38) est en-dessous de la limite MIDI40 du modèle NOTE strict actuel.

Pour tester **seulement le piano**, sans guitare ni modèle, avec une SoundFont existante :

```powershell
python -m scripts.diagnose_low_strings --test-piano --note E2 --soundfont "C:\Users\user\Documents\Codex\guitarRig\.venv309\Lib\site-packages\pretty_midi\TimGM6mb.sf2"
```

Interprétation : si E2 ne s'entend pas en test piano direct, examiner d'abord la sortie audio, le périphérique d'écoute ou la SoundFont. Si le piano joue E2 mais que son énergie est très faible ou absente dans le WAV brut, examiner capture iRig, gain et traitements Windows. Si le WAV a une harmonique grave visible mais aucune note E2 dans `irig_E2_notes.jsonl`, examiner le modèle, les caractéristiques et les seuils de sélection de NOTE.

Pour la comparaison avec `midi`, réutiliser **exactement ce WAV brut** dans `guitar-midi-poly-transcribe` avec le bundle polyphonique adéquat, une fois avec `--no-auto-level` puis avec `--auto-level` ; ne pas comparer deux interprétations différentes de la guitare. Les différences de note ou de gain ne prouvent rien sans inspecter les rapports.

Les tests de cette branche sont à exécuter dans le venv ; aucun test d'iRig physique ou mesure réelle par corde n'a encore été effectué. Ne pas changer le modèle ou amplifier chaque note à l'aveugle.

## Gain automatique causal adapté de `midi` (expérimental, opt-in)

Le fichier `scripts/live_input_level.py` reprend le contrôleur `CausalModelInputLeveler` du dépôt `Andriamarosoa/midi/src/polyphonic/input_level.py`, à gain borné +18 dB et gestion de la décroissance des notes. `note` strict ne dispose **pas** de la calibration d'activité du LIVE `midi` : un adaptateur spécifique exige les seuils bruts peak -55 dBFS et RMS -65 dBFS. Ce n'est donc pas une copie complète du pipeline MIDI. Sans `--auto-level`, les données transmises au modèle restent inchangées.

Exemple dans le venv actif, à partir du dépôt NOTE :

```powershell
git fetch origin
git switch codex/note-live-audio-monitor-wasapi
git pull --ff-only
$env:PYTHONPATH = ".;src"
python -m unittest -v test.test_live_input_level
python -m scripts.live_strict_notes --mic --device "WASAPI:External Microphone" --auto-level --soundfont "C:\Users\user\Documents\Codex\guitarRig\.venv309\Lib\site-packages\pretty_midi\TimGM6mb.sf2"
```

Comparer **le même** WAV sans et avec auto-level (noms de sortie distincts) :

```powershell
python -m scripts.live_strict_notes --wav "irig_E2_wasapi.wav" --out "e2_base.jsonl"
python -m scripts.live_strict_notes --wav "irig_E2_wasapi.wav" --auto-level --out "e2_autogain.jsonl"
Get-Content e2_base.jsonl | Select-String '"type": "note_on"'
Get-Content e2_autogain.jsonl | Select-String '"type": "note_on"'
```

Réglages : `--auto-level-max-db 18` (0–24), `--auto-level-peak-gate-dbfs -55` et `--auto-level-rms-gate-dbfs -65`. Ne pas monter les seuils/gain sans examiner les notes fantômes. Le contrôleur n'applique aucune normalisation individuelle aux queues de notes. Les diagnostics du contrôleur sont imprimés sur stderr lors de l'arrêt propre.

**Limitation majeure pour cet iRig** : le signal E2 enregistré à RMS ≈ -116.9 dBFS est quasi nul, et l'export PCM16 peut quantifier les faibles composantes à zéro. Un maximum +18 dB ne reconstitue pas une fondamentale manquante ; le gain vise plutôt les signaux faibles mais encore exploitables. Aucune amélioration F1 ou résultat matériel Windows n'est établi. Valider les faux positifs et régressions sur WAV tenus à l'écart avant de promouvoir cette option.

## Audit d'une corde grave perçue comme accord (E2 -> faux harmoniques)

Sur une prise **MME**, E2 enregistrée à peak -6.5 dBFS / RMS -23.4 dBFS est bien présente spectralement. Le LIVE sans gain a émis MIDI 52 (E3), 59 (B3), 64 (E4) et 68 (G#4), coïncidant respectivement avec les harmoniques 2, 3, 4 et 5 de E2, plus MIDI 61 (C#4). Il n'a pas émis E2 MIDI 40. Ce cas prouve une confusion sur une prise guitaristique réelle, mais ne suffit pas pour une statistique F1.

Un défaut observable : la configuration native comporte `count_confidence=1.1`. La sortie de la tête K est une softmax (valeurs entre 0 et 1), donc la sélection actuelle n'utilise **jamais** le budget K pour limiter les notes. Cette valeur peut être issue d'une ablation sélectionnée par validation ; elle ne doit pas être changée sur les jeux d'évaluation sans audit de régressions.

```powershell
git pull --ff-only
python -m unittest -v test.test_live_count_veto
python -m scripts.audit_live_pitch_count --wav "irig_E2_wasapi.wav" --expected-pitch 40 --report "audit_e2.json"
$a = Get-Content "audit_e2.json" -Raw | ConvertFrom-Json
$a.note_on_by_selection | ConvertTo-Json -Depth 5
$a.first_original_attack_frame | Select-Object time_s,predicted_k,k_confidence,k_probabilities_0_through_6,expected | Format-List
$a.first_original_attack_frame.pitch_ranking | Select-Object -First 8 pitch,name,onset,sustain
```

Trois rejouages causaux sont effectués **sur les mêmes sorties des mêmes têtes** :
- `original`: configuration gelée, sans veto K,
- `count_gate_050`: veto fondé sur la vraie distribution K, si sa probabilité maximale >= .5,
- `mono_budget_1`: borne artificielle à 1 note, simple témoin de diagnostic incompatible avec la polyphonie.

Ne jamais promouvoir directement le mono budget ou le count_gate_050 à partir de la seule corde E2. Il faut également vérifier les vrais accords K2–K6, les régressions, et le comportement LIVE strict. Le code d'audit n'ajoute aucun délai ni aucune fonction à l'architecture de production.

## Après audit E2 v2 : fondamentale/harmoniques et trajectoire causale

Sur le WAV E2 fourni par l'utilisateur (MME ; peak -6.5 dBFS, RMS -23.4 dBFS), l'audit v2 au premier déclenchement (t≈1.155 s) montre :
- E2 MIDI40: score natif 0.69099, 8192 score 0.13487 ; onset 0.00679, sustain 0.00336.
- E3 MIDI52: score natif 1.01425, 8192 score 0.22322 ; onset 0.43562, sustain 0.28253.
- B3 MIDI59: score natif 0.95941 ; onset 0.74672, sustain 0.50990.
- C#4 MIDI61: onset 0.77878, sustain 0.57643.
- K3 softmax=0.42313 au premier déclenchement ; puis K4 confidence 0.84286 à 1.184218 s, alors qu'une seule corde était jouée.
- H2 de E2 sur la fenêtre 8192 et H1 de E3 partagent **exactement** la même valeur log-FFT (0.26238 au premier déclenchement ; 1.09344 au second). H3 de E2 et H1 de B3 partagent aussi la même valeur (0.26672 puis 1.07671). Il s'agit de la même composante physique vue sous plusieurs hypothèses de hauteur, et non de preuves de plusieurs notes. Le sélecteur actuel n'applique pas de décomposition explicite des harmoniques.
- La fenêtre 8192 est **causale et alignée à droite**. Les scores au tout début de l'attaque peuvent être bas même lorsque le diagnostic spectral trouve une fondamentale clairement visible dans la fenêtre la plus forte du WAV. Cet écart temporel doit être audité avant toute modification d'architecture.

Version v3 de l'audit (rejoue le même fichier, aucune future information fournie au LIVE) :

```powershell
git pull --ff-only
python -m unittest -v test.test_live_pitch_trajectory
python -m scripts.audit_live_pitch_count --wav "irig_E2_wasapi.wav" --expected-pitch 40 --report "audit_e2_timeline.json" > $null
$a = Get-Content "audit_e2_timeline.json" -Raw | ConvertFrom-Json
$a.attack_to_sustain_trajectory.sampled_causal_frames | Format-Table offset_requested_ms,time_s,e2_onset,e2_sustain,e2_h1_8192,e2_harmonic_score_8192,e3_harmonic_score_8192,predicted_k
$a.attack_to_sustain_trajectory.maxima_during_next_500_ms | ConvertTo-Json
```

Les échantillons à +25/+50/+100/+200/+350/+500 ms ne servent qu'à évaluer la *progression* du modèle déjà causal ; les utiliser pour retarder l'émission violerait le LIVE strict. Aucun seuil, poids, tête, modèle ni sélection n'a été promu. Si la fondamentale devient visible plus tard, la prochaine expérience appropriée est une **tête entraînée de cohérence d'harmoniques à l'attaque** utilisant uniquement l'histoire déjà reçue, avec validation sur accords où les octaves sont des notes réelles.

## Résultat E2 du contre-factuel onset=0.15 et audit de coïncidence des têtes

Sur le seul WAV MME E2 annoté (MIDI40), les statistiques fournies par l'utilisateur sont :
- Original : E2=0, autres notes=5 (MIDI52,59,61,64,68).
- Count gate 0.5 : E2=0, autres=5.
- Mono budget 1 : E2=0, autres=2 (MIDI52,61).
- Onset 0.15 + reset 0.03 : E2=0, autres=5.

Ne PAS assimiler `max(onset)=0.15213` et `max(sustain)=0.99455` à une coïncidence temporelle. Le sélecteur exige `onset >= seuil` et `sustain >= active_on` **sur la même trame**, plus armement et budget. À t≈1.2074, sustain E2=0.63835 mais onset=0.10857. L'audit suivant mesure la vraie coïncidence et rejoue neuf combinaisons de seuils dans des sélecteurs distincts, **sans rien modifier au LIVE** :

```powershell
git pull --ff-only
python -m unittest -v test.test_live_joint_head_overlap
python -m scripts.audit_live_pitch_count --wav "irig_E2_wasapi.wav" --expected-pitch 40 --report "audit_e2_overlap.json" > $null
$a = Get-Content audit_e2_overlap.json -Raw | ConvertFrom-Json
$a.joint_head_overlap.conditional_maxima | Format-Table
$a.summary | ConvertTo-Json -Depth 4
```

Les variantes explorent onset {0.05,0.10,0.15} x sustain {0.25,0.40,0.50}, mais l'exploration d'un seul signal n'est **pas** une optimisation validée. Les détections inattendues comptent comme erreurs uniquement sous l'hypothèse documentée « une seule corde E2 ». Pour changer le modèle, les têtes ou les sélections en production, il faut des morceaux/accords indépendants, des mesures d'attaques, des fausses notes et la non-régression Exact-K.

## E2 — résultats expérimentaux finaux sur le seul WAV iRig/Realtek fourni (aucune promotion)

Tests locaux utilisateur : `test.test_live_joint_head_overlap` : 3/3 OK. Le balayage des neuf seuils est terminé. Toutes les variantes ci-dessous ont été appliquées au même unique enregistrement d'une corde E2 (MIDI40), non à un test indépendant.

| Sélection | note_on(E2) | note_on(autres) | premier E2 (s dans WAV) |
| --- | ---: | ---: | ---: |
| Référence | 0 | 5 | — |
| Count gate 0.5 | 0 | 5 | — |
| Budget mono 1 | 0 | 2 | — |
| Attaque 0.15 / maintien 0.50 | 0 | 5 | — |
| 0.05 / 0.25 | 4 | 10 | 1.190023 |
| 0.05 / 0.40 | 4 | 8 | 1.195828 |
| 0.05 / 0.50 | 4 | 7 | 1.207438 |
| 0.10 / 0.25 | 1 | 7 | 1.195828 |
| 0.10 / 0.40 | 1 | 6 | 1.195828 |
| **0.10 / 0.50** | **1** | **5** | **1.207438** |
| 0.15 / 0.25 | 1 | 7 | 1.195828 |
| 0.15 / 0.40 | 1 | 6 | 1.195828 |
| 0.15 / 0.50 | 0 | 5 | — |

Premières fausses notes de la référence à t=1.155193 s. Le E2 unique de 0.10/0.50 arrive environ +52.25 ms après ; les autres cinq fausses attaques restent. Les 4 `note_on` E2 obtenus avec seuil 0.05 ne sont **pas** quatre vrais positifs distincts sur une seule corde, mais des redéclenchements indésirables.

**Les maxima de têtes ne coïncident pas** :
- Pour `sustain >= 0.4`, max onset conjoint = 0.15213 à t=1.195828 ; sustain=0.47819.
- Pour `sustain >= 0.5`, max onset conjoint = 0.10857 à t=1.207438 ; sustain=0.63835.
- Pour `sustain >= 0.7`, max onset conjoint = 0.03083 à t=1.219048.

### Décision et prochaine architecture candidate

**Aucune variante de seuil ne remplace la référence** : impossible de sauver E2 sans conserver, voire multiplier, les fausses notes sur cette prise. Conserver strictement la latence causale, les têtes natives NOTE et les vraies notes polyphoniques.

Candidate à évaluer puis entraîner : une tête native de **reconnaissance de parent harmonique / fondamentale manquante** qui apprend la compétition entre E2 et les notes d'harmoniques E3, B3, E4, G#4. Tester comme indices spécifiques les 5e et 7e partiels non expliqués par les hypothèses E3/B3 ainsi que les attaques individuelles et la cohérence temporelle. En présence d'une vraie superposition E2+E3, ne pas supprimer automatiquement E3. Garder une tête de comptage adaptée au signal guitaristique ; K3-K5 sur une monodie montre une confusion parent/harmoniques. Former sur plusieurs cordes et accords réels avec hold-out par prise/jeu/instrument et négatifs difficiles (E3 seul, B3 seul, E2+E3 et triades).

Contrainte : les données de cette ligne proviennent uniquement du **log d'audit de l'utilisateur** ; le WAV ne se trouve pas dans la session de travail du développeur. Aucune mesure F1, calibration, entraînement ou validation sur plusieurs instruments n'a été faite avec ce cas. Mesurer séparément capture->première note erronée, capture->première note correcte, et latence physique du jeu.

## Mesures indépendantes du WAV réel iRig E2 reçu dans la conversation (2026-10-16)

Le fichier **`irig_E2_wasapi.wav`** a été reçu et inspecté directement (il s'agit bien de la capture MME antérieure, malgré son nom). L'audit ci-dessous est fondé sur les échantillons de **175 104 PCM mono, 44 100 Hz, 3,9706 s**, pas sur une sortie prédite ou un WAV synthétique. Pic = 0,4746704 (-6,47 dBFS), RMS = 0,067455 (-23,42 dBFS), pas de saturation numérique. Une corde grave est pincée vers **1,12 s**, puis décroît jusqu'à EOF ; les premières fausses notes NOTE interviennent à t=1,155193 s (environ 35 ms après le début du signal perceptible).

### Série physique harmonique confirmée sur le vrai signal

Spectres calculés par FFT/Hann, sur fenêtres glissantes séparées de la prédiction `NOTE_strict_live.npz`. Entre 1,3 et 1,6 s, les pics observés autour des composantes d'un F0 ~81,18 Hz sont :

| Harmonique | Pic mesuré (Hz) | Niveau approx. (dBFS) | Hypothèse interprétée à tort comme note |
|---|---:|---:|---|
| H1 | 81,1 | -41,8 | E2 MIDI40 (la vraie fondamentale) |
| H2 | 162,2 | -17,1 | E3 MIDI52 |
| H3 | 243,3 | -19,4 | B3 MIDI59 |
| H4 | 324,7 | -24,3 | E4 MIDI64 |
| H5 | 406,4 | -27,9 | G#4 MIDI68 |
| H6 | 488,2 | -27,6 | harmonique supérieure |
| H7 | 569,6 | -28,8 | harmonique supérieure |

La fondamentale est environ **24,7 dB sous H2**, 22,4 dB sous H3 : un cas classique de fondamentale faible, PAS un problème de niveau de capture. C#4 MIDI61 reste une fausse note qui ne correspond pas directement aux premières harmoniques entières. La hauteur estimée varie d'environ 81,7 Hz (1,17–1,29 s) à 80,6 Hz (2–2,5 s), soit environ -15 à -40 cents par rapport à E2 théorique 82,4069 Hz ; la petite dérive de hauteur ne suffit pas à expliquer la confusion d'octaves.

### Évolution temporelle de la preuve

En prenant des fenêtres de 8192 échantillons **causales alignées à droite** et le maximum FFT autour des partiels attendus :
- À **1,155193 s** (premières fausses émissions), H2≈161,5 Hz et H3≈251,7 Hz sont seulement ~2,3 dB au-dessus du fond spectral local : indices précoces ambigus, fondamentales très faibles.
- À **1,184218 s**, H2≈162,2 et H3≈247,0 Hz deviennent environ +8,4/+7,8 dB plus distincts.
- À **1,207438 s**, H2≈162,8 et H3≈245,6 Hz deviennent +12,3/+13,0 dB, avec ratio H3/H2 ≈1,508 : forte indication du parent harmonique E2 (~81,5 Hz).
- À **1,358367 s**, H2≈162,8 et H3≈244,3 Hz sont très nets (~+41 dB de contraste spectral local chacun).

Le modèle causal tend à émettre trop tôt plusieurs enfants harmoniques, avant que sa tête d'attaque E2 ne monte suffisamment ; plus tard `sustain(E2)` est ~0,995 alors que `onset(E2)` s'est effondré. Une architecture future doit exploiter les relations d'harmoniques **dans les trames déjà reçues**, avec parent E2 et preuves d'indépendance acoustique des enfants. Elle ne doit NI attendre 100–500 ms après l'attaque, NI supprimer systématiquement E3/B3 lorsque ces notes peuvent aussi être réellement jouées.

Le fichier utilisateur **n'a pas été publié dans GitHub**, le dépôt ne contient que cette analyse textuelle. Aucune validation globale de cette nouvelle hypothèse n'a eu lieu : ne pas promouvoir un correcteur construit sur le seul WAV E2. Tester d'abord des vrais E3 seuls, B3 seuls, E2+E3, accords et contre-exemples avec fractionnement indépendant par prise.
