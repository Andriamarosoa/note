# Détection temporelle onset–offset

## Objectif unique

À partir d’un flux audio, émettre immédiatement les frontières détectées :

```text
onset(event_id, position)
offset(event_id, position)
```

L’offset porte le même identifiant opaque que son onset. Cet identifiant sert
uniquement à associer les deux événements.

## Sortie publique

La sortie reste strictement limitée aux deux frontières demandées :

```text
onset(event-000001, 100)
offset(event-000001, 700)
```

Le système ne publie ni hauteur, ni corde, ni cardinalité.

## Traitement live

`LiveEnergyDetector.process_chunk(...)` reçoit des blocs audio contigus. Il
émet l’onset dès le franchissement du seuil, puis l’offset lorsque 16
échantillons silencieux consécutifs confirment la fin. L’offset est horodaté au
premier de ces échantillons silencieux.

Cette baseline reste volontairement monophonique.

Le chemin polyphonique est maintenant implémenté :

```text
audio mono causal
  -> modèle onset/offset à 6 slots internes
  -> LiveBoundaryScoreDecoder
  -> onset(event_id, sample) / offset(event_id, sample)
```

Un onset ouvre un identifiant opaque dans son slot interne. Un offset dans le
même slot ferme exactement cet identifiant. L’offset est appliqué avant l’onset
au même échantillon afin de permettre un ré-enclenchement immédiat. Les scores
sont décodés par front montant : plusieurs échantillons consécutifs au-dessus du
seuil n’émettent pas plusieurs événements.

`LiveModelDetector` conserve la même interface live par blocs audio contigus.
`KerasBoundaryPredictor` garde uniquement le contexte causal requis entre deux
blocs. Le modèle n’accède jamais aux échantillons futurs.

La fin explicite de la source est un cas séparé d’une détection acoustique.
Pour un flux contenant les échantillons `0..N-1`, `finalize_stream()` ferme à
la position exclusive `N` chaque `eventId` encore ouvert. Cette opération
terminale conserve la multiplicité, ne prédit aucun padding et ne crée aucune
trame `[N:N+512)`. Tant que la source live reste ouverte, elle ne s’exécute pas
et le traitement bloc par bloc ne change pas. Une fermeture terminale signifie
« le flux s’est arrêté avec cet événement ouvert », pas « le modèle a entendu
son offset ».

`LiveOnsetOffsetPipeline` relie ce détecteur au `RestartScheduler` : les offsets
700 et 820 créent donc réellement les deux trames `[700:1212)` et
`[820:1332)` dans le chemin intégré. Après analyse, `complete_frame(...)`
supprime la demande terminée et compacte l’audio devenu inutile afin que la
mémoire ne croisse pas avec la durée du flux.

L’horodatage reste celui de l’échantillon prédit. L’émission physique intervient
à la fin du bloc qui contient cet échantillon : avec des blocs de 512 à
44,1 kHz, le tampon ajoute au maximum 11,6 ms, auxquels s’ajoute le temps
d’inférence.

L’adaptateur live conserve les états nécessaires de chaque convolution causale
au lieu de recalculer tout le passé. Le CLI préchauffe aussi le graphe avant de
lire le flux, sans consommer d’échantillon ni ouvrir d’événement.

Mesure locale du pipeline intégré sur 50 blocs de 512, après préchauffage, avec
l’architecture par défaut non entraînée : médiane 6,21 ms, p95 9,16 ms,
maximum 10,32 ms, soit 50/50 blocs sous le budget audio de 11,61 ms. Cette
mesure n’inclut pas la capture et ne constitue pas une garantie temps réel sous
charge système concurrente.

Les checkpoints binaires restent locaux et sont ignorés par Git. La baseline
de travail actuelle est le checkpoint récupéré V7-e8, verrouillé par son
empreinte, ses métadonnées et son rapport d’évaluation. Elle améliore V5 mais
sur-détecte encore fortement ; elle ne doit pas être présentée comme une
détection opérationnelle. Son état complet est décrit dans
`docs/BASELINE_V7_EPOCH08_2026-08-31.md`.

Pour ce checkpoint V7-e8, la calibration live officielle sur les 60 pistes de
validation utilise un seuil commun onset/offset de `0.55`. Le seuil générique
du décodeur et des CLI reste `0.5` : toute exécution V7-e8 doit donc passer
explicitement `--onset-threshold 0.55 --offset-threshold 0.55`. Cette valeur est
une calibration de validation ; le joueur `05` reste verrouillé et n'a pas été
lu.

## Apprentissage

GuitarSet supervise six slots internes grâce à ses six pistes de notes. Cette
information sert seulement pendant l’apprentissage et n’est jamais exposée
dans les événements publics.

Le chargeur :

- lit directement les ZIP sans extraction ;
- n’admet que les joueurs `00` à `04` ;
- convertit chaque `time` et `duration` JAMS en onset/offset à 44,1 kHz ;
- rejette explicitement le joueur `05`.

Le split garde ensemble tous les joueurs ainsi que les variantes `comp/solo`
d’une même composition. Les crops internes masquent de la loss les
`receptive_field - 1` premiers échantillons qui n’ont pas leur vrai contexte ;
aucun padding droit synthétique n’est ajouté. La validation réutilise un jeu de
batchs fixe entre les époques et l’arrêt anticipé restaure les meilleurs poids
de validation.

La pondération est élémentaire, de forme `(batch, temps, slot)` : un onset ou
offset positif reçoit le poids 64, tandis que les cinq slots négatifs au même
échantillon gardent le poids 1. Tous les slots du warmup masqué ont le poids 0 ;
la BCE conserve donc l’axe des slots jusqu’à l’application de ces poids.

Le poids `64` reste la valeur historique et la valeur par défaut du CLI. La
baseline de travail V7-e8 utilise explicitement un poids positif de `28` pour
les deux têtes, sans modifier le poids négatif.

Les expériences 10 à 13 auditent séparément une architecture suivante qui
prédirait uniquement les comptes anonymes exacts `onset_count[t]` et
`offset_count[t]`, avant association déterministe au runtime. Cette
architecture n'est pas encore implémentée et il n'existe pas de V8. L'audit du
sampler Exp13 répare la couverture minimale des négatifs immédiatement voisins
mais conclut toujours **NO-GO entraînement** à cause des comptes rares, du
prior artificiel, de la loss non sélectionnée et du crop ponctuel non vérifié.
Les résultats complets sont dans
`docs/EXPERIMENT_13_HARD_NEGATIVE_SAMPLER.md`.

Par défaut, les cibles onset et offset restent des impulsions exactes d’un
échantillon. `--onset-target-width-samples 512` et
`--offset-target-width-samples 512` permettent de les élargir séparément en
fenêtres binaires causales absolues `[frontière, frontière + 512)`, découpées
aux limites du crop même si la frontière précède celui-ci. Une largeur de `1`
préserve exactement le comportement historique.

Installation des dépendances d’entraînement :

```powershell
python -m pip install -r requirements-train.txt
```

Smoke training CPU borné :

```powershell
$env:PYTHONPATH='src'
python -B scripts/train_boundaries.py --smoke --players 00 `
  --output model/smoke-only.keras
Remove-Item Env:PYTHONPATH
```

Le smoke vérifie uniquement le câblage et ne produit pas un modèle utilisable.
Un entraînement refuse d’écraser un modèle existant sauf avec l’option explicite
`--overwrite`.

Après entraînement et validation d’un vrai checkpoint, la sortie live peut lire
un WAV par blocs :

```powershell
$env:PYTHONPATH='src'
python -B scripts/detect_live.py `
  --model model/causal-boundaries-weight28-window512-v7.epochs/epoch-08.keras `
  --metadata model/causal-boundaries-weight28-window512-v7-epoch08.recovery.metadata.json `
  --onset-threshold 0.55 `
  --offset-threshold 0.55 `
  --wav chemin/vers/audio-mono-44100-pcm16.wav
Remove-Item Env:PYTHONPATH
```

Pour une capture externe, `--stdin-pcm16` accepte un flux mono PCM16 little
endian à 44,1 kHz. Chaque événement est imprimé et vidé immédiatement :

```text
onset(event-000001, 100)
offset(event-000001, 700)
```

L’EOF du WAV ou la fermeture de stdin déclenche exactement une finalisation du
flux. Les offsets de contrôle ainsi produits gardent le même format public,
mais doivent être comptés séparément des offsets acoustiques dans une
évaluation.

Le CLI passe par le pipeline intégré : les trames de renouvellement sont créées,
accusées après disponibilité et purgées, mais la sortie texte reste limitée aux
onsets/offsets demandés.

## Renouvellement demandé

Lorsqu’un offset est détecté, une nouvelle trame commence exactement à cet
offset. Les demandes ne s’écrasent pas :

```text
A ferme à 700  -> trame [700:1212)
B reste ouvert
B ferme à 820  -> trame [820:1332)
```

Même si la trame 700 ne détecte rien entre 700 et 820, la trame 820 est tout de
même conservée et analysable dès que les 512 échantillons sont disponibles.

## Vérification

```powershell
$env:PYTHONPATH='src'
python -B -m unittest discover -s test
Remove-Item Env:PYTHONPATH
```

Le test polyphonique contrôlé exige exactement :

```text
onset(A, 100)
onset(B, 200)
offset(A, 700)
offset(B, 820)
```

Il vérifie aussi que les offsets réutilisent les deux bons identifiants à
travers plusieurs blocs live.
