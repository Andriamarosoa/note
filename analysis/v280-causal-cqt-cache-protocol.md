# V28.0-C — protocole figé du cache CQT causal outer-clean

Statut au gel : **implémenté et testé localement, minage réel pas encore lancé**.

Ce jalon ne mesure aucun score et n'entraîne aucun modèle. Il matérialise la
représentation acoustique causale de V28 sur les 240 pistes GuitarSet
`outer-clean`, afin que le prochain modèle de comptage polyphonique puisse être
entraîné sans recalculer le CQT à chaque fold.

## Entrées immuables

| Source | Run | Commit | Artefact | Digest |
|---|---:|---|---|---|
| Identités V10 | `33647694565` | `345dea1e92f6281583c58590a0b1286ce4df459b` | `v104-nested-spectral-bundle` | `sha256:d54768cae40a9cf9b2c97a1be58fb6f52a681af3d79b79b148b5c92a427a9d99` |
| GuitarSet vérifié | `34287254337` | `d11f73079a9eb12b2ebce3c8d1df8209b0bd6bdf` | `v272-verified-guitarset` | `sha256:e90bd513877f8592c7f749410167aa581feb7a4b5d3250f12c980a3b8dc74587` |

Les MD5 GuitarSet restent figés à
`b39b78e63d3446f2e54ddb7a54df9b10` pour `annotation.zip` et
`aecce79f425a44e2055e46f680e10f6a` pour
`audio_mono-pickup_mix.zip`.

## Isolation des labels

Le bundle V10 n'est utilisé que pour deux tableaux :

- `schema_version` ;
- `track_members`.

Le code vérifie le schéma complet, mais ne désérialise jamais `target`,
`exact`, `slot_targets`, `members`, `top_samples`, `sequence`, `stats`, `mask`
ou `spectral`. Les tests placent volontairement des objets non chargeables dans
les tableaux de supervision : le préflight ne peut réussir que s'ils restent
inaccessibles.

L'archive JAMS est indexée par nom pour reproduire le split figé
`seed=1337`, `validation_fraction=0.20`. Aucun payload JAMS n'est ouvert. Les
60 noms de validation historique servent uniquement à prouver leur exclusion ;
leur audio n'est jamais décodé. Le locked12 n'est ni indexé ni évalué.

Le préflight doit établir simultanément :

- 300 pistes GuitarSet admises (`player 00` à `04`) ;
- exactement 240 pistes `outer-clean` et 60 pistes de validation ;
- égalité exacte entre les 240 identités V10 et les 240 identités train ;
- aucun chevauchement de piste ou de groupe de composition ;
- huit shards V10, sans identité dupliquée.

## Représentation figée

Le fichier `src/causal_note/v280_causal_cqt.py` définit
`v280_causal_octave_cqt_v1`, SHA-256 de configuration :
`5fae41ea91e90b5cb2a7611edb41c6dba633eaf8853341a481ea3a2c9ba1e805`.

| Paramètre | Valeur |
|---|---:|
| Fréquence d'échantillonnage | 44 100 Hz |
| Hop | 256 échantillons |
| Résolution | 3 bins par demi-ton, 36 par octave |
| Départ | MIDI 40 |
| Plafond de sortie | MIDI 83 |
| Harmoniques couvertes | ordres 2 à 8 |
| Bins internes | 238 |
| Plus grande fenêtre causale | 27 528 échantillons, 624,22 ms |
| Stockage | magnitude `float16`, endpoints et métadonnées explicites |

Chaque frame est alignée à droite sur son endpoint et ne dépend que de
`audio[:endpoint]`. Aucun centrage, right-padding ou lookahead n'est permis.

## Exécution et reprise

Le workflow `v280-causal-cqt-cache.yml` utilise un seul worker CPU et un seul
job lourd. Il exécute dans cet ordre : tests bornés, validation des runs et
digests, checksums GuitarSet, préflight réel sans décodage, puis minage des 240
pistes.

Une cache `.npz` indépendante est écrite atomiquement par piste. Après chaque
piste, `progress.json` est remplacé atomiquement. En cas d'échec, l'essai
suivant restaure ce checkpoint avec `--resume`, puis relit et vérifie chaque
cache existante avant de continuer. Un fichier modifié, absent, surnuméraire ou
issu d'une autre configuration bloque la reprise.

## Gates d'acceptation

Le résultat formel n'est accepté que si :

1. les 240 identités sont présentes une fois, en ordre canonique ;
2. le digest d'identités minées égale le digest `outer-clean` ;
3. chaque SHA-256 de fichier, forme, endpoint, fréquence et configuration est
   revérifié par une seconde lecture ;
4. les agrégats de frames, samples et octets se recomposent exactement ;
5. les drapeaux confirment : aucun label lu, aucun checkpoint/teacher chargé,
   aucun fold entraîné ou évalué, aucune validation historique ou locked12
   évaluée ;
6. `manifest.json`, `verification.json`, les 240 caches et le log sont réunis
   dans l'artefact `v280-causal-cqt-outer-clean-cache`.

Une réussite de ce jalon prouve seulement que la nouvelle entrée acoustique est
disponible et intègre. Elle ne constitue pas encore un gain d'Exact-K. Le fit du
nouveau compteur reste un jalon séparé avec sélection strictement interne.
