# Expérience 11 — finalisation explicite de fin de flux

État figé le 1er septembre 2026. La politique EOF est implémentée et auditée.
Aucun modèle, aucune cible acoustique, aucun sampler et aucune loss n'ont été
modifiés. Aucun entraînement n'a été lancé.

## Résultat

Pour un flux contenant les échantillons `0..N-1`, la réception d'un EOF
explicite applique désormais :

```text
pour chaque eventId encore ouvert :
    émettre offset(eventId, N)
```

Il s'agit d'une **fermeture de contrôle**, pas d'un offset entendu par le
modèle. Elle n'est exécutée qu'à la vraie fin d'un WAV ou lorsque stdin est
fermé. Un flux live qui continue ne reçoit aucune fermeture artificielle.

## Pourquoi cette règle est nécessaire

GuitarSet contient 408 notes dont l'offset est exactement égal à la longueur
exclusive du WAV. Il n'existe donc aucun échantillon audio à cette position :

| Split | Pistes concernées | Offsets internes supervisables | Offsets EOF |
|---|---:|---:|---:|
| Train | 140/240 | 43 894 | 326 |
| Validation | 36/60 | 9 459 | 82 |
| Total | 176/300 | 53 353 | 408 |

Les 408 notes vérifient toutes :

```text
onset <= N - 1 < offset = N
```

Elles sont donc encore actives au dernier véritable échantillon. Les déplacer
à `N-1` créerait un faux label acoustique ; ajouter du padding donnerait du
futur synthétique. Elles restent hors de la cible du modèle.

## Audit avant/après

Les annotations ont été rejouées avec leur identité dans le véritable
`LiveEventTracker`. Les offsets internes ferment normalement leurs événements,
puis la nouvelle opération terminale s'exécute à `frame_count`.

| Mesure | Avant finalisation | Après finalisation |
|---|---:|---:|
| Offsets acoustiques internes | 53 353 | 53 353, inchangés |
| Événements encore ouverts à EOF | 408 | 0 |
| Fermetures terminales émises | 0 | 408 |
| Fermetures terminales manquantes | — | 0 |
| Fermetures terminales supplémentaires | — | 0 |

La multiplicité est conservée. Les pistes concernées contiennent de 1 à 6
offsets répétés à leur unique position EOF :

| Multiplicité EOF par piste | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| Nombre de pistes | 84 | 21 | 27 | 25 | 13 | 6 |

Aucun contenu annotation ou audio Player05 n'a été ouvert ou utilisé.

## Comportement runtime

La finalisation est maintenant propagée à travers :

- `LiveEventTracker` ;
- les décodeurs énergie et modèle ;
- `LiveOnsetOffsetPipeline` ;
- `RestartScheduler` ;
- les chemins CLI WAV et stdin.

Les invariants vérifiés sont :

- même `eventId` entre onset et offset terminal ;
- ordre déterministe par onset puis identifiant ;
- tous les IDs ouverts sont validés avant mutation ;
- un lot incomplet, dupliqué, inconnu ou mal positionné est refusé sans
  finalisation partielle ;
- un échec de cohérence détecteur–scheduler laisse le flux utilisable ;
- la finalisation et le traitement après finalisation sont refusés ;
- WAV et stdin finalisent exactement une fois ;
- aucune trame de renouvellement `[N:N+512)` n'est créée.

Les éventuelles trames normales demandées avant EOF restent distinctes. Une
trame terminale serait impossible à compléter puisqu'il n'existe aucun audio
après `N`.

## Limite importante

L'audit oracle prouve la compatibilité de la règle avec les annotations. Il ne
prouve pas la qualité de l'association prédite : un faux onset encore ouvert à
EOF sera lui aussi fermé. Les métriques futures devront donc séparer :

```text
offset acoustique prédit
offset terminal de contrôle
```

Le format public reste néanmoins strictement `eventId, type, position` ; le
contexte EOF fournit la provenance sans ajouter de champ public.

## Vérification et décision

- politique oracle : acceptée ;
- `408/408` références EOF reproduites ;
- `0` manquante, `0` supplémentaire, `0` événement oracle restant ;
- `52/52` tests ciblés réussis ;
- `195/195` tests complets réussis dans l'environnement TensorFlow réel ;
- comportement causal avant EOF inchangé ;
- aucune trame terminale ;
- aucun entraînement.

La contrainte des offsets EOF est donc résolue. Conformément au protocole, le
travail s'arrête ici avant la prochaine modification : concevoir puis auditer
un sampler et une loss pour la cible anonyme exacte.

## Artefacts

- protocole préenregistré au commit `e4b8976`, SHA-256
  `68DD2207BB3F84A9688750613D26ADE8CFE3CAC852832E0879408EF232003E41` ;
- implémentation et audit au commit `e9d854f` ;
- détecteur, SHA-256
  `7D25E9601884CC8D2C7896E2A450F9A897E054A94481737CE7A361702DC2155F` ;
- scheduler, SHA-256
  `A98EAB662D2EBEFE586D69190E1F18BF46327E81AD089FD485410C6ABFA5CE30` ;
- pipeline, SHA-256
  `9C6FA49C4660274A12C298DBDBF2DE1C11A148D21F7F12E5433D7C1808F64331` ;
- CLI live, SHA-256
  `C913789643C87AB63DC36CAE324DED359F1C7D5F2FAFF3E01AAF754061C5FDFC` ;
- script d'audit, SHA-256
  `93FF47884D574B52553087DC4BE53B6FCBC2133826FD6D05355817CAEBB4087F` ;
- résultat JSON, SHA-256
  `19E18FE798D9E204AFDD6C243A8CD17939DBDC9FF286F29AAD70E272C2AC870F`.
