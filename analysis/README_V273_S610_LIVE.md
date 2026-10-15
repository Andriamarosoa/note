# S610 — archive de l’intégration retirée en S611

**Statut : intégration retirée à la demande de l’utilisateur.** Elle réutilisait directement la reconnaissance de `midi`, alors que la demande était d’améliorer l’architecture de NOTE. Le code, les poids importés et le lanceur ont été retirés par S611 ; les mesures ci-dessous décrivent uniquement le commit historique `e850198`. Voir [le portage natif S611](README_V273_S611_NATIVE_STREAM.md).

Description historique : la priorité devenait le Live. Une nouvelle application autonome, `scripts/live_notes.py`, relie capture microphone, modèle causal, notes nommées et sortie MIDI facultative. [Instructions de démarrage](../LIVE.md).

## Ce qui est effectivement intégré

- Modèle TFLite polyphonique `midi` v2.2.1 et logique de décodage figés au commit `536536f54290fdc4f63db28aa063d2e324121642`. Empreintes des dix fichiers repris, dont les poids et métadonnées, dans le manifeste de provenance.
- Entrée mono 44,1 kHz, blocs de 256 échantillons ; contexte passé borné à 4 096 échantillons ; aucun lookahead. Sortie E2–E5, maximum six notes actives.
- Calibration sur le microphone, activité audio et attaque physique, probabilités de présence et d'attaque par hauteur, support harmonique, réattaque avec NoteOff avant NoteOn, extinction par release/silence.
- Horloge de capture conservée malgré les blocs perdus ; identifiants stables ; événements datés au moment de décision. Arrêt, erreur et coupure ferment les notes, avec panic MIDI sur le canal choisi.
- File bornée, saut limité d'inférence sans perdre l'audio pour les petits retards, abandon du vieux contexte lorsque le retard devient excessif. Statistiques en mémoire bornée, journaux facultatifs écrits au fil de l'eau.

**Ce moteur n'est ni S601 ni un Exact-K S80 causal.** La limite de six notes actives n'est pas un estimateur du nombre de nouvelles notes. Les scores S601 de 80,06 % sur GuitarSet et 82,11 % sur IDMT ne sont pas attribués à S610. Les six fichiers protégés du manifeste historique sont vérifiés inchangés ; aucun modèle S80, S601 ou script gelé n'est modifié.

## Donnée et périmètre de la vérification

Le test de flux emploie les **30 premières secondes** du fichier déjà présent dans `midi`, `runs/polyphonic/wav_live_validation_v221/ab_listening_pack/source.wav`, au même commit figé. Le manifeste amont le rattache à `Until I Found You_basic_pitch.wav`. Sa provenance instrumentale n'est pas certifiée ici : **ce n'est pas un nouveau benchmark GuitarSet, IDMT ou une validation sur plusieurs guitares**.

Le fichier source mono PCM16 est à 22 050 Hz. Sa SHA256 est `8872801df7723cec12b9ae8881cae869acb19fbf90adecbf02c83f8ab19d5123`. L'extrait est préparé une fois par `scipy.signal.resample_poly(x, 2, 1)` (SciPy 1.14.1), arrondi en PCM16 après multiplication par 32 767. Le WAV commun à tous les essais contient 1 323 000 échantillons, SHA256 `e42f0a51410d928ea6c49b0022d3d2d9cec0928bf3af8ec5a3d9bd035580340b`. Cette conversion hors ligne prépare le signal de test ; elle ne décrit pas un rééchantillonneur microphone causal dans S610.

Aucune annotation musicale n'est utilisée ni aucun seuil du modèle retouché. Les **90 NoteOn** de la référence sont des prédictions, pas un nombre de vraies notes reconnu. Les 248 derniers échantillons incomplets n'engendrent pas d'inférence sur du silence artificiellement ajouté ; les notes restantes sont fermées à la vraie fin du fichier.

## Résultats logiciels

**34 tests passent** dans l'environnement Live Python 3.11, NumPy 1.26.4, TensorFlow 2.15.1 : état des notes, silence, réattaque, limite polyphonique, identité et ordre des événements, coupure audio, horloge après rejet, conservation de l'audio et de la date d'attaque lors d'une inférence sautée, fin incomplète, mémoire bornée, échec MIDI, arrêt microphone, échec réseau et préservation du Live historique.

Le rejeu direct avec le modèle réel donne **180 événements exactement identiques** à une exécution séparée des fichiers du dépôt `midi` d'origine : 90 NoteOn et 90 NoteOff, mêmes hauteurs, vélocités, temps de décision et motifs. Le test de préfixe retrouve les mêmes décisions lorsque la suite de l'audio n'est jamais fournie au moteur. Le test synthétique complémentaire vérifie silence, sorties finies et fermeture ; il ne mesure pas une précision musicale.

## Rejeu cadencé et surcharge

Tous les essais sont conservés dans [evidence/v273-s610-live](evidence/v273-s610-live), y compris ceux qui ont révélé des pertes. Les traitements sont exécutés sur CPU dans un environnement partagé, sans périphérique microphone ni synthétiseur matériel raccordé. Les sorties de ces essais sont des journaux JSONL, sans port MIDI physique ouvert.

| Essai | NoteOn / NoteOff | Blocs abandonnés | Remises à zéro | p95 callback → fin de traitement |
|---|---:|---:|---:|---:|
| Première politique, abandon dès >3 blocs, TF 2.14.1 | 98 / 98 | 4 | 1 | 4,06 ms |
| Première politique, pause volontaire de 200 ms, TF 2.14.1 | 89 / 89 | 33 | 1 | 3,95 ms |
| Politique corrigée, premier essai TF 2.15.1 | 92 / 92 | 17 | 1 | 3,46 ms |
| Politique corrigée, pause volontaire de 200 ms, TF 2.15.1 | 90 / 90 | 34 | 1 | 3,08 ms |
| Politique corrigée, rejeu avec mesure du retard de livraison, TF 2.15.1 | 90 / 90 | 0 | 0 | 5,12 ms |

La première politique fermait le contexte trop vite lors des pauses courtes. La politique finale conserve l'audio, autorise au plus **deux inférences sautées sur 32 blocs** au-delà de trois blocs en attente, et ne jette l'ancien contexte qu'au-delà de **huit blocs en attente**. La file de capture contient au plus 16 blocs. Les seuils musicaux et les poids restent identiques.

Sur le dernier rejeu sans perte, le **p99 callback → fin de traitement est 13,27 ms et le maximum 29,18 ms**. Le retard de livraison du producteur WAV atteint 6,91 ms au maximum. Les événements sont identiques à ceux du rejeu direct. **Le précédent essai avec cette même politique a néanmoins perdu 17 blocs : un seul passage sans perte ne certifie pas une stabilité de production.** Les percentiles portent sur les 4 096 dernières observations ; les maxima et compteurs sur la totalité de la séance.

La pause volontaire de 200 ms provoque une reprise propre et **aucune note laissée active**. Son p95 ne représente pas la pause : les blocs abandonnés ne sont pas traités et ne figurent donc pas dans la distribution de latence des blocs conservés. Le nombre de notes égal à la référence ne démontre pas l'égalité de toutes les décisions pendant une coupure.

## Limites et suite Live

La capture est implémentée et ses chemins d'erreur testés, mais **aucune capture du microphone de l'utilisateur ni sortie MIDI matérielle n'a été mesurée ici**. PortAudio n'est pas présent sur cet hôte de calcul. La vérification disponible est le rejeu cadencé, la référence indépendante et les tests de contrats.

Le délai présenté exclut ADC, instant acoustique de l'attaque, éventuelle confirmation du décodeur, synthèse et DAC. Le hop de 5,80 ms et le p95 logiciel ne doivent pas être présentés comme latence musicale bout-en-bout. Les prochaines mesures utiles sont la capture sur le poste de jeu, les réattaques/notes étouffées/accords réellement joués, les pauses longues et une mesure loopback de la sortie choisie. La qualité des hauteurs doit ensuite être évaluée sur plusieurs guitares avec des annotations et des délais causaux, avant d'adapter ou de promouvoir les têtes propres à NOTE.

Reproduction de la comparaison indépendante, après préparation du WAV décrit ci-dessus et checkout du commit `midi` figé :

```text
python scripts/s610_validate_live.py --wav extrait_30s_44100.wav --upstream-root ../midi --output logs/reference_parity.json
```

La CI S610 exécute les 34 contrats et le test du vrai modèle sur silence/signal synthétique. Elle n'ouvre aucun microphone et ne produit aucun score de précision Live.
