# S874–S881 | NOTE hybride : STANDARD −12 corrigé par IA

Statut au 24 octobre 2026 : **huit audits/boucles**, performances CPU volontairement mises de côté, **aucun nouveau modèle promu**. La référence de recherche reste S873 pour le son d'une véritable phrase de guitare.

## Scores réellement mesurés
Distance magnitude log-STFT au teacher offline −12 (plus bas = meilleur) ; ce n'est ni une précision musicale en pourcentage ni une mesure d'écoute humaine.

| Modèle | CAL (6 notes) | Anciennes AUDITED (6 notes) | Phrase de guitare 3,5s | Dernière seconde |
|---|---:|---:|---:|---:|
| Standard DSP grain8192 | 0,16548 | 0,20692 | 0,44768 | 0,58300 |
| **S873 hybride (réf.)** | **0,14258** | **0,17657** | **0,40664** | **0,52626** |
| S875 EQ ±20% | 0,15381 | 0,19086 | non mesuré dans ce tableau | — |
| S876 EQ ±40% | 0,14896 | 0,17829 | 0,44501 | 0,60925 |
| S878 S873 + EQ S876 | **0,13530** | **0,15855** | 0,41977 (moins bon) | 0,56701 (moins bon) |
| S880 + augmentation accords | 0,14570 (EQ seul) | 0,17911 | 0,41557 (cascade scratch) | 0,55674 |

6/6 hauteurs CAL correctes (YIN ±0,5 demi-ton) sur S873, S875, S876 et S878 dans ces essais. **Ne pas communiquer « 100 % de capacité à transposer »** : six notes de calibration ne valident pas un morceau complet.

## Enseignement
La cascade S878 améliore de **5,11 %** la distance spectrale sur CAL et 5/6 mélanges numériques de notes, mais détériore une vraie prise (y compris la fin). Elle est REJETÉE. S880 avec 19 accords artificiels TRAIN n'a pas résolu cette régression. S881 a révélé un wrapper qui émet pratiquement uniquement du silence en pseudo-streaming PCM256 ; rejet de l'intégration, pas de Rubber Band natif.

## Méthode et limites
**Contraintes** : transposition de guitare aiguë vers grave −12 demi-tons, 256 nouveaux PCM à 44,1 kHz par appel, audio passé autorisé mais aucun futur. La latence intrinsèque du DSP standard n'est PAS nulle ni auditée ici; la performance CPU a été volontairement ignorée. Six têtes NOTE et MIDI inchangés, aucune promotion. 39 extraits réels GuitarSet S847 : 27 TRAIN, 6 CAL, 6 anciennes AUDITED déjà consultées, donc pas de généralisation indépendante. L'enseignant offline librosa −12 n'est jamais présent à l'inférence.
**Ne pas confondre performance de calcul et retard acoustique.** Les tailles de grains DSP influencent encore un décalage sonore intrinsèque. La mesure comparant des fichiers offline de sources identiques ne valide pas ce retard sur un dispositif réel.

Les rapports individuels sont dans series/S874 à series/S881 : chacun possède resume.md + resultat.md. La branche GitHub contient une tête expérimentale runnable `research/transpose256/s876_neural_eq256.py` et tests, sans checkpoint binaire. **La vraie suite reproductible (entraînement, poids, WAV, JSON et 4 tests passés)** a été archivée dans le fichier conversationnel `NOTE_S874_S881_TRANSPOSEUR_HYBRIDE_IA_AUDITS_PCM256.zip`.

Rappel scientifique : la phase d'un vocodeur et la préservation des transitoires ont des effets audibles; un score spectro-only ne suffit pas à juger une vraie guitare. [Signalsmith Stretch](https://github.com/Signalsmith-Audio/signalsmith-stretch) dispose d'une API streaming native et expose séparément inputLatency/outputLatency ; à tester sans supposer la latence nulle.

**Décision :** S873 référence locale conservée, S875–S881 non promus. Aucun score officiel NOTE ni MIDI recalculé.
