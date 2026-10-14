# S424 — Conserver le Live et les meilleurs scores MIDI / Exact-K **sans régression**

**14 octobre 2026 — architecture à deux parcours, protection rétrocompatible.**

Le dépôt `note` conserve son ancien moteur causal de traitement des blocs audio et les checkpoints/rapports de son meilleur système de transcription *hors ligne*. Un **nouveau pont MIDI Live facultatif** accepte des événements **déjà prédits** `K, MIDI, horodatage, NoteOff` et peut produire du MIDI sur une sortie physique. **Il n'y a PAS encore de réseau S80 entièrement causal connecté au microphone ni de certification de sa précision en Live.**

## 1. Trois exécutables Live, aucun remplacement silencieux

| Mode | Entrée | Sortie | État |
|---|---|---|---|
| `live-boundaries` | WAV ou PCM16 mono 44 100 Hz, paquets de 512 échantillons | `onset(event-id, sample)` / `offset(event-id, sample)` | **Original préservé sans modification** |
| `live-v8` | WAV ou PCM16 mono 44 100 Hz | Événements de frontière associés à des identifiants | **Original préservé sans modification** |
| `live-midi` | **JSONL de décisions prédites en amont** `S80_K + MIDI + temps` | NoteOn/NoteOff JSONL, port MIDI optionnel et `.mid` | **Nouveau pont causal**, pas de modèle audio inféré en interne |

Modules Live historiques **inchangés** : `scripts/detect_live.py`, `scripts/detect_v8_live.py`, `src/causal_note/pipeline.py`, `src/causal_note/v8_runtime.py`. Le nouveau module ne les importe pas ni ne modifie leurs seuils, associations de slot, checkpoints ou états.

Point d'entrée commun : `scripts/note_runtime.py`, qui délègue le traitement aux scripts originaux **avec les mêmes arguments**. Il n'impose aucune dépendance PyTorch à l'ancien Live.

### Exécuter le Live original

```bash
PYTHONPATH=.:src python -m scripts.note_runtime --mode live-boundaries \
  --model /path/to/causal.keras --stdin-pcm16 --chunk-size 512
```

ou version V8 :

```bash
PYTHONPATH=.:src python -m scripts.note_runtime --mode live-v8 \
  --model /path/to/causal_v8.keras --stdin-pcm16
```

Cela exécute **l'ancien modèle causal**, pas S80/S422. Les statistiques Exact-K S80 n'étaient pas mesurées par ces moteurs originaux.

### Pont MIDI Live K-first (entrée externe de prédictions)

```bash
PYTHONPATH=.:src python -m scripts.note_runtime --mode live-midi \
  --stdin-jsonl --midi-file live_session.mid
```

Le flux de **décisions K et hauteur provenant d'une future tête causale** doit écrire sur stdin un JSON par ligne, par exemple :

```jsonl
{"type":"decision","sample":128,"s80_K":2,"midi_pitches":[48,60]}
{"type":"decision","sample":256,"s80_K":0,"midi_pitches":[]}
{"type":"decision","sample":512,"s80_K":0,"midi_pitches":[],"note_off_ids":["note-000000001"]}
{"type":"decision","sample":768,"s80_K":1,"midi_pitches":[60]}
{"type":"eof","end_sample":1024}
```

Résultat : deux NoteOn à l'échantillon 128, **zéro nouvelle note à K0**, fermeture de 48 pendant K0, réattaque de 60 à 768 (NoteOff de l'instance précédente **avant** le nouveau NoteOn), fermeture de la dernière instance à EOF. Le modèle **ne peut créer plus ou moins de K nouvelles notes**, et les Offset sont indépendants du comptage K. Toutes les sorties sont horodatées **en échantillons**, aucun délai inventé à partir de l'horloge réelle.

Pour envoyer vers un vrai port MIDI :

```bash
PYTHONPATH=.:src python -m scripts.note_runtime --mode live-midi \
  --stdin-jsonl --midi-port "NOM DU PORT"
```

Nécessite `mido` et un backend `python-rtmidi` installés ; le port est libéré et un message All Notes Off est envoyé à l'arrêt. Une entrée terminée sans `end_sample` est refusée : le programme **n'invente jamais** la durée des notes. `--midi-file` écrit un fichier MIDI standard à l'EOF validé, sans dépendance externe.

**Limitation** : `live-midi` **n'accepte pas directement du WAV brut** et **ne calcule pas K**. Il attend des décisions K et hauteurs calculées par une tête externe suffisamment causale. Un flux de démonstration ne prouve pas la qualité musicale Live.

## 2. Scores conservés et protégés (développement hors ligne)

| Référence / challenger | Mesure GuitarSet | Conservée |
|---|---:|---|
| `S80_S397_K5_GUARDED` | **53 529/59 309 = 90,25 % Exact-K global** | Oui |
| `S80_S397_K5_GUARDED` | **6 054/7 385 = 81,98 % Exact-K poly** | Oui |
| `S402_K_CONDITIONED_352D_MLP` | 36,76 % accords poly entièrement exacts | Oui, référence active |
| `S408` | 38,14 % F1 notes complètes | Oui, référence Offset active choisie |
| `S419` | 37,58 % accords poly, challenger | Conservé |
| `S422` | **47,76 % F1 notes complètes** | Oui, meilleur challenger natif hors ligne |

Le manifest **`analysis/v273_live_and_offline_protected_modes.json`** enregistre les identifiants exacts, les comptes bruts et les empreintes Git des **six fichiers intouchables** (quatre Live + référence active + S422).

Le workflow `.github/workflows/v273-s424-preserve-live-and-offline-best.yml` vérifie :
- qu'aucune des empreintes **Git blob SHA1** originales n'a changé ;
- que les références K S80, hauteur S402, Offset S408, meilleur challenger S422 sont **identiques** ;
- que les tests unitaires historiques Live/pipeline sont toujours valides ;
- que le K0 peut conserver une note active, qu'un NoteOff peut être émis pendant K0, que le **retrigger** ferme une instance avant d'ouvrir la suivante ;
- que les commandes invalides sont rejetées sans mutation d'état, le flux respecte K0..K6 et le véritable fichier `.mid` est produit ;
- que le pont ne présente jamais les scores hors ligne comme des scores Live.

Les 90,25 % / 81,98 % / 47,76 % sont des **résultats de développement GuitarSet connus avec recoupements historiques d'entraînement** : conserver le code et le score ne constitue pas une validation indépendante ni une garantie de précision au microphone.

## 3. Architecture Live de production restante (non réalisée)

```text
Microphone -> tampon audio causal -> S80 K vraiment causal [PAS ENCORE INTEGRE]
                                      |
                              K0..K6 [figé]
                                      v
       tête pitch causale K-conditioned [PAS ENCORE INTEGREE]
                                      |
                          midi_pitches = exactement K
                                      v
                        LockedKLiveMidi [IMPLEMENTE]
                                      |
                      état par hauteur / NoteOn / NoteOff
                                      |
                            port MIDI + fichier .mid
```

Il faut **réentraîner/adapter** les têtes temporelles S407/S419/S422 aux fenêtres causales avant un véritable bout-en-bout. S422 voit jusqu'à **5 s de futur** ; le brancher tel quel produirait une latence d'au moins ce contexte pour certaines décisions, ce qui n'est pas le Live demandé.

Les règles de promotion restent strictes : même comptage S80 (aucune régression Exact-K), comparaison MIDI Onset+Pitch+Offset sur des morceaux inconnus, latence de capture/inférence/décodage mesurée **sur matériel**, p95/p99, tests K0/réattaques/étouffement/notes longues, et validation indépendante. Les métriques de l'ancien Live ne peuvent pas être comparées directement au F1 hors ligne de S422 sans même échantillon et même protocole.

**Décision :** deux parcours conservés sans perte de code/score ; pont Live MIDI fonctionnel sur **entrées de notes K déjà prédites**, sans promotion abusive d'un modèle Live S80 ou S422 non testé.
