# NOTE S634–S636 — Transfert du transport LIVE MIDI et métrique de comparaison

**Branche isolée :** `codex/s634-midi-live-transport` (issue de
`codex/s533-s535-note-fundamental-loops`).

**Date d'audit :** 16 octobre 2026. **Statut : tests logiciels réussis ;
pas de promotion de S614, ni de revendication de nouveau F1.**

## But : garder la précision de NOTE et reprendre la fiabilité de MIDI

Sources originelles vérifiées :

- [midi V2.2.1 Windows](https://github.com/Andriamarosoa/midi/blob/main/readme/README.md)
- [midi causal onset evaluator](https://github.com/Andriamarosoa/midi/blob/main/src/polyphonic/causal_event_metrics.py)
- [NOTE S614 original native live](../src/causal_note/strict_live.py)
- [NOTE S632 adaptive cost heads](README_V273_S632_COST_AWARE_ADAPTIVE_LATENCY.md)

Les deux moteurs ont une inférence strictement causale, mais pas les
mêmes protocoles de notation, sources évaluées ou budgets matériel.
**On ne peut pas annoncer NOTE 69,03 % > MIDI 17,51 % comme résultat d'un
test A/B rigoureux.** Ces valeurs proviennent de sources et protocoles
distincts.

## S634 : file d'entrée audio et reprise après discontinuité

Code : [s634_safe_live_transport.py](../src/causal_note/s634_safe_live_transport.py).

Adaptation à NOTE des principes du transport de MIDI :
capture callback non bloquante ; inférence MIDI hors callback ;
file bornée ; abandon de blocs dépassés en cas de backlog ;
validation de la continuité des indices d'échantillons ; panic
NoteOff avant tout redémarrage des états acoustiques ; arrêt propre
lorsque le microphone ne fournit plus de trames.

Pas d'insertion de silence arbitraire, pas de prédiction sur
un historique audio incomplet, pas d'antidatage d'un NoteOn.
Les événements de sortie utilisent les indices absolus des
échantillons reçus et les notes MIDI sont fermées à la reprise.
Cela ne garantit pas en soi que la pile I/O Windows réelle
ne retardera pas les événements.

[Workflow S634 réussi après correction du motif de fermeture](https://github.com/Andriamarosoa/note/actions/runs/38808524226) :
17 tests comprenant le transport, la référence S614 et le runtime S632.

Le premier passage avait détecté un changement de motif de
NoteOff normal `stream_end`→`transport_shutdown`, corrigé pour
préserver **exactement** les sorties S614 sans incident.

## S635 : comparaison NOTE vs MIDI réellement contrôlable

Code : [s635_shared_midi_metric.py](../src/causal_note/s635_shared_midi_metric.py).
CLI : [s635_shared_note_midi_compare.py](../scripts/s635_shared_note_midi_compare.py).

Exigences strictes :

1. même WAV PCM et SHA256 identique communiqué par les deux sorties ;
2. même annotation de notes de référence ;
3. plage MIDI commune 40–76 (NOTE a nativement 40–88) ;
4. même appariement causal : pitch exact, prédiction non antérieure
   à l'attaque de référence, un pour un, réattaque la plus récente ;
5. F1 recalculé indépendamment avec échéances
   **12 / 18 / 24 / 35 / 46 / 80 / 250 ms** ;
6. calcul TP/FP/FN, délai d'émission p50/p95 sans simuler la
   latence physique du microphone.

Le masque 40–76 s'applique aux références **et** aux deux sorties ;
les notes ignorées hors plage sont comptées dans le rapport, pas cachées.

[CI unitaire S635 validée](https://github.com/Andriamarosoa/note/actions/runs/38808959047).

[CI croisée S635 utilisant le vrai matcher du dépôt MIDI validée](https://github.com/Andriamarosoa/note/actions/runs/38809576763),
révision MIDI épinglée
`536536f54290fdc4f63db28aa063d2e324121642`.
Identité des **paires appariées**, faux positifs et notes manquées :
280 vérités/350 prédictions aléatoires,
100 vérités/200 réattaques et
12 vérités/36 propositions d'accords, sur sept deadlines.

**Cette vérification ne constitue pas un nouveau F1 musical** :
il manque encore les deux prédictions sur le même corpus WAV
avec les mêmes labels et même normalisation. Le banc ne fabrique
aucun pseudo-label ou résultat en l'absence de ces fichiers.

Pour rejouer un lot final (format JSON documenté dans le code) :

```bash
export PYTHONPATH=.:src
python -m scripts.s635_shared_note_midi_compare \
  --manifest comparison_frozen.json --out comparison_report.json
```

## S636 : commande optionnelle microphone → NOTE → périphérique MIDI

Code : [s636_note_live_desktop.py](../scripts/s636_note_live_desktop.py).

Prérequis sur Windows, en environnement Python NOTE adapté :

```powershell
python -m pip install -r requirements-live-strict.txt
python -m pip install sounddevice mido python-rtmidi
```

Lister les périphériques disponibles :

```powershell
$env:PYTHONPATH = ".;src"
python -m scripts.s636_note_live_desktop --list-devices
python -m scripts.s636_note_live_desktop --list-midi
```

Faire un essai S614 original sans MIDI connecté :

```powershell
python -m scripts.s636_note_live_desktop --no-midi --duration-s 20 --report-json run_s614.json
```

Émettre vers un port MIDI identifié :

```powershell
python -m scripts.s636_note_live_desktop --midi-port "NOM_DU_PORT" --duration-s 60 --report-json run_live_midi.json
```

Pour activer **l'assistance S632 figée**, récupérer les trois
checkpoints `S632_Q_*_train_only.joblib` dans
[la CI S632](https://github.com/Andriamarosoa/note/actions/runs/38803974784)
et fournir :

```powershell
python -m scripts.s636_note_live_desktop --s632-dir C:\note\S632_models --midi-port "NOM_DU_PORT" --duration-s 60 --report-json run_s632.json
```

Les SHA256 de S614 et des trois poids S632 sont contrôlés avant
d'ouvrir le micro. Le modèle S80 n'est jamais importé ni réentraîné ;
S632 est **optionnel** et reste un assistant, pas une nouvelle référence.

Le callback ne fait aucune inférence ni opération MIDI ; le thread
consommateur produit les messages NoteOn/NoteOff et contrôle les
débordements. À l'arrêt, le système tente de fermer chaque note
active, puis envoie MIDI CC123.

[CI S636 réussie](https://github.com/Andriamarosoa/note/actions/runs/38809326564) :
tests de code, de mode diagnostic, de sortie MIDI simulée,
de parité S614 et de sécurité du transport. **Pas de périphérique
microphone ou de port MIDI physique présent dans cette CI.**

## Résultats et limites pour une promotion

| Domaine | S634–S636 | Validation |
|---|---|---|
| S614 sans perte de blocs | Exactement mêmes événements NoteOn/NoteOff | CI réussie |
| Rupture de capture | Panic, reset, nouvelle époque causale | CI réussie |
| Débordement de file | Drop explicite, aucun faux audio continu | CI réussie |
| NOTE vs MIDI appariement | Exactes paires et mêmes deadlines | CI croisée réussie |
| MIDI port Windows matériel | Commande implémentée | **Non encore testée sur matériel** |
| Inférence réelle du S632 sur Windows | Chargeur SHA + point d'entrée | **Non encore testée sur matériel** |
| F1 même source sur MIDI et NOTE | Protocole prêt, aucun audio annoté commun | **Mesure non exécutée** |
| Rejeu WAV identique MIDI V2.2.1 / NOTE S614 | 59,79 s, 121 / 65 NoteOn | **Exécuté sans vérité terrain** |
| Latence micro→MIDI totale p95 | Métriques I/O/CPU séparées | **Pas de loopback** |
| Régression Exact-K S80 | S80 non modifié | Évaluation fonctionnelle non réexécutée |

**Référence officielle : S614, F1 NoteOn 69,03 % sur son
TEST historique.** S632 progresse légèrement en DEV (63,23 %
GuitarSet, 54,71 % URMP) mais n'est pas promu.

## Suite technique avant la prochaine promotion

Préparer un lot partagé de morceaux annotés que les deux moteurs
n'ont pas utilisé pour sélectionner leurs seuils ; calculer exactement
les mêmes métriques S635 sur les deux sorties sans correction
postérieure ; mesurer les émissions, retards et NotesOff par source.
Sur Windows, capturer un signal réel, le rendre ou réinjecter MIDI,
et mesurer en loopback le délai acoustique→sortie avec p50/p95,
le nombre de pertes et les bascules de surcharge.

Après seulement, essayer un vrai hybride NOTE + politiques d'attaque
de MIDI sur le développement, avec une comparaison contre S614
et une preuve d'absence de régression des cas Exact-K pertinents.


## S637 — Premier rejeu des deux moteurs sur le même WAV réel

[Workflow S637 exécuté et réussi](https://github.com/Andriamarosoa/note/actions/runs/38810059806).
[Rapport JSON original de l'expérience](https://github.com/Andriamarosoa/note/actions/runs/38810059806/artifacts/12007388810).

Fichier source authentique :
`Andriamarosoa/midi/runs/polyphonic/live_poly_v2.wav`
(44,1 kHz mono, **2 636 800** échantillons, **59,791 s**).
Le test GitHub a vérifié son SHA256 :
`dff1b9d1fc6389de65c5f024089221196ebf71090a6bf92f8aed525f0f321d0d`.
Revision MIDI figée :
`536536f54290fdc4f63db28aa063d2e324121642`.

La sortie **déjà produite auparavant** par `midi` V2.2.1 avec
le décodeur prudent `fallback90` est dans
`runs/polyphonic/live_poly_v2_v221_fallback90.mid`.
Son rapport indique **121 NoteOn**.

Le **même WAV** a été envoyé au système original `note`
S614 (SHA256 gelé) via `SafeLiveTransport` S634.
Résultat sur ce fichier : **65 NoteOn**, tous dans la plage
commune MIDI 40–76. Aucun abandon de bloc ni recovery de file ;
le nombre de NoteOff égale le nombre de NoteOn. Sur le runner
GitHub, le calcul et le traitement de la capture simulée ont
pris **14,50 secondes**, contre 59,79 secondes de signal, soit
un rejeu plus rapide que le temps réel sur cette machine CI.

**Interprétation cruciale :** ces 121 et 65 événements ne sont
ni des vraies notes vérifiées ni une métrique F1. Cette capture
n'a pas d'annotations musicales vérifiées et ne démontre donc
ni que MIDI surdétecte ni que NOTE manque des notes. Des
protocoles de silence, de calibration et de sélection différents
peuvent également modifier les émissions. Il faut comparer
leur F1 sur des références humaines/multitracks annotées,
échantillons identiques et tolérances temporelles identiques.

Ce test prouve uniquement qu'un vrai WAV de la pile `midi`
peut désormais traverser le moteur S614 protégé et le nouveau
transport S634, avec une trace structurée et sans perte de bloc.
**Ne pas promouvoir** S634/S636 ou modifier le chiffre de
F1 officiel S614 sur cette base.
