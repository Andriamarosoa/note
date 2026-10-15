# S638/S639 — NOTE vs MIDI : vrai F1 causal sur le même GuitarSet

**Branche :** `codex/s634-midi-live-transport`.

## Protocole gelé

- **Modèle NOTE :** S614 natif original SHA256
  `7ccf046e3a1870eaca3a108e6f9565298aeeadf43732a2f83559542c03f5f17f`.
- **Modèle MIDI :** V2.2.1 TFLite original SHA256
  `4a4df49d34c1fd359fab98c20bb598f37c7f42f22493c70923a2e829176df744`.
- **Code MIDI d'origine :** révision
  `536536f54290fdc4f63db28aa063d2e324121642`.
- **Sources :** GuitarSet WAV microphone réels + annotations JAMS
  `note_midi`, téléchargement vérifié par empreintes SHA256.
- **Prétraitement :** un **unique** WAV de 44 100 Hz mono PCM16 écrit
  une fois, lu par les deux modèles. Aucune retouche temporelle
  ni séparation d'instrument par oracle.
- **Inférence MIDI :** décodeur exact `src.polyphonic.transcribe`,
  paramètres V2.2.1 gelés, sans automatic model level, gain=1.
- **Inférence NOTE :** `StrictLiveStream` S614, sans assistant S632.
  Le S80 historique reste préservé.
- **Score :** S635 common original MIDI causal matcher,
  intervalles de délai 12/18/24/35/46/80/250 ms, NoteOn hauteur
  identique, appariement 1:1 sans reconnaissance anticipée.
- **Plage de référence commune** : MIDI 40–76. Les vérités et
  prédictions hors plage sont comptées séparément, puis exclues
  symétriquement du F1 comparatif.
- Pas d'alignement rétrospectif des sorties MIDI après le fait ;
  les temps sont les temps d'émission du décodeur originel.

### Test pilote — trois compositions DEV distinctes

[CI S638 pilote : réussite](https://github.com/Andriamarosoa/note/actions/runs/38811275351)
[Rapport JSON du pilote](https://github.com/Andriamarosoa/note/actions/runs/38811275351/artifacts/12009170702).

Extraits :
- `04_BN3-119-G_comp` : 135 références, 106 NoteOn NOTE, 109 MIDI ;
- `04_Funk3-112-C#_comp` : 283 références, 285 NOTE, 216 MIDI ;
- `04_Rock1-130-A_comp` : 190 références, 195 NOTE, 92 MIDI.

Trois fichiers audio vérifiés chacun par SHA256 commun entre modèles.

| Score causal 80 ms | NOTE S614 | MIDI V2.2.1 |
|---|---:|---:|
| TP | **329** | 73 |
| FP | **257** | 344 |
| FN | **279** | 535 |
| Précision | **56,14 %** | 17,51 % |
| Rappel | **54,11 %** | 12,01 % |
| F1 | **55,11 %** | 14,24 % |

F1 à 35 ms : NOTE **37,86 %**, MIDI **7,22 %**.

**Résultat direct vérifié :** NOTE S614 est nettement meilleur que
MIDI V2.2.1 sur **ces trois extraits précis** sous le protocole
causal S635 commun, sans mélange de benchmarks historiques.

### Limites cruciales

Ce pilote est un **DEV déjà exposé historiquement** au projet NOTE,
et les intersections exactes avec l'ensemble d'entraînement de
MIDI ne sont pas auditées. Il ne peut donc pas être présenté
comme une nouvelle évaluation indépendante ni comme une preuve
de supériorité pour tout instrument, toute captation ou tout style.

**Ne pas confondre** le F1 original S614 de 69,03 %
sur son TEST historique (tolérance NoteOn ±80 ms symétrique)
avec ce F1 causal retard-only obtenu sur trois extraits
différents. Le protocole S635 exige que le NoteOn émis
n'arrive pas avant l'attaque de référence ; cela explique
une partie de la différence de métriques.

Les décisions de promotion du modèle, les poids S80/S614/S632
et leurs seuils restent inchangés. Il reste à tester de
nouveaux enregistrements non utilisés et à mesurer
la latence microphone→MIDI sur matériel physique.

## Extension S638 et audit S639

Le workflow d'extension S638 rejoue les 12 morceaux
GuitarSet DEV (trois compositions, quatre variations chacune),
sans ajuster les paramètres après le pilote.
Les artefacts comprennent un manifeste avec les prédictions
événementielles et les labels, des statistiques par clip
et les scores aggrégés pour toutes les échéances.

`scripts/s639_audit_same_audio_disagreement.py` examinera
ensuite les vrais événements de référence appariés par les deux
moteurs, par NOTE seulement, par MIDI seulement ou manqués
par les deux, ainsi que les registres graves/médium/aigus et la
densité locale des attaques. La densité locale d'attaques
n'est **pas** l'Exact-K S80 historique.

Résultat final et rapport intégral : à compléter uniquement
après succès de la CI complète, sans extrapolation du pilote.
