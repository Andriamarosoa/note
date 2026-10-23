# Onset PCM256 — résultat vérifié et inférence compilée

Cette branche poursuit uniquement Onset, indépendamment des cinq autres
caractéristiques. Chaque appel reçoit **256 nouveaux échantillons** à
44100 Hz et décide à la frontière du bloc courant : **0 audio futur et
0 échantillon d'attente ajoutée**.

Le [run 39915038301](https://github.com/Andriamarosoa/note/actions/runs/39915038301)
a terminé deux replays complets des 80 WAV FIT80 et entraîné dix experts HGB.
Les modèles et seuils retenus sont figés sur CAL avant l'évaluation OUTER.

| Référence comparable | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|
| Contrôle69 historique | 8127 | 4519 | 3140 | 67,971396 % |
| Nouveaux choix HGB/contrôle | 8012 | 4264 | 3255 | **68,062694 %** |

Gain : **+0,091297 point**, avec 255 FP de moins et 115 TP de moins.
SS1 perd 99 TP ; le gain global masque donc une régression sur cette composition.
Le correctif NoteOff seul ajoute 8 FP sans gagner de TP sur FIT80.

- [Résultats par composition et audit](series/ONSET_EVENT_INTEGRITY_HGB256/resultat.md)
- [Rapports et prédictions scellées](series/ONSET_EVENT_INTEGRITY_HGB256/evidence/)
- [Poids sauvegardés, métadonnées et chargement](model/onset256_research_39915038301/)
- [Validation de l'inférence compilée](series/ONSET_COMPILED256/resultat.md)
- [Protocole de compilation et reproduction](series/ONSET_COMPILED256/resume.md)

Installer `requirements-onset256.txt` avec Python 3.11 ou 3.12.
Le runtime `LowLatencyOnsetStream` compile les poids avant réception audio
et partage les observations causales entre experts. Il n'ajoute ni confirmation
sur un bloc futur ni compensation des dates d'événements.

Le score concerne FIT80, déjà utilisé dans plusieurs expériences et partiellement
vu par les modèles historiques S613/S701. Il ne prouve pas une généralisation
externe. Les cinq plis sont des poids de recherche, sans promotion MIDI.
La cadence de 256 échantillons (5,805 ms) n'est pas une mesure de la latence
matérielle de bout en bout. Le score historique S708 à 1024 n'est pas comparable.

La parité de la variante compilée est vérifiée sur les 80 WAV. Le nouveau
benchmark local dépasse néanmoins le budget CPU : p95 compilé 15,335 à
20,513 ms selon le passage. **Le temps réel sans dépassement n'est pas validé.**
