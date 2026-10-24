---
serie: S842
statut: termine
type: experience
reference: S708
branche: codex/note-s842-real-guitarset-neural-transpose256
run_id: 40083706274
commit_sha: d99beb4186d9faf5eeff3a06ac4a3e5e92e8593d
promotion: false
---

# S842 — Apprentissage IA de transposition d'octave sur vraies notes GuitarSet, PCM256

## Objectif
Expérimenter concrètement la transposition **latente apprise par réseau de neurones**, au service de la détection de notes, particulièrement la reconnaissance de E2, E3 et E2+E3. Ne pas produire de son pitch-shifté, ne pas toucher au MIDI et ne jamais remplacer les scores Onset par des proxys. Le score Onset PCM256 témoin demeure **69,2584331 % F1** sur les 80 WAV GuitarSet (7607 TP, 3093 FP, 3660 FN).

## Hypothèse
Le transpositeur S840 peut apprendre à associer deux **véritables enregistrements indépendants** dont les fondamentales annotées sont séparées de ±12 demi-tons, au lieu de calculer le changement de fréquence. Le réseau reçoit 256 échantillons actuels, transporte des représentations latentes et revient à la hauteur d'origine. Les exemples S841 GT prouvent 153 attaques E2, 278 E3 et 115 intersections de durées de E2+E3 sur 8 enregistrements ; ces intersections ne sont pas nécessairement des accords distincts. Un modèle réellement entraîné pourrait disposer d'une preuve acoustique additionnelle, mais les 256 échantillons ne garantissent pas qu'on puisse identifier une fondamentale E2 à 82 Hz.

## Modifications
`scripts/s842_extract_real_octave_pairs256.py` extrait les seules fenêtres brutes **exactement 256 échantillons**, à partir des WAV mono 44,1 kHz originaux signés et des notes GT avec durées. L'audio est pris pendant qu'une note est la seule active durant le bloc, sans lire de futur à l'inférence. Source et cible proviennent de fichiers originaux distincts, dans une même partition TRAIN/CAL/OUTER disjointe par composition. Les couples ±12 couvrent de multiples notes MIDI 40–88 ; ils ne sont **pas** le même instrument rejoué à l'identique transposé, donc il s'agit de *paires sémantiques faibles* et non de preuve physique de timbre transposé. Aucun DSP pour transformer le son.

`src/onset256/learned_latent_transposer256.py` est corrigé dans cette branche pour **apprendre aussi les poids de fusion de vues** (initialement non entraînés dans S840). Les vues sont désormais les octaves −12 et +12, et le score fusionné est supervisé sur les véritables notes d'origine.

`scripts/s842_compare_neural_e2_e3_probe256.py` compare sur CAL et OUTER le même encodeur CNN avec/sans transposition, en séparant GT E2, E3 et leur vrai chevauchement. Les seuils sont choisis exclusivement sur CAL et gelés avant OUTER. `tests/test_s842_real_octave_pairs256.py` vérifie la géométrie des notes réelles, le bloc brut 256 et le respect des splits.

## Protocole
1. Télécharger et vérifier les 80 WAV et 80 traces S800 par SHA. Former un inventaire de GT actifs **sur toute** la fenêtre 256, sans mélanger attaques simplement proches et vrai sustain.
2. Construire des paires de vraies notes uniques, avec source et cible enregistrées séparément sur des tracks distincts, différence de GT de ±12 demi-tons. TRAIN BN1/Funk2/Jazz1, CAL Rock3, OUTER SS1 ; pas de track partagé entre partitions.
3. Entraîner S840 (transporteur, encodeur, retour aux notes d'origine et fusion) sur les paires réelles : checkpoint choisi uniquement sur CAL; ablation même encodeur sans vues.
4. Mesurer sur OUTER les probabilités de E2, E3 et présence simultanée de E2 et E3, avec prédictions avant de consulter les annotations. Comparer seuils CAL figés et TP/FP/FN. **Ces scores par blocs GT actifs sont des proxys de Frame Pitch**, pas des Onset NoteOn F1.
5. Un échec ou un recul de la vue transposée par rapport au témoin doit être reporté explicitement sans promotion. Même en cas de gain, cette preuve n'autorise pas une correction du MIDI avant apprentissage d'un vrai générateur causal de NoteOn et replay apparié des 80 WAV puis validation externe.
6. Ne jamais utiliser de données de futur audio, d'entrée >256, de poids entraînés à 1024 ou de spectrogramme historique dans cette nouvelle tête. Le temps CPU p99 Windows/iRig devra tenir 5,805ms **pour la chaîne complète** avant promotion.

## Audits obligatoires
| Domaine | Critère |
|---|---|
| Données | vrais audio indépendants et GT actifs pour 256, SHA, TRAIN/CAL/OUTER indépendants |
| Acoustique | uniquement courant 256, pas de FFT historique, zéro future/hold |
| Architecture | CNN+transport+gating appris, contrôle ablation, 5 autres axes inchangés |
| Apprentissage | poids init frais, source/cible paires sémantiques réelles, CAL hors TRAIN |
| Métriques | distinguer Frame proxy E2/E3/conjonction du vrai Onset F1 80 WAV |
| LIVE | aucune écriture MIDI, pas de délai, p99 réel encore UNKNOWN |

## Statut
**Terminé, non promu.** [Run CI #40083706274](https://github.com/Andriamarosoa/note/actions/runs/40083706274) terminé SUCCESS, 8 945 paires audio réelles GT, 8 époques d'entraînement de l'IA. Sur SS1 OUTER, les proxys de classification d'activité ont régressé : E2 **12,22→0 %**, E3 **77,86→66,85 %**, E2+E3 **13,48→0 %**. Il ne s'agit pas de F1 Onset ; son contrôle reste **69,2584331 %** sans changement. Les autres cinq têtes et MIDI restent inchangés. Voir `resultat.md` pour TP/FP/FN complets, succès technique distinct de l'échec scientifique et audit hypothèses ouvertes.
