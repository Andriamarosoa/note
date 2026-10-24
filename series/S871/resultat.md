# S871 — Forensic audit actual -12 vs merely low

**Statut : expérience locale terminée et non promue.**

| Métrique | Résultat |
|---|---|
| Cible de transposition | -12 demi-tons |
| Justesse 6 notes CAL | audit |
| Justesse 6 notes déjà examinées | 0/6 |
| Descente grossière >=5 demi-tons sur six anciennes OUTER | 5/6 pour S870 seulement |
| Tolérance de la note exacte | ±0,5 demi-ton YIN comparée à l'audio standard réellement mesuré |

Confirmation : 6/6 références standard sont une octave sous les notes originales ; aucune des trois IA n'est correcte à ±0,5 st.

Données : 39 notes réellement enregistrées GuitarSet S847, 27 TRAIN / 6 CAL / 6 anciennes OUTER réutilisées et donc non aveugles. Référence -12 produite offline pour apprentissage et benchmark uniquement. Aucun DSP pitch shift dans le modèle neuronal. Inference : strict PCM256 44,1 kHz, pas de futur, mémoire passée seulement. Six têtes NOTE et moteur MIDI inchangés, aucun modèle promu. Score Onset 69,2584331 % historique non réévalué. Tests de causalité structurels uniquement, p99 Windows/iRig et latence acoustique non prouvés.

**Verdict : REJETÉ.** Les scripts, poids et WAV figurent dans l'archive NOTE_S868_S871_TRANSPOSEUR_AIGU_VERS_GRAVE_PCM256.zip de la conversation ; rapport d'ensemble dans README_TRANSPOSEUR_IA_DOWN12_S868_S871.md.
