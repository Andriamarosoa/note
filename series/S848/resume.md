---
serie: S848
statut: en_cours
type: experience
reference: S708
branche: codex/note-s848-ia-vs-standard-transpose256
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S848 — Comparer le vrai transposeur IA à une transposition standard (+12)

## Demande et preuve attendue
L'utilisateur demande une comparaison **sur exactement la même prise originale de guitare**, entre transposition conventionnelle +12 demi-tons et transposition IA +12. C'est une comparaison de référence valable et beaucoup plus informative qu'un simple test « fichier WAV non vide ». Le transposeur standard est utilisé UNIQUEMENT en test indépendant, pas pour l'inférence IA ni pour fabriquer des données d'entraînement. Une similarité sonore objective n'est pas une preuve absolue d'une meilleure qualité mais une preuve de proximité avec la référence.

## Version évaluée
S847 : réseau CNN brut PCM256 + GRU passé, code d'intervalle de 12 demi-tons, décodeur de probabilités des notes de destination, puis copie de sons de guitare **réellement enregistrés et certifiés GuitarSet** depuis la banque de 39 notes (E2/E3 inclus). Le son n'est **pas** une transposition fidèle de la performance, mais une sélection de notes authentiques entraînée sur JAMS. Le benchmark standard est `librosa.effects.pitch_shift(original, sr=44100, n_steps=12)`, méthode DSP conventionnelle offline (référence non causale, sans promesse de latence live). Les deux sont produits à 44100 Hz et même longueur, à partir du même extrait original.

## Mesures prévues
- Trois WAV indépendants **original**, **standard +12**, **IA +12** et un seul A/B/C d'écoute, avec segment égal et silence de séparation.
- WAV source de vraie guitare confirmé par git blob SHA : `e1e382b27dbdc5ec30e796ca39405a03742b86b5`, extrait mono GitHub `pdx-cs-sound/wavs/main/gc.wav`. Originalité du son garanti par enregistrement archivé et contrôle de source, pas par IA.
- Centroïde/médiane du spectre, parts d'énergie 80–250Hz et >10kHz ; énergie/absence de sortie et décisions de notes choisies.
- Corrélation des enveloppes RMS sur les frontières PCM256, distance log-mel dB entre IA et standard. Ces chiffres ne sont pas des F1 Onset, ni une certification d'octave juste à eux seuls.
- Choix du seuil IA **0,23 déclaré préalablement depuis CAL S847** ; aucune sélection après observation des résultats OUTER ; ne pas utiliser la sortie standard pendant l'inférence.
- S847 mesure déjà sur held-out +12 la cible E3 (source E2) : **14,74365 % F1** de notes actives (TP 3123 / FP 35352 / FN 766), preuve de gros défauts de sélection, pas une réussite.

## Contraintes
- Entrée IA **exactement 256 nouveaux PCM** par frontière, état GRU du passé uniquement, zéro attente et aucune écoute de micro futur ; la référence standard de comparaison offline **n'est pas** soumise à ces contraintes et ne doit pas être présentée comme comparable en latence.
- Le standard peut utiliser un pitch shifter DSP pour SERVIR DE TÉMOIN, jamais pour entraîner/générer le son S847.
- Les notes de banque viennent des vraies prises originales GuitarSet et non du synthétiseur ; mais il s'agit de clips rejoués, non d'un décodeur neuronal audio de même performance transposée.
- Tests physiquement auditables et rapport, impossible de prétendre victoire si le modèle produit silence, clics, mauvaises notes ou un son très différent.
- Préserver toutes les caractéristiques Onset/Offset/Exact-K/Active Exact-K/Frame Pitch/Full Note, référence F1 Onset **69,2584331 %** et MIDI officiel.

## Statut
Expérience non promue, comparaison en cours. Précédente démo S845 invalidée par S846 (médiane de spectre sortie 11,7kHz et 56,5 % énergie >10kHz). Toute sortie similaire, inaudible ou sémantiquement incorrecte doit être explicitement rejetée.
