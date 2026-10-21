---
serie: S826
statut: en_cours
type: experience
reference: S708
branche: codex/note-s826-exact-s708-native256-rebuild
run_id: en_attente
commit_sha: en_attente
promotion: false
---
# S826 — Reproduction de l'architecture **S708 elle-même**, strict MIDI256

## Objectif

L'utilisateur impose que **chaque extraction, inférence et décision S708 soit déclenchée à 256 échantillons de PCM @44100Hz (5,805ms)**. Il ne demande ni une tête Active-K autonome ni une copie des agrégats S795/S823, ni une distillation vers une autre architecture. S708 = frontend natif 126D, postérieurs préexistants, représentation 192D de contexte causal, spectres sharp 96D, tête temporelle/harmonique 288D multitâche Onset/Sustain/Release + K0..6, **même décodeur GuidedSelector**, même machine NoteOn/NoteOff. Supprimer la décision « 4 × 256 = 1024 samples » de la chaîne, pas le modèle.

## Hypothèse

Le calcul de S708 a deux horloges : l'extracteur CausalPitchFeatures(hop=256) depuis S613 et la fréquence effective de décision via `BATCH=4` dans `S712StrictLiveStream` et `JointSelector.step`. Retirer ce second blocage et calculer la représentation 288D à chaque hop de 256 permet de faire sortir une décision causale tous les 256. Le modèle temporel `s708_selected.pt` a toutefois été **entraîné avec un pas 1024** : ses coefficients et filtres temporels n'ont pas été optimisés à 256 et il est incorrect de prétendre obtenir les scores de S708 sans réentraînement et validation.

## Modifications

Introduire des paramètres `decision_hop_samples` (défaut 4×256 inchangé) dans le décodeur JOINT et le GuidedSelector, `batch_hops` (défaut 4 inchangé) pour le stream S712. Ajouter `S826S708Native256Stream` : héritage du moteur strict S708 d'origine, mêmes `latest_s708_acoustic_features`, `IncrementalS708TemporalHead`, 288 canaux, extraction sharp à chaque 256 avec alignement phase sur précédent hop, `GuidedSelector` configuré avec stride de 256. **Aucun état S795 historique, aucune feature MIDI précédente réinjectée**, mais la machine S708 garde évidemment ses propres notes actives pour le décodeur.

## Protocole

Tests : état antérieur S712 à 4×256 **inchangé**, préfixe causal, découpage callback variable, 49×288 exactement égal aux caractéristiques `acoustic_features(...)` + `sharp_track(audio,256-grid)` hors ligne, six sorties du modèle et événement MIDI sur chaque 256, max 6 notes simultanées, aucun NoteOn/Off daté rétroactivement, lookahead=0/hold ajouté=0. Validation originale sur vrai checkpoint S708 `s708_selected.pt` SHA + policy figée; WAV réel/son synthétique en recherche isolée. Aucune référence promotion avant évaluation six métriques au même protocole source, puis **réentraînement et calibration à 256** avec composition disjoint. Les fenêtres spectrales de 2048,4096,8192 utilisent le **passé uniquement**, elles ne sont pas des décisions différées ou un tampon de futur.

## Audits obligatoires

Séparer conformité **mécanique** (256, 288D, causalité, identité code S708) et **performance apprise** (checkpoint initial entraîné à 1024, puis S708 réentraîné pour 256). Ne pas assimiler 5,805ms de résolution à la latence CPU ou audio matérielle. Vérifier aucun changement des références officielles S708 CAL32, des cinq autres têtes ou du LIVE productif. Calibrer la totalité Onset, Exact-K, Active Exact-K, Offset, Full Note, Frame Pitch sur le même flux de notes. Non promotion jusqu'à six métriques sans régression et validation externe.

## Statut

Construction et tests de la **même architecture S708 strict256**. Aucun score scientifique nouveau validé.
