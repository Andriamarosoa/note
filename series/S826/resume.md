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

Port mécanique S708 entier strict256 validé sur vraie exécution pilote CI #39593842534 (5 pistes, six scores). Original S708 1024 intact et S826 256 branché en SHADOW, pas de promotion car Onset régresse. Réentraînement S708 véritable 256 et validation FIT80 complet encore requis.


## Contrôle réel du checkpoint original — PILOTE 5 pistes GuitarSet, 21 octobre 2026

**GitHub CI [#39593842534](https://github.com/Andriamarosoa/note/actions/runs/39593842534) SUCCESS**, [preuve audio/MIDI JSON et rapport #12343420944](https://github.com/Andriamarosoa/note/actions/runs/39593842534/artifacts/12343420944). Le premier run a échoué sur la recherche de l'archive S800; fix appliqué. Le second run a validé l'identité SHA du checkpoint original `s708_selected.pt`, son `GuidedPolicy` et l'archive GuitarSet WAV source; mêmes **cinq enregistrements d'origine**, un par composition, trois flux MIDI évalués sur **exactement les mêmes trames** : ancien S712-S708 à 1024, nouveau S826 véritable 256 et H0 émis source S800. Ceci est un **pilote de conformité**, pas l'évaluation CAL32 ni FIT80 complet.

| Mesure musicale (score sur les 5 mêmes WAV) | S712 ancien 1024 | S826 port strict256 | H0 S800 MIDI source |
|---|---:|---:|---:|
| Onset F1 | 71,1772 % | **69,5431 %** | 71,2329 % |
| Exact-K | 39,0689 % | **41,8154 %** | 35,6395 % |
| Active Exact-K | 36,8402 % | **39,5940 %** | 33,4461 % |
| Offset accuracy | 56,8889 % | **58,3949 %** | 62,8889 % |
| Full Note F1 | 40,0469 % | **40,1015 %** | 44,3053 % |
| Frame Pitch F1 | 78,5279 % | **80,2813 %** | 76,7727 % |

Le contrôle physique de disponibilité a démontré sur BN1 **3845** décisions/infères temporels réels à 256 vs **961** en mode source 1024, et 1 appel modèle chaque 256 pour les cinq morceaux, pas une duplication de postérieurs 1024. Les mêmes 288 caractéristiques (192D du S708 + sharp 96D) sont recalculées sur la nouvelle horloge **sans regard vers l'avenir**. La tranche de décision est 5,805 ms; les FFT 2048/4096/8192 sont exclusivement des **fenêtres d'historique**. Le portage fonctionne mais **Onset régresse −1,63 point**, donc veto de promotion; le checkpoint original a été appris à cadence 1024 et ses 30 pas temporels représentent désormais une durée historique plus courte.

Le runtime CPU CI pour ces cinq morceaux traite le signal à ~0,62–0,65 × sa durée audio, soit environ **3,6–3,8 ms CPU par hop en moyenne** sur le runner. Cela n'est **pas** une preuve d'absence de deadline miss à chaque 5,805 ms, ni une mesure d'audio carte iRig sous Windows. Il faut auditer p95/p99/max du callback et la latence E2→MIDI matériel. **Les temporisations de seuil héritées de la politique S708 restent présentes** : 35ms d'âge minimal pour certains NoteOff, 65ms pour certains re-onsets et 120ms pour une correction d'attaque guidée. Le portage S826 n'ajoute **aucune nouvelle** temporisation mais n'annule pas ces mécanismes historiques; ils doivent être audités séparément avant de prétendre un décodage totalement sans délai de confirmation.

**État : l'architecture S708 originale a été effectivement portée et testée à chaque 256. Les poids restent les originaux 1024 ; réentraînement à 256 encore NON réalisé.** L'évaluation complète FIT80 80 pistes signées est lancée dans le même protocole, non encore validée au moment de cette section; les six scores CAL32 et le LIVE production restent inchangés. Après FIT80, réentraîner la **même TemporalHeads S708** sur des entrées 288D recalculées chaque 256, calibration composition-disjointe du même GuidedSelector et comparaison des six métriques. Aucun score de 76,88 % (qui était SHADOW S823 et dépendait d'autres entrées) ne peut être imputé à ce portage.
