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
# S826 — Réexécution S708 entière sur grille d'extraction, inférence et événements MIDI256

## Exécution

Moteur S708 complet strict256 à construire et tester sur checkpoint d'origine SHA vérifié; **le premier checkpoint est entraîné à une cadence 1024**. Résultats du modèle adaptés à 256 NON MESURÉS.

## Métriques

| Évaluation | S708 référence CAL32 | S826 strict256 |
|---|---:|---|
| Onset F1 | 74,2639 % | Pilote 5 pistes 69,5431 % ; CAL32 NON MESURÉ |
| Exact-K | 62,0455 % | Pilote 5 pistes 41,8154 % ; CAL32 NON MESURÉ |
| Active Exact-K | 56,2762 % | Pilote 5 pistes 39,5940 % ; CAL32 NON MESURÉ |
| Offset accuracy | 77,3292 % | Pilote 5 pistes 58,3949 % ; CAL32 NON MESURÉ |
| Full Note F1 | 57,1319 % | Pilote 5 pistes 40,1015 % ; CAL32 NON MESURÉ |
| Frame Pitch F1 | 76,4626 % | Pilote 5 pistes 80,2813 % ; CAL32 NON MESURÉ |

## Comparaison à S708

S708 ancien : extraction des 126D natifs chaque 256 mais 288D et tête temporelle+GuidedSelector tous les 1024. S826 : même architecture tous les 256, sans dépendance à un nouveau modèle S825. **La parité des poids ne garantit pas parité des scores quand l'horloge change : réentraîner la tête S708 avec des séquences à 256.** S823 et ses 76,8775 % ne sont pas un score de la même chaîne S708 strict256.

## Audits

| Domaine | Statut | Objectif |
|---|---|---|
| Données | PASS pilote | 5 morceaux originaux GuitarSet SHA 44.1kHz, test FIT80 80 en cours |
| Acoustique | PASS pilote | Égalité originale 192D + 96D chaque 256, pas de futur |
| Architecture | PASS pilote | Checkpoint S708 original et même GuidedSelector; 1 inférence/256 sur les vrais WAV |
| Apprentissage | INCOMPLET | Checkpoint S708 signé original entraîné à 1024 : réentraînement native256 requis |
| Métriques | PASS pilote | Six scores sur 5 morceaux S708 ancien / S826 256 / H0; CAL32 inchangé |
| LIVE | UNKNOWN | CPU CI moyen mesuré, pas de métrique temps microphone ni p99 |

## Verdict

Pas de promotion avant preuve complète et réentraînement 256; ne pas présenter le contrôle 1024 comme une reproduction 256.

## Prochaine action

Évaluer la parité à 1024 et la cadence à 256 sur un vrai checkpoint; puis réentraîner la **même** TemporalHeads S708 avec `decision_ends=np.arange(256,N+1,256)` et recalibrer GuidedSelector sans hold ajouté, comparer les six scores, K0–K6, la latence réelle et les non-régressions.


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
