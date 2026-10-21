---
serie: S824
statut: en_cours
type: experience
reference: S708
branche: codex/note-s824-s795-feature-family-ablation
run_id: en_attente
commit_sha: en_attente
promotion: false
---
# S824 — Résultats des ablations par famille de variables S795

## Exécution

Relecture des 80 traces d'origine S800, sans redécompresser de PCM et sans tourner le modèle S708. Les observations utilisées à MIDI256 sont les snapshots déjà disponibles à 1024 (max 768 échantillons d'âge). Scores authentiques uniquement après run GitHub validé.

## Métriques

| Source FIT80 | Active Exact-K | Commentaire |
|---|---:|---|
| H0 MIDI256 | 57,019538 % | MIDI complet historique, même dénominateur |
| S823 23 variables S795 à 256 | 76,462145 % | SHADOW, dépend S708+MIDI |
| S824 FULL23 contrôle de reproduction | NON MESURÉ | Doit être identique S823 à l'unité |
| S824 WITHOUT MIDI, WITHOUT S708, ACOUSTIC ONLY et 8 autres masques | NON MESURÉ | Abrasion sous même protocole |

## Six axes officiels S708 CAL32

| Caractéristique | S708 officiel | S824 |
|---|---:|---|
| Onset F1 | 74,2639 % | inchangé |
| Exact-K | 62,0455 % | inchangé |
| Active Exact-K | 56,2762 % | expériences SHADOW FIT80 séparées |
| Offset accuracy | 77,3292 % | inchangé |
| Full Note F1 | 57,1319 % | inchangé |
| Frame Pitch F1 | 76,4626 % | inchangé |

## Comparaison à S708

Aucun poids S708, aucune émission MIDI source ni moteur LIVE n'est changé. L'ablation ne transforme pas les vieilles features 1024 en une DSP audio 256; les ablations issues d'autres têtes **ne garantissent pas l'indépendance des architectures**.

## Audits

| Rôle | Statut | Motif |
|---|---|---|
| Données | UNKNOWN | SHA S800 et H0 FIT80 |
| Acoustique | UNKNOWN | Contrôle audio 16..22 du modèle S708, pas nouveau frontend indépendant |
| Architecture | UNKNOWN | 11 masques préenregistrés et fit privé |
| Apprentissage | UNKNOWN | mêmes 48 TRAIN/16 inner-CAL/16 OUTER |
| Métriques | UNKNOWN | identité S823 190705/249411 et delta par masque |
| LIVE | UNKNOWN | strict zéro futur/hold, pas de mesure hardware |

## Verdict

Pas de promotion quels que soient les gains. FIT80 est déjà exploré et aucune autre métrique MIDI final n'est mesurée.

## Prochaine action

Comparer contributions des trois familles S708, MIDI et acoustique; retenir les seules bonnes idées transférables à une vraie architecture de cardinalité indépendante audio MIDI256, à valider sur données externes gelées.
