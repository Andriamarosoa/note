---
serie: S847
statut: en_cours
type: experience
reference: S708
branche: codex/note-s847-real-recorded-transposition256
run_id: 40094119273
commit_sha: 2cf7d4670abca5d7d8c8070700f5fe6dfc6766ce
promotion: false
---

# S847 — Choix par IA de notes de guitare réellement enregistrées

## Exécution
Nouvelle tête de transposition neuronale causale à PCM256 avec banque de véritables extraits sonores. Alternative au décodeur audio S845, qui avait généré un bruit aigu et échoué à reconstruire les voies hex mieux que le silence.

La sortie audio provient de clips GuitarSet **captés physiquement**, sélectionnés via l'IA. Elle ne correspond **pas encore** à une véritable transposition de tout le morceau : la note et la forme d'onde peuvent ne pas suivre fidèlement l'attaque, le timbre et l'expression de l'entrée. Une banque d'échantillons authentiques fournit un **vrai timbre instrumental**, mais ne prouve pas la justesse de transposition ni le fonctionnement polyphonique.

## Métriques
| Valeur | Résultat |
|---|---|
| Tête S847 −12/+12 sur vrai GuitarSet | **NON MESURÉ** |
| Nombre de notes physiques disponibles en banque | **NON MESURÉ** |
| E2 +12 → E3 / E3 −12 → E2 | **NON MESURÉ** |
| Fidélité de la PERFORMANCE originale | **NON VALIDÉE**, technologie basée sur notes enregistrées |
| F1 E2 du modèle S845 d'origine | **14,31 %** proxy de présence |
| F1 E3 du modèle S845 d'origine | 17,22 % proxy de présence |
| F1 E2+E3 S845 | 10,05 % proxy de présence |
| Onset référence PCM256 | **69,2584331 %** (7 607 TP / 3 093 FP / 3 660 FN), intact |
| Exact-K / Active Exact-K / Offset / Frame Pitch / Full Note | inchangés / non réévalués |
| MIDI officiel | inchangé |

## Comparaison à S708
S708/H0 à 73,252019 % historique modèle entraîné 1024, exclu de toute référence scientifique de cette tête raw PCM256. Il n'existe pas de nouveau score Onset S847.

## Audits
| Domaine | État |
|---|---|
| Données | UNKNOWN : acquisition réelle, SHA/MD5, couverture E2/E3, mapping canaux hex/JAMS |
| Acoustique | PASS code : sortie 256 vrais échantillons de clip, sans DSP pitch-shift |
| Architecture | PASS code : CNN+GRU, shift embedding + tête cible indépendante |
| Apprentissage | UNKNOWN : vrai entraînement GitHub à confirmer |
| Métriques | UNKNOWN : F1 cible −12/+12, continuité waveform et FP |
| LIVE | UNKNOWN : CPU Windows p99 non mesuré, et absence d'intégration MIDI |

## Verdict
En développement, non promu. S846 démontre seulement que le vieux modèle ne produisait pas une guitare transposée audible. S847 doit démontrer que la **sélection est correctement effectuée par l'IA**, pas seulement que la banque contient un son réel.

## Prochaine action
Lancer téléchargement du même GuitarSet certifié, extraction de clips physiquement enregistrés, entraînement CNN+GRU et validation held-out. Sans précision de notes raisonnable et contrôle temporel, ne pas publier une nouvelle démo sonore comme si la guitare avait été transposée.

## Traçabilité du run de recherche
[GitHub Actions S847 #40094119273](https://github.com/Andriamarosoa/note/actions/runs/40094119273). Cette expérience vérifie l'intégrité des sources mic/hex/JAMS avant la banque de vrais clips, l'entraînement et le test extérieur. À la dernière vérification, le téléchargement des archives officielles n'était pas terminé. **Aucun nouveau score ni démo S847 fidèle ne sont revendiqués avant résultat validé.** Tests de conformité PCM256 antérieurs à la phase de téléchargement effectués sans échec.
