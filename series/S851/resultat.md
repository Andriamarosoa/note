---
serie: S851
statut: termine
type: experience
reference: S708_historique
branche: codex/note-s851-identical-guitar-standard-ab256
run_id: 40098180183
commit_sha_execution: a80f2a0b6879d02e517778d84a6d9eaddf808c50
promotion: false
---

# S851 — Résultat du véritable benchmark guitare IA vs standard +12

## Exécution
GitHub Actions [#40098180183](https://github.com/Andriamarosoa/note/actions/runs/40098180183) : SUCCESS technique le 24 octobre 2026. Le test musical échoue pour les quatre sorties.
Même guitare réelle 3,5 s, même transposition standard offline +12 (S848), mêmes 603 décisions PCM256, deux checkpoints réellement entraînés S850 (BCE, RANK), GRU continue ou reset32.
Artefact S851_REAL_GUITAR_STANDARD_BCE_RANK_PCM256, ID 12562115617 : quatre sorties WAV et resultat.json.
Provenance : S847 banque physique #40094119273 ; S848 source + standard #40095759550 ; S850 vrais modèles #40097584744.

## Métriques réellement mesurées

| Variante | Corrélation enveloppe vs standard | Distance log-mel vs standard dB | Note dominante / 603 | Notes distinctes | Acceptée |
|---|---:|---:|---:|---:|---|
| S847 témoin antérieur | 0,1164 | 18,3773 | MIDI73 603 | 1 | Non |
| S850 BCE continue (retenue sur CAL) | 0,2567 | 17,9036 | MIDI73 307 | 3 | Non |
| S850 BCE reset32 (diagnostic uniquement) | **0,3825** | 17,3710 | MIDI74 281 | 4 | Non |
| S850 RANK continue | -0,0867 | 13,9738 | MIDI68 601 | 2 | Non |
| S850 RANK reset32 | 0,1190 | 13,3958 | MIDI68 573 | 2 | Non |

Standard +12 comparé au WAV original : corrélation enveloppe **0,9038**. Ceci concerne une autre paire que la métrique IA vs standard et ne constitue pas une accuracy de note.

## Vérité terrain indépendante S850
Sur guitariste OUTER 05, jamais vu à l'entraînement, la variante BCE +12 a **8,4935 %** de décisions top-1 contenant une note GT ; présence E2→E3 : **26,8071 % F1** (TP1463, FP7782, FN207) ; E3→E4 : **39,9559 % F1** (TP1993, FP4977, FN1013). Ces métriques concernent des présences de hauteurs, **pas le F1 Onset**.

## Audit causal et musical
- Le blocage de S847 sur une note unique est partiellement levé par BCE, mais le résultat ne conserve pas suffisamment la dynamique originale (enveloppe <0,5). RANK demeure quasiment bloqué sur une hauteur.
- La sortie IA est le **rejeu de blocs de vraies notes GuitarSet**, pas une transposition réelle de la performance source. L'enregistrement physique de la banque ne garantit ni timbre, ni phase, ni attaque, ni phrasé original. Un contrôle de hauteur vraie indépendant reste nécessaire.
- Entrée strictement 256 nouveaux PCM à 44 100 Hz à chaque décision (~5,805 ms), pas de futur ni hold. GRU utilise uniquement le passé. Le transposeur standard offline est témoin seulement ; p99 Windows/iRig non mesuré.
- Les six axes NOTE et la chaîne MIDI n'ont pas été changés. Onset de référence 80 WAV **69,2584331 % F1**, inchangé.

## Verdict
**S851 terminée, aucune promotion.** CI SUCCESS ≠ réussite musicale. BCE progresse en diversité et enveloppe par rapport à S847, mais le niveau reste insuffisant ; réévaluer la reconnaissance GT et apprendre à préserver le signal de la performance réelle avant toute promotion.
