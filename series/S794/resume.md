---
serie: S794
statut: termine
type: experience
reference: S708
branche: codex/note-s794-activek-zero-delay-forensics
run_id: 39336923230
commit_sha: 645639f0180f39b17069d55861189943f98adb4a
promotion: false
---

# S794 — Active Exact-K sans retard

## Objectif

Déterminer pourquoi Active Exact-K S708 reste à **56,2762 %** sur GuitarSet CAL32, dans un audit de vérité du MIDI causal à 256 échantillons. L'objectif de recherche demeure de dépasser 80 % sur six axes sans la moindre latence supplémentaire volontaire.

## Hypothèse

L'erreur Active Exact-K est une erreur du **nombre de notes réellement actives dans les événements NoteOn/NoteOff**, et non simplement un mauvais argmax K du réseau. Des manques de voix distincts peuvent provenir des fondamentales graves, de la polyphonie, de transitions Sustain/Onset ou d'interférences harmoniques. Il faut d'abord quantifier leurs contributions avec les vrais événements avant d'entraîner un nombre relatif de spécialistes.

## Modifications

- scripts/s794_activek_failure_taxonomy.py : diagnostic par K réel K0–K6, sous- et surcomptages sur les 256 échantillons natifs ; registres et signaux acoustiques contemporains candidats.
- scripts/s794_activek_midi256_rootcauses.py : rejeu des flux S708 signés sur GuitarSet FIT80/CAL32 et comparaison exacte des six métriques à S704 et S708 CAL.
- src/causal_note/s793_zero_added_latency.py hérité : aucun événement ne doit être retardé ou modifié et zéro nouveau tampon/lookahead/confirmation.
- test/test_s794_activek_failure_taxonomy.py : vérifications et cas limites avec tableaux synthétiques et preuve d'absence de futur audio.
- scripts/s794_six_role_audits.py : six contrôles indépendants et consensus avant de valider le diagnostic.
- scripts/s794_parity_drift_probe.py : double rejeu CAL32, 0 divergence événementielle entre deux répétitions.
- scripts/s794_source_path_parity.py : comparaison directe ancien chemin S708 et wrapper S733, 32/32 flux identiques.
- scripts/s794_activek_midi256_rootcauses.py a été rétabli sur les fonctions originales `s708_source` et `s708_decode`, sans modifier les règles S708.

Ce S794 est un **audit préliminaire**, pas un nouveau modèle multi-têtes entraîné et pas une promotion. Les quatre familles de signaux acoustiques candidats sont hypothétiques : leurs décomptes sont une couverture indicative et ne démontrent pas une amélioration de F1.

## Protocole

S708/S707 immuables, mêmes fichiers GuitarSet FIT80 et CAL32 que S792. Les six métriques sont vérifiées sur le même flux MIDI sérialisé à 256 échantillons. La vérité terrain est ouverte après génération du MIDI et intervient uniquement dans les audits. À chaque trame, le signal d'analyse acoustique provient exclusivement du dernier fragment reçu (zéro lookahead). La proposition de modèle suivante sera évaluée hors composition FIT, CAL servant seulement à tester un choix gelé.

## Audits obligatoires

- Données : 80 FIT, 32 CAL, SHA et compositions disjointes.
- Acoustique : fondamentales graves, harmoniques et persistance, signaux disponibles dans le présent.
- Architecture : source MIDI S708 inchangée, pas de retard, aucune tête fictive proclamée entraînée.
- Apprentissage : diagnostics descriptifs, aucune optimisation ni sélection sur CAL.
- Métriques : six scores officiels et répartition sous-/surcomptages K1–K6 à 256 échantillons.
- LIVE : aucune modification de la production, aucun échantillon tamponné, aucune mesure matérielle nouvellement revendiquée.

## Statut

**Terminée, [GitHub Actions #39336923230](https://github.com/Andriamarosoa/note/actions/runs/39336923230) SUCCESS, 6/6 audits PASS et consensus PASS.** FIT80/CAL32 complets et signature S708 CAL32 strictement reproduite sur les six métriques/compteurs. Active Exact-K reste **56,2762 %**. Sur CAL32, le diagnostic dénombre **36 890 trames actives sous-comptées**, **6 940 surcomptées**, contre **56 413 correctes**. La détection des voix manquantes en accompagnement est prioritaire ; K5 n'est correct que dans 279/2 971 trames, K6+ dans 49/766. Les indicateurs acoustiques sont des descriptions, pas des modèles entraînés.

Les trois contrôles de reproductibilité ont démontré que le premier échec [#39333999180](https://github.com/Andriamarosoa/note/actions/runs/39333999180) était bloqué par une petite différence avec les compteurs historiques, non par le diagnostic des erreurs. Le run final passe le contrôle sans tolérance. Aucun nouveau modèle n'est promu, aucune latence de décision n'est ajoutée et la latence physique Windows/iRig n'est pas mesurée. Détails dans `series/S794/resultat.md`.
