---
serie: S829
statut: termine
type: experience
reference: S708
branche: codex/note-s829-s809-exactk-native256
run_id: 39624926356
commit_sha: 655d14bba15c028b257f6c01a701e086f715b2c3
promotion: false
---

# S829 — Exact-K 256 échantillons, architecture indépendante

## Objectif
Tester une chaîne d'Exact-K indépendante de bout en bout : PCM 44,1 kHz → DSP causal possédé par l'axe → deux classifieurs propres à l'axe (harmonique et temporel) → sélection privée calibrée → décision K0–K6 toutes les 256 valeurs PCM (5,80 ms de cadence). Conserver les cinq autres axes et le MIDI officiel strictement intacts.

## Hypothèse
Une architecture sans checkpoint S708/S810/S828 ni états/postérieurs provenant d'un autre axe peut apprendre Exact-K avec des signaux acoustiques multirésolutions privés. S809 à 1 024 et S828 Active-K natif256 restent des témoins historiques avec métriques ou protocoles différents, non interchangeables.

## Modifications
Création d'un frontal PCM/FFT propre, de deux modèles HGB propres, d'un sélecteur calibré, et d'un chemin d'inférence en ligne. Élimination des imports et utilitaires de notation provenant des autres caractéristiques. Aucun fichier du détecteur LIVE de production, ni des six modèles historiques, n'a été modifié ; PR #133 reste DRAFT. La seule ressource commune est le PCM brut et les bibliothèques mathématiques génériques.

## Protocole
80 pistes GuitarSet authentiques et vérifiées SHA dans FIT80 S800 ; cinq groupes de composition. Pour chaque groupe OUTER, entraîner sur trois compositions, choisir le sélecteur avec la quatrième, évaluer sur la cinquième : 48/16/16 pistes. Métrique Exact-K calculée sur **toutes** les 306 536 décisions y compris K0, sans filtrer les silences. Comparaison paire avec H0 historique sur les mêmes trames. Les événements historiques et annotations ne sont pas des entrées d'inférence. Ces FIT80 historiquement explorés ne suffisent pas comme généralisation externe HELD.

## Audits obligatoires
- Données : SHA audio/manifeste, 80 pistes, cinq groupes de composition disjoints — PASS sur GitHub.
- Acoustique : causalité et invariance au découpage sur signaux synthétiques — PASS limité ; coût matériel non mesuré.
- Architecture : imports et chemins isolés — PASS en CI ; perturbations adverses multi-axes non réalisées.
- Apprentissage : entraînement FIT et calibration disjoints — PASS sur les groupes et l'exécution.
- Métriques : résultats vérifiés K0–K6, comparateurs au même dénominateur — PASS.
- LIVE : pas de preuve p95/p99 sur iRig, ni de déploiement/métriques six-axes — UNKNOWN.
- Trois avis NVIDIA authentiques et agents Work : non exécutés, aucun consensus 9/9 déclaré.

## Statut
**Terminé, non promu.** Run [#39624926356](https://github.com/Andriamarosoa/note/actions/runs/39624926356), commit de code `655d14bba15c028b257f6c01a701e086f715b2c3`. Exact-K 57,5247 % (176 334 / 306 536), H0 61,6972 % sur les mêmes trames ; régression −4,1724 points. Sous-comptages 83 523, surcomptages 46 679. La priorité est de corriger les K2–K6 sans mutualiser aucune composante. Pas de changement des références officielles.
