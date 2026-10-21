---
serie: S831
statut: termine
type: experience
reference: S708
branche: codex/note-s831-audit-corrections256
run_id: 39640517671
commit_sha: 38708a5c4498580677da49bcbcb0858a0a427b4d
promotion: false
---

# S831 — Correction acoustique et audit complet des propositions Exact-K PCM256

## Objectif
Évaluer sans partage inter-axes trois nouvelles preuves acoustiques propres à Exact-K (attaques, maintien, relâchement), et corriger les confusions graves de S830 : forte confiance due au silence, perte des liens E2/E3, réattaques, exécution de têtes inutilisées et absence de résultats des candidats rejetés.

## Hypothèse
Les preuves harmoniques inter-hauteurs et des vraies attaques répétées peuvent corriger les K2–K6, à condition de ne pas sacrifier les décisions K0/K1. Les propositions individuelles doivent être mesurées par K et décomposées en corrections/régressions plutôt que seulement par une exactitude globale.

## Modifications
- `src/causal_note/s830_private_musical_evidence256.py` : 85 variables par hauteur dont 12 relations inter-hauteurs ; note-positive confidence ; sélecteur local `rescue` ; experts inactifs sautés en mode `s829`.
- `scripts/s831_train_audit_corrected256.py` : cibles Onset/Release depuis événements réels (réattaques préservées), et tous les résultats CAL/OUTER par candidat et par classe K.
- `test/test_s831_audit_corrected256.py` : tests de 96 blocs (tampon FFT 32 blocs), harmonique E2/E3, faux risque silence, réattaque sans trou, experts désactivés.
- Les six caractéristiques officielles et le moteur MIDI ne sont pas modifiés. Les têtes Onset/Release S831 sont internes à Exact-K.

## Protocole
GuitarSet S800 FIT80 signé, 80 pistes, cinq folds 48 TRAIN/16 inner-CAL/16 OUTER compositions disjointes, 306 536 frontières MIDI256 à 44,1 kHz. Comparaison appariée S831/S829/H0, audit de tous les modes. CAL choisit un mode pour chaque composition, sans utiliser OUTER pour sélectionner.

## Audits obligatoires
- Données : PASS, provenance S800 et 80 WAV ; aucune portion CAL/HELD/TEST réservée ouverte.
- Acoustique : PASS synthétique, causalité sur 96 blocs et réattaque/E2/E3 ; généralisation HELD et iRig non testées.
- Architecture : PASS de non-modification des cinq autres axes et d'anti-imports.
- Apprentissage : PASS entraînement réel cinq têtes par fold ; splits disjoints.
- Métriques : PASS 306 536 trames, toutes propositions et régressions par K.
- LIVE : UNKNOWN, contrôle d'inférence synthétique uniquement, pas de p95/p99 total iRig ni poids de déploiement enregistrés.
- Avis des six agents Work et modèles NVIDIA : non obtenus, aucune validation revendiquée.

## Statut
**TERMINÉ — NON PROMU.** [GitHub Actions #39640517671](https://github.com/Andriamarosoa/note/actions/runs/39640517671) PASS (`smoke` et `signed_fit80`) ; artefact `S831_FIT80_AUDIT_CORRECTED_EXACTK_PCM256`, id `12364854124`. Exact-K S831 : **56,9943 % = 174 708/306 536**, inférieur au S829 apparié **57,5247 % = 176 334/306 536**, régression **−0,5304 point / −1 626 bonnes décisions**. H0 **61,6972 %**.
Le gain partiel est réel : expert `blend` K2 40,41 % contre S829 22,02 % ; K5 40,48 % contre 12,85 %. Cependant le même candidat régresse K1 83,37 % → 67,74 %. Les décisions CAL `blend` pour Jazz1 et `rescue` pour Rock3 n'ont pas généralisé ; abandon de leur sélection globale. Continuer par un sélecteur local avec veto contre régressions, calibré uniquement sur CAL, toujours sans partager les cinq autres axes.
