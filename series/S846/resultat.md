---
serie: S846
statut: termine
type: audit
reference: S708
branche: codex/note-s846-transpose-audio-fidelity-gate256
run_id: 40092562504
commit_sha: 31a7409de5cbc729a7fc1306abac36ec3e7c64e0
promotion: false
---

# S846 — Résultat : l'ancienne démo S845 est invalide

## Exécution
Audit du fichier exact `S845_demo_guitare_reelle_original_puis_transposition_IA_plus12.wav` de 7,35 secondes, généré par [GitHub #40091786843](https://github.com/Andriamarosoa/note/actions/runs/40091786843). Contrairement au test de création WAV, le nouvel audit teste le **contenu sonore** du fichier comparé à la guitare réellement enregistrée.

## Métriques acoustiques
| Critère | Vrai son de guitare | Sortie IA +12 |
|---|---:|---:|
| Centroïde spectral | 621,58 Hz | **10979,65 Hz** |
| Médiane spectrale | 199,18 Hz | **11714,06 Hz** |
| Énergie 80 à 250 Hz | **72,87 %** | 1,92 % |
| Énergie >10kHz | 1,57 % | **56,50 %** |
| RMS après normalisation | 0,1798 | 0,1798 |

La présence d'un RMS non nul ne prouve pas une musique réelle ; l'ancienne démo a amplifié une sortie IA très aiguë (facteur ×15,64) et a ainsi caché son insuffisance sonore.

## Métriques des notes et référence
| Axe | Mesure |
|---|---|
| S845 transposeur neural E2 GT par bloc | **14,31 % F1** |
| S845 E3 GT par bloc | **17,22 % F1** |
| S845 E2+E3 GT par bloc | **10,05 % F1** |
| Fidélité sonore transposition +12 | **NON VALIDÉ** (audit spectral fortement négatif) |
| Fidélité reconstruction vraie voie hex S845 | MAE **0,029422** vs sortie silence **0,029205** |
| Onset référence | **69,2584 % F1**, 7607 TP/3093 FP/3660 FN |
| Exact-K | Intact / non réévalué |
| Active Exact-K | Intact / non réévalué |
| Offset | Intact / non réévalué |
| Frame Pitch F1 | Intact / non réévalué |
| Full Note F1 | Intact / non réévalué |

## Comparaison à S708
S708 historique entraîné sur 1024 n'est pas introduit ni promu dans cet audit. Le prototype reste une tête de recherche PCM256, non intégrée au MIDI.

## Audits
| Domaine | État |
|---|---|
| Données | PASS : vraie guitare dans le WAV original |
| Acoustique | **FAIL : sortie IA ultra-aiguë et faible fidélité**, l'ancien « succès WAV » était insuffisant |
| Architecture | garde-fou logiciel ajouté, modèles S845 et six axes MIDI gelés |
| Apprentissage | NON EFFECTUÉ : pas de nouveaux poids |
| Métriques | résultat bruit/fidélité constaté, aucune amélioration Onset |
| LIVE | inchangé, non validé Windows/iRig |

## Verdict
**La démo S845 n'est pas un son de guitare transposé exploitable et doit être retirée des démonstrations de réussite.** Le garde de qualité S846 est nécessaire mais non suffisant pour prouver qu'un nouveau modèle a appris +12. Un traitement passe-bas ou un gain audio ne peut pas faire disparaître ce défaut conceptuel.

## Prochaine action
Vérifier l'alignement micro/hex, puis obtenir des enregistrements physiques de la *même performance transposée* (vraie guitare rejouée), et entraîner explicitement un décodeur latent sonore +12 supervisé sur de vrais sons enregistrés. Mesurer f0, harmonique et dynamique sur jeu OUTER, avec une écoute critique. Ne pas présenter de nouveau fichier démo comme vrai transposeur tant que ces critères ne sont pas satisfaits.

## Validation automatisée du vrai WAV initial
[GitHub Actions #40092562504](https://github.com/Andriamarosoa/note/actions/runs/40092562504) **SUCCESS** après 4 tests. L'audit a retéléchargé exactement la démo précédemment fournie et prouvé : médiane source 199,18 Hz vs sortie 11714,06 Hz (facteur 58,81), énergie au-dessus de 10kHz 1,57 % vs 56,52 %, coefficient de variation d'enveloppe 46,78 % vs 1,10 %. **Trois raisons indépendantes de rejet** ; la règle interdit désormais de publier une démo S845 présentant ce défaut sonore. L'ancien WAV reste un témoin d'échec, pas une démonstration valide. Les six têtes NOTE et le MIDI restent intacts. 
