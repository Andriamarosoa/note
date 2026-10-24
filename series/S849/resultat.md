---
serie: S849
statut: termine
type: audit
reference: S708
branche: codex/note-s849-transpose-constant-note-audit256
run_id: 40096750574
commit_sha: 53972b9bae6c2e292e12b2d455d853e18fe96a5a
promotion: false
---

# S849 — Audit saturation MIDI73 de transposition neuronale

## Exécution
L'audit mesure l'ancienne tête S847, le vrai fichier micro d'origine S848, les mêmes 39 notes acoustiques physiques en banque. Les états GRU seront conservés ou diagnostiquement réinitialisés à 8 et 32 blocs. Les scores de note sont mesurés et archivés, sans entraîner ou modifier de modèle.

## Métriques
| Résultat | Mesure |
|---|---|
| S848 cible IA +12 | MIDI 73 = 603/603 blocs |
| S848 dynamique face au standard | 11,64 % de corrélation |
| Standard +12 face à l'original | 90,38 % de corrélation |
| S849 reset8 top1 et variance | **603/603 MIDI73 ; médiane écart type des probabilités 0,002063** |
| S849 reset32 top1 et variance | **NON MESURÉ** |
| Onset F1 PCM256 témoin | **69,2584331 %**, inchangé |
| MIDI et six têtes | inchangés |

## Comparaison à S708
S708 1024 entraînement historique exclu ; seul le chemin S847 natif 256 est audité.

## Preuves réelles

Le [run S849 #40096750574](https://github.com/Andriamarosoa/note/actions/runs/40096750574) est SUCCESS, avec les 603 mêmes décisions S848 et exactement le même checkpoint S847, sans réentraînement. Résultats :

| GRU causale | Choix cible +12 | Choix source non transposé | Écart-type médian des probabilités | Écart-type maximal |
|---|---|---|---:|---:|
| État continu | 603/603 MIDI73 | 603/603 MIDI58 | 0,0004553 | 0,0016388 |
| Reset GRU tous les 8 blocs | 603/603 MIDI73 | 603/603 MIDI58 | 0,0020630 | 0,0091046 |
| Reset GRU tous les 32 blocs | 603/603 MIDI73 | 603/603 MIDI58 | 0,0011778 | 0,0054628 |

**Verdict démontré : un défaut de réinitialisation ne suffit pas à expliquer le blocage.** La variation des probabilités reste petite, et même à horizon d'entraînement 8 le top1 est figé. Un décodeur fabriquant de la diversité ne serait pas une nouvelle compétence du réseau. Nouvelle architecture S850 avec vrai GT multi-note, ranking de négatifs et TRAIN long 32 hops justifiée.

## Audits
| Rôle | État |
|---|---|
| Données | PASS : modèles et WAV authentiques téléchargés |
| Acoustique | code strict 256, sans futur |
| Architecture | ancienne IA intacte |
| Apprentissage | volontairement non effectué |
| Métriques | PASS : 603/603 MIDI73 pour les trois horizons, variance auditée |
| LIVE | latence Windows/iRig UNKNOWN |

## Verdict
**Terminé, non promu.** La diversité des notes à elle seule n'est PAS une preuve de justesse musicale ; même des labels alternés peuvent être faux. La comparaison standard audio reste le témoin musical indépendant.

## Prochaine action
Créer l'architecture S850 dotée d'une vraie supervision contrastive multi-notes à plus long horizon causal si la saturation persiste.
