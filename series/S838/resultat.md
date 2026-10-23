---
serie: S838
statut: termine
type: experience
reference: S708
branche: codex/note-s838-fragment-risk-gates256
run_id: 40000089112
commit_sha: 4448f460a656a3c524114f18a6b7385e96fb74f4
promotion: false
---

# S838 — Rapport signé : apprentissage des sélecteurs de K par fragment

## Exécution
[GitHub Actions #40000089112](https://github.com/Andriamarosoa/note/actions/runs/40000089112) **SUCCESS**, 36 tests. Artifact `S838_REAL5_PRIVATE_K_FRAGMENT_GATES_TRAIN_CAL_OUTER` (ID 12521936945). Sur cinq vrais enregistrements GuitarSet authentiques (1 par composition), 20 062 frontières causales PCM256 à 44,1 kHz. Aucun modèle de production, aucune source S708/NOTE, aucun événement MIDI et aucune autre tête modifiée.

## Métriques
| Axe SHADOW sur 5 WAV | Témoin S834 | S838 sélecteur par fragment | Écart |
|---|---:|---:|---:|
| Exact-K accuracy | 56,599541 % (11 355 / 20 062) | **55,926627 %** (11 220 / 20 062) | −0,672914 point |
| Active Exact-K accuracy | 46,772514 % (8 985 / 19 210) | **47,241020 %** (9 075 / 19 210) | +0,468506 point |

Les deux têtes ne franchissent pas 80 %. Les quatre autres métriques S837 restent **inchangées par ce code** : Onset F1 témoin 68,4475 %, Offset F1 témoin 18,1818 % (candidat isolé S837 à 49,3175 % NON promu), Frame Pitch F1 témoin 84,9695 %, Full Note F1 témoin 67,7172 %. Ne pas additionner ces scores comme une seule sortie MIDI.

### Exact-K — les erreurs récupérées ne compensent pas les pertes SS1
Les CAL BN1/Funk2/Jazz1/Rock3 choisissent de **ne pas activer** le gate sur leurs OUTER correspondants. Seul OUTER SS1 est modifié après calibration sur BN1 (55,40→57,32 % CAL). Le gate modifie 1 251 fragments SS1, récupère 231 K corrects, en perd 366, soit **−135 nets** :
- K1 : +22/−83 = −61.
- K2 : +73/−116 = −43.
- K3 : +110/−0 = +110.
- K4 : +1/−119 = **−118**.
- K0 : −5 ; K5 −32 ; K6 +14.
- 108 décisions modifiées restent fausses avant et après.

**Cause prouvée :** la calibration de risque agrégée sur BN1 ne transfère pas de manière fiable au profil K1/K4 de SS1, malgré de bons cas K3. **Cause harmonique acoustique non prouvée**, nécessitant inspection des spectres. Le gate Exact-K est REJETÉ.

### Active Exact-K — un petit gain à préserver comme nouvelle sélection
Gains nets de blocs corrects par OUTER : BN1 **+20**, Funk2 **+67**, Jazz1 **−2**, Rock3 **+2**, SS1 **+3**, total **+90**.
- BN1 : 21 blocs récupérés, 1 perdu, 102 fragments commutés.
- Funk2 : 214 récupérés, 147 perdus ; **214 fragments modifiés toujours faux**. Gain local réel mais beaucoup de sélection non utile.
- Jazz1 : 119 récupérés, 121 perdus ; régression petite mais réelle.
- Rock3 : 4 récupérés, 2 perdus ; sélecteur conservateur.
- SS1 : 25 récupérés, 22 perdus, 616 fragments commutés.

La correction Active-K est une candidate SHADOW **à auditer plus profondément et valider sur d'autres pistes**, pas un remplacement automatique. Voir les tableaux `per_true_k` et les exemples échantillon par échantillon dans l'artifact et dans [audit_loop1.md](audit_loop1.md).

## Comparaison à S708
Modèle-source S708 appris historiquement à pas 1024 ; les fragments réévalués 256 ne sont pas une preuve de réentraînement natif. S708/H0 FIT80 80 pistes Onset F1 73,252019 % et Offset accuracy 69,886229 % restent protégés ; impossible de comparer ces nombres à l'Offset **F1 événement** S837 ou à seulement cinq WAV. Aucune sixième tête n'est modifiée par S838, aucune fusion MIDI.

## Audits
| Rôle | État | Interprétation |
|---|---|---|
| Données | PASS | 5 WAV signés, 5 compositions originales ; 75 autres pistes non jouées |
| Acoustique | PASS | 20 062 échantillonnages de décision PCM256, sans futur |
| Architecture | PASS | deux banques K et sélecteurs privés, quatre autres têtes gelées |
| Apprentissage | PASS | TRAIN 3, CAL 1, OUTER 1 composition ; aucun usage OUTER pour seuil |
| Métriques | FAIL | Exact-K −135 blocs corrects, Active-K +90, aucun ne dépasse 80 |
| LIVE | UNKNOWN | coût du gate par hop sur Windows/iRig, p99 et MIDI unifié non validés |

## Verdict
**TERMINÉ EN RECHERCHE ; NON PROMU.** L'hypothèse de sélection fragmentaire permet un petit gain Active-K mais ne supprime pas le défaut de généralisation Exact-K. Ce n'est pas un record de qualité Live.

## Prochaine action
Auditer le profil spectral des K1/K4 SS1 et apprendre/calibrer des familles de risque distinctes en utilisant exclusivement les compositions TRAIN/CAL ; valider sur de nouveaux WAV, ne pas conserver une règle parce qu'elle marche sur le SS1 déjà examiné. Développer séparément des têtes Onset et Offset avec modèles de veto FP/FN et zéro attente. Ensuite seulement répéter 80 WAV signés et tester le budget p95/p99 sur Windows/iRig. Aucun agent NVIDIA/Work personnalisé n'a été exécuté ici.
