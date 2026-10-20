---
serie: S807
statut: termine
type: experience
reference: S708
branche: codex/note-s807-blind-activek-three-model-review
run_id: 39378681127
commit_sha: 07aa63dbad13dba4922a16b22da6b2b920a06061
promotion: false
---

# S807 — résultats vérifiés, contre-audit des modèles et nouvelles preuves FIT80

## Exécution

Premier run [#39377951484](https://github.com/Andriamarosoa/note/actions/runs/39377951484) : SUCCESS protocole, 0/3 avis de qualité acceptables. Gemma a produit une réponse fondée en partie sur de faux chemins ; Nemotron a rendu un contenu final vide ; Poolside HTTP503.

Run correctif unique [#39378681127](https://github.com/Andriamarosoa/note/actions/runs/39378681127), SHA scientifique `07aa63dbad13dba4922a16b22da6b2b920a06061` : SUCCESS protocole, **1/3 avis substantiel selon le validateur de citations** (Nemotron), Gemma 0 chemin de fichier littéral validé, Llama 70B HTTP410. Le changement de configuration Nemotron `chat_template_kwargs.enable_thinking=false` a permis de recevoir une réponse finale. Sur ces deux appels, le prompt et le contexte FIT80 étaient **identiques** (SHA `c8f933486dd91c826cb17e1c1ced2c47e55ac895f0c893f177d3ef3e853571d4` et `dcba23763f0caa41a775fe88d8f8d2f0a41268d12125329ac17e1fe00417761d`).

La validité scientifique des avis est **NON VALIDÉE**, indépendamment de leur prose. Les trois modèles n'ont jamais communiqué entre eux ni reçu une architecture à soutenir. Les six agents personnalisés Work n'ont pas été lancés sous leur nom. Les deux runs ne constituent pas neuf audits PASS.

## Métriques

Les valeurs GuitarSet CAL32 officielles S708 restent :

| Métrique | S708 officielle CAL32 | Après S807 |
| --- | ---: | --- |
| Onset F1 | 74,2639 % | NON MESURÉ |
| Exact-K | 62,0455 % | NON MESURÉ |
| Active Exact-K | 56,2762 % | NON MESURÉ |
| Offset accuracy | 77,3292 % | NON MESURÉ |
| Full Note F1 | 57,1319 % | NON MESURÉ |
| Frame Pitch F1 | 76,4626 % | NON MESURÉ |

La source FIT80 indépendante S794/S800 (les scores ne sont **pas** interchangeables avec CAL32) contient exactement 249 411 trames actives, 142 213 correctes, 80 654 sous-comptées, 26 544 surcomptées et 146 984 trames-pitch positives manquées. Les sous-comptages font 75,24 % des **107 198 erreurs d'active count**. Le critère restrictif d'indice acoustique S794 ne couvre que **21 500 / 80 654** sous-comptages (26,66 %), ce qui ne démontre PAS l'absence physique d'énergie sur les autres.

### Nouvelle partition indépendante des vrais événements MIDI256

L'analyse locale des **80 véritables JSON S800 FIT80**, menée sans modèle NVIDIA et avec parité `active_frames`, `active_k_correct`, `frame_pitch_fp`, `frame_pitch_fn` piste par piste, donne :

| Classe de trames-hauteur réelles manquées | Comptage | Part |
| --- | ---: | ---: |
| Note entièrement absente de la timeline prédite | 50 442 | 34,32 % |
| Début de note non couvert (prédiction tardive) | 48 957 | 33,31 % |
| Fin de note non couverte (prédiction terminée tôt) | 29 525 | 20,09 % |
| Interruption interne d'une note | 18 060 | 12,29 % |
| **Total** | **146 984** | **100 %** |

Ces types partitionnent les **trames-hauteur manquées**, non les trames Active-K incorrectes ; plusieurs hauteurs peuvent être manquées simultanément. Une « fin non couverte » est une relation temporelle, pas la preuve que le code NoteOff particulier a été fautif. Les 835 trames avec double annotation de même hauteur constituent un possible confondant limité (758 d'entre elles sous-comptées), pas un plafond d'Active-K démontré.

Reproduction : `python -m scripts.s807_midi_gap_diagnostic --traces-zip s800_fit80_traces.zip --out /tmp/s807_gap.json`. Résultat signé : `series/S807/fit80_pitch_gap_summary.json` et ZIP GitHub [S800 #39352880035](https://github.com/Andriamarosoa/note/actions/runs/39352880035).

## Comparaison à S708

Aucune mutation de S708, S707, S795, S797 ou LIVE, et **aucun score amélioré revendiqué**. Les modèles ont discuté de prototypes non promus ; leurs idées ne remplacent pas un replay authentique des six métriques à latence inchangée.

## Audits

| Audit | Statut | Preuve |
| --- | --- | --- |
| Données | UNKNOWN | Rejeu manuel S800 80/80 compteurs ; pas un audit CI indépendant S807 |
| Acoustique | UNKNOWN | Pas encore de WAV original dans ce contre-audit |
| Architecture | UNKNOWN | Revues externes comportant des erreurs de référence |
| Apprentissage | UNKNOWN | Aucun nouveau modèle entraîné |
| Métriques | UNKNOWN | Parité FIT original ; aucun nouveau MIDI candidat |
| LIVE | UNKNOWN | Aucun changement LIVE ; pas de nouvelle mesure physique |

Contrôles particuliers PASS : preuves GitHub signées FIT80, source MIDI d'origine, tests unitaires de contre-exemples de S807 et distinction des modèles réels. **Consensus de recommandations valides 0/3**, malgré 1 avis substantiel au sens textuel. Cela ne constitue PAS un 9/9 PASS.

### Critique technique des propositions externes

- **Hystérésis S763 : REJET comme explication directe de S708.** Le code `src/causal_note/s763_independent_decoders.py` expose `ActiveKDecoder` shadow ; le compteur officiel est reconstruit depuis les vrais `NoteOn/NoteOff` dans `scripts/s704_joint_metrics.py`.
- **Routeur S797 : REJET comme cause historique officielle.** `src/causal_note/s797_fragment_router.py` est expérimental ; l'identité est la source S708 et ses variantes S803, S805 et S806 n'ont jamais fourni un candidat sans régression.
- **Lissage gaussien S741 : NON DÉMONTRÉ.** Une distribution discrète `exp(-(x-K)^2/(2 sigma^2))` garde le même argmax au plus proche de x lorsque sigma change, si considérée isolément. Une fusion ultérieure pourrait changer, mais devrait être mesurée avec un vrai décodeur.
- **Modifier la grille ou supprimer les frames polyphoniques : REJET comme preuve de progrès.** C'est changer la convention d'évaluation ou le dénominateur, pas nécessairement corriger des événements MIDI physiques. La grille 4 × 256 / 44 100 vaut **23,22 ms**, pas 11,7 ms comme l'indique Nemotron.
- **Confusion de chemins : REJET.** Nemotron place faussement `ActiveKDecoder` dans `s795_activek_experts.py` et Gemma cite notamment `scripts/s763_independent_decoders.py` au lieu de `src/causal_note/s763_independent_decoders.py`. Des citations de fichiers vrais ne suffisent pas à prouver la phrase associée.

## Verdict

**S807 est terminé en tant qu'audit contradictoire, sans solution promue.** Premier constat réellement nouveau : les attaques et débuts absents forment 67,6 % des trames-hauteur manquées, les suffixes et interruptions 32,4 %. Il faut étudier **les deux mécanismes** et distinguer les compositions ; une nouvelle variation de seuil seule est insuffisante. S708 officielle inchangée.

## Prochaine action

Concevoir S808 sur GuitarSet FIT80, avec des têtes **d'entrée de voix** et **de continuité de voix** différentes, sélectionnées selon le fragment/la hauteur et entraînées hors composition. Mesurer la réduction de chacun des quatre types de trames-hauteur manquées, ainsi que les nouvelles erreurs et les six métriques sur les événements MIDI256 *réels*. Conserver une ablation témoin par type et les 12 compteurs historiques ; rejeter toute régression globale ou par composition, toute confirmation après attaque et tout ajout de latence. Exiger une validation indépendante avant promotion.
