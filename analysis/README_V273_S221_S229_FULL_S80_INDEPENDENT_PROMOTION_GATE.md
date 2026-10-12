# S221–S229 — Chaîne originale S80 complète, mais validation externe encore absente

**Date :** 12 octobre 2026. **Recherche :** `codex/v273-scope-rescue`. **Référence officielle :** `freeze_local_combo`. **Promotion : NON.**

## Reconstitution réellement exécutée

| Étape | Run terminé | Source et preuve |
|---|---|---|
| S221 | [38295588099](https://github.com/Andriamarosoa/note/actions/runs/38295588099) | Deux véritables sélecteurs S69 à règles historiques pré-apprises (14 balanced, 22 polymax), 36 PR16 sources authentiques, déploiement S69 quatre contre-factuels sans oracle de fold, zéro intervention observée |
| S222 | [38296428889](https://github.com/Andriamarosoa/note/actions/runs/38296428889) | 4 modèles logistiques S70 d’origine, seuils d’apprentissage sauvegardés, 0 écart sur 59 309 décisions historiques pour chacune des 2 variantes S70 |
| S223 | [38296662215](https://github.com/Andriamarosoa/note/actions/runs/38296662215) | 4 modèles S70 originaux exécutés sur 57 événements audio GOAT, 7 éligibles et 0 intervention acceptée par 3 folds sur 4 |
| S224 | [38296880555](https://github.com/Andriamarosoa/note/actions/runs/38296880555) | 4 modèles S71 historiques entraînés d’origine, 119 caractéristiques, 6 familles de transitions ; 0 écart sur les variantes risk/poly archivées, 59 309 lignes |
| S225 | [38297080728](https://github.com/Andriamarosoa/note/actions/runs/38297080728) | 4 modèles S72 temporels historiques HistGradientBoosting, 465 caractéristiques et leurs seuils ; 0 écart sur les 4 variantes enregistrées (59 309 lignes) |
| S226 | [38297421600](https://github.com/Andriamarosoa/note/actions/runs/38297421600) | Exécution des 8 modèles S71/S72 sur de vrais signaux audio GOAT, géométrie PR16 originale 46D, séquences 42×49 ; aucune intervention parce que S70 n’en fournit pas |
| S227 | [38297654063](https://github.com/Andriamarosoa/note/actions/runs/38297654063) | Exécution des 4 modèles K1 S79 originaux sauvegardés + vraie formule finale historique S80 K0/K2 avec les 36 votes PR16 et 193 politiques S46 ; **57 prédictions finales S80** |
| S228 | [38297962398](https://github.com/Andriamarosoa/note/actions/runs/38297962398) | Deuxième implémentation NumPy indépendante : 6/6 contrôles réussis sur 57 prédictions finales ; même compte des actions, même alignement à la référence |

Les premières parties authentiques sont également disponibles : S5→S18, S29→S35, 16 sources S45, 304 risques Ridge originaux, 193 politiques S46, avec la règle de vote expérimentale S217–S220 [documentée ici](README_V273_S217_S220_S18_BRANCH_RESOLUTION.md).

**Attention à l’identité du système :** on a retrouvé des paramètres de modèles et reproduit numériquement leurs décisions historiques avec les véritables données d’origine, mais les règles de sélection d’un fold et des 19 modèles « pièce exclue » sur de nouveaux morceaux sont de nouveaux choix expérimentaux. Cette chaîne de déploiement n’est **pas** une preuve de stricte identité avec le S80 historique.

## Verdict sur GOAT (57 événements, déjà exposés)

- Histogramme S80 prédit K0→K6 : **45, 9, 3, 0, 0, 0, 0**.
- Actions propres à S79 K1 : **0** ; règle finale K0 : **0** ; règle finale K2 : **0**.
- Différences finales S80 vs son propre parent S72 : **0**.
- Différences de prédiction par rapport à `freeze_local_combo` : **4** événements non étiquetés : `example2_di.wav:136137`, `example2_di.wav:177822`, `example2_di.wav:202637`, `example2_di.wav:232745`.
- **Corrections prouvées : INCONNU. Régressions prouvées : INCONNU. Exact-K global et K2+ : NON MESURÉS.**
- Ces 57 événements GOAT ont été exposés auparavant. Ni leur précision ni une validation interne historique ne peuvent remplacer l’épreuve song-disjoint indépendante.

Ne pas confondre le succès du **rejeu technique indépendant S228** avec l’évaluation **statistique indépendante de la précision** : le premier a réussi, la seconde n’a pas été réalisée.

## Le véritable blocage restant

Le dépôt contient un corpus candidat IDMT-SMT-GUITAR V2 (Zenodo 7544110). [S188](../scripts/s188_idmt_dataset2_note_xml_index.py) a défini, **avant accès aux annotations**, un tri SHA256 déterministe de 16 WAV `dataset2` et repéré des couples WAV/XML.

Le modèle officiel `freeze_local_combo` dépend des **groupes d’événements natifs du V8.4/V8.6**. Les scripts historiques externes S82–S97 et les scripts récents S209–S228 sont largement liés à `example1_di.wav` et `example2_di.wav` ou à une matrice de 57 lignes. Enregistrer des attaques acoustiques provisoires sur IDMT n’est **PAS** suffisant pour calculer des métriques Exact-K comparables : il faut exécuter exactement le même générateur d’événements natifs pour les prédictions S80 et `freeze_local_combo`, avant de consulter le XML de notes.

Le nouveau [S229](../scripts/s229_idmt_dataset2_sealed_audio_only_S97.py) sert au préalable à sélectionner et sceller par SHA256 les enregistrements IDMT et à extraire les vues audio d’origine 58D/245D/3D/42×49/490D/294D sans ouvrir XML/MIDI. Il ne doit **jamais** être présenté comme un test Exact-K complet et ne doit pas influencer le choix des architectures après inspection des annotations.

## Conditions strictes de promotion

1. Exécuter les vrais événements natifs V8 et le baseline officiel sur le corpus IDMT réellement non exposé ; unifier son traitement WAV et la chaîne S5→S80 pour un nombre variable d’événements.
2. Geler tous les noms et hachages de fichiers WAV, identifiants d’événements, K prédit du baseline, K prédit de la nouvelle chaîne, règle de vote, seuils et poids **avant la moindre lecture XML/MIDI**.
3. Lire seulement après la preuve de gel le contenu des annotations, prouver qu’elles représentent l’Exact-K simultané aux instants choisis, et exclure les cas ambigus selon un protocole fixé à l’avance.
4. Mesurer le comparatif apparié **Exact-K global, K2+ polyphonique, K0, K1, K2, K3, K4, K5, K6**, corrections, régressions, analyse d’incertitude groupée par morceau et éventuelle fuite.
5. Ne promouvoir que si le nouveau déploiement versionné dépasse effectivement `freeze_local_combo` selon les critères expérimentaux définis. **Aucun transfert automatique de référence sans score indépendant.**

Conclusion : **l’exécution technique S80 de bout en bout est démontrée ; la promotion n’est pas démontrée.**
