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

# S794 — Résultats Active Exact-K causal MIDI 256

## Exécution

**TERMINÉ — SUCCESS :** [run S794 #39336923230](https://github.com/Andriamarosoa/note/actions/runs/39336923230), commit scientifique `645639f0180f39b17069d55861189943f98adb4a`. 80/80 enregistrements FIT et 32/32 CAL traités ; **6 audits indépendants PASS et consensus PASS**. Archive `S794_REAL_MIDI_ACTIVEK_ROOTCAUSE_FIT80_CAL32`.

La reconstruction utilise le chemin acoustique et le décodeur originaux S708, sans modifier aucun événement MIDI. SHA256 du modèle S708 signé : `c81d98dbce8d096b54d71dedd88f521162ea95b19d0c059ab13baf69d5cf0588`. **Parité stricte CAL réussie pour les six scores et les douze compteurs demandés**, notamment `k_correct=78366`, `active_k_correct=56413`. HELD/TEST fermés, **zéro ajout de buffering/lookahead/confirmation/hold** dans cette expérience analytique. Aucune mesure physique microphone nouvelle.

## Métriques

Les six scores officiels CAL32, calculés sur les événements MIDI 256, restent **exactement S708** : S794 est un audit de causes d'échecs, pas un modèle nouvellement entraîné.

| Mesure officielle GuitarSet CAL32 | S708 signée | Rejeu S794 | Variation |
| --- | ---: | ---: | ---: |
| Onset F1 | 74,2639 % | 74,2639 % | 0 |
| Exact-K | 62,0455 % | 62,0455 % | 0 |
| Active Exact-K | 56,2762 % | 56,2762 % | 0 |
| Offset accuracy | 77,3292 % | 77,3292 % | 0 |
| Full Note F1 | 57,1319 % | 57,1319 % | 0 |
| Frame Pitch F1 | 76,4626 % | 76,4626 % | 0 |

### Causes de cardinalité au niveau des événements réellement émis

| K réel (K6 regroupe 6+) | Trames CAL32 | Exactes | Sous-comptées | Surcomptées | Exactitude K |
| --- | ---: | ---: | ---: | ---: | ---: |
| K0 | 26 061 | 21 953 | 0 | 4 108 | 84,24 % |
| K1 | 52 950 | 41 378 | 8 450 | 3 122 | 78,15 % |
| K2 | 14 903 | 5 318 | 8 165 | 1 420 | 35,68 % |
| K3 | 11 880 | 3 909 | 6 380 | 1 591 | 32,90 % |
| K4 | 16 773 | 5 480 | 10 572 | 721 | 32,67 % |
| K5 | 2 971 | 279 | 2 606 | 86 | 9,39 % |
| K6+ | 766 | 49 | 717 | 0 | 6,40 % |

Sur les **100 243 trames actives**, 56 413 sont correctes, **36 890 sous-comptées** et **6 940 surcomptées**. La majorité des erreurs vient des voix manquantes, surtout K2–K6.

### Registres / indices acoustiques causaux

| Registre | Trames-pitch de notes réelles manquées | Trames-pitch parasites |
| --- | ---: | ---: |
| E2–B2 | 10 747 | 2 636 |
| C3–B3 | 28 649 | 9 591 |
| C4–E6 (49 hauteurs MIDI 40–88) | 24 625 | 10 277 |
| **Total** | **64 021** | **22 504** |

**6 843 sur 36 890** trames sous-comptées présentent au moins une hauteur manquante avec la preuve acoustique restrictive (`sustain ≥ 0.30`, `onset ≥ 0.10`, `release < 0.78`) déjà disponible au moment de la décision ; **30 047** n'en présentent pas selon ce seul critère. Il ne s'agit **ni de notes récupérées, ni d'un gain F1**, et les seuils d'audit ne représentent pas nécessairement toutes les formes d'indices utiles.

| Signal spécialisé candidat | Trames sous-comptées avec indice pour hauteur absente | Fausses hauteurs également couvertes |
| --- | ---: | ---: |
| Voix grave | 995 | 1 631 |
| Nouvelle voix en polyphonie | 4 741 | 10 153 |
| Transition Sustain | 9 385 | 12 561 |
| Collision harmonique | 894 | 1 823 |

Les familles se recoupent et les chiffres de transition emploient un seuil différent du test restrictif ci-dessus. Ils représentent une **couverture diagnostique**, pas quatre têtes entraînées ni quatre corrections exploitables.

## Comparaison à S708

**Parité stricte CAL32 atteinte après reprise du chemin historique S708.** Six métriques et compteurs identiques. Aucun nouvel entraînement, aucun retard ajouté aux événements et aucune promotion. L'objectif >80 % Active Exact-K reste ouvert.

## Audits

| Rôle | Statut | Preuve |
| --- | --- | --- |
| Données | PASS | Run scientifique #39336923230, `S794_AUDIT_data` |
| Acoustique | PASS | Run scientifique #39336923230, `S794_AUDIT_acoustics` |
| Architecture | PASS | Run scientifique #39336923230, `S794_AUDIT_architecture` |
| Apprentissage | PASS | Run scientifique #39336923230, `S794_AUDIT_training` |
| Métriques | PASS | Run scientifique #39336923230, `S794_AUDIT_metrics` |
| LIVE | PASS | Run scientifique #39336923230, `S794_AUDIT_live` |
| **Consensus** | **PASS** | `S794_SIX_ZERO_DELAY_ACTIVEK_AUDITS` |

## Incident de reproductibilité — Run #39333999180

Le calcul scientifique a traité **80/80 FIT et 32/32 CAL**, mais le garde-fou de signature S708 a refusé la comparaison. GitHub [run #39333999180](https://github.com/Andriamarosoa/note/actions/runs/39333999180) : `failure`, causée précisément par `k_correct=78367` contre `78366` signé. Les autres petites différences comptables sont `active_k_correct: 56414 vs 56413`, `frame_pitch_tp: 140540 vs 140541`, `frame_pitch_fp: 22501 vs 22504`, `frame_pitch_fn: 64022 vs 64021`. Aucun changement au modèle LIVE n'est à l'origine de cet échec, mais la **parité stricte n'est pas établie**. Onset et Offset sont inchangés. Ces données demeurent diagnostiques et ne peuvent autoriser aucune promotion.

L'audit a néanmoins calculé sur CAL32 les erreurs suivantes, sur les événements émis :

| Nombre réel de voix | Trames | K correct | Sous-comptages | Surcomptages |
| --- | ---: | ---: | ---: | ---: |
| K1 | 52 950 | 41 379 | 8 450 | 3 121 |
| K2 | 14 903 | 5 318 | 8 165 | 1 420 |
| K3 | 11 880 | 3 909 | 6 380 | 1 591 |
| K4 | 16 773 | 5 480 | 10 572 | 721 |
| K5 | 2 971 | 279 | 2 606 | 86 |
| K6+ | 766 | 49 | 717 | 0 |

Total trames actives sous-comptées : **36 890** ; surcomptées : **6 939**. Les fréquences élevées de sous-comptage ne constituent pas de nouveaux scores de modèle : elles motivent la recherche de spécialistes de voix manquantes.

Une deuxième exécution de reproductibilité (CAL32 deux fois avec hachage de chaque flux MIDI) est lancée dans [run #39335625292](https://github.com/Andriamarosoa/note/actions/runs/39335625292). Ne pas masquer l'écart par une tolérance sur la parité.

## Investigation indépendante de la reproductibilité

- [Rejeu CAL32 deux fois #39335625292](https://github.com/Andriamarosoa/note/actions/runs/39335625292) : **SUCCESS**. Les **32/32 fichiers donnent des événements MIDI strictement identiques d'un rejeu à l'autre** (`0` enregistrement divergent). Les deux répétitions présentent exactement le même écart de **+1 `k_correct`** avec le rapport signé de S708. Le problème n'est donc pas une variation aléatoire **entre ces deux répétitions**.
- [Comparaison ancienne voie S708 et wrapper moderne #39336494854](https://github.com/Andriamarosoa/note/actions/runs/39336494854) : **SUCCESS**. Les deux voies produisent les **mêmes événements sur 32/32 fichiers**. Toutes deux affichent les mêmes différences avec les compteurs signés, donc **le wrapper S733 n'est pas la cause**.
- Le score officiel S708 reste celui qui figure dans le rapport original. Les micro-différences entre le rapport historique et sa reconstruction exacte ne peuvent ni être masquées par une tolérance de parité, ni être présentées comme une amélioration de l'Active Exact-K.
- La divergence demeure **une question de reproductibilité inter-exécutions** (environnement numérique, calculs sensibles aux seuils ou autre facteur encore indéterminé). Aucun mécanisme précis n'a été démontré.
- La chaîne S794 a été replacée sur les fonctions S708 d'origine, strictement inchangées. **Le dernier run #39336923230 atteint bien la parité stricte** : cet incident historique est clôturé pour S794, même si sa cause élémentaire n'a pas été isolée par le seul test comparatif. Les futures promotions de modèles restent bloquées sans preuve S793 de latence matérielle et validation des six mesures.

## Verdict

**S794 TERMINÉ : SUCCESS, 6/6 audits PASS, consensus PASS, parité MIDI256 exacte.** Cette série a identifié les causes de faiblesse d'Active Exact-K mais ne constitue **pas une amélioration**. Aucune architecture multi-têtes entraînée ni modèle LIVE promu. Zéro ajout de temps d'attente logiciel ; pas de mesure physique de latence supplémentaire.

## Prochaine action

Concevoir sur FIT une banque de spécialistes **de nouvelles attaques réellement manquées** (voix polyphoniques, transitions, basses), séparés des mauvaises extinctions. Choisir le nombre de spécialistes selon l'apport marginal OOF par composition, puis évaluer directement les événements MIDI256 avec six métriques, parité S708 et **zéro latence ajoutée vérifiée par S793**. Ne pas choisir d'expert ni de seuil sur CAL.
