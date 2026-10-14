# S525–S529 — Pourquoi les nouvelles notes échouent, et récupération protégée K0 (15 octobre 2026)

**Verdict : cinq nouveaux workflows vérifiés.** Aucun nouveau modèle n'est une référence officielle ; ni S80, ni Exact-K historique, ni `freeze_local_combo`, ni Live, ni onset/offset existants n'ont été modifiés.

## Motif du travail

S519/S520 puis S521–S524 ont établi que le sélecteur à 14 têtes sur les seuls événements canoniques Basic Pitch supprimait plus de notes qu'il n'en découvrait. En S523 (têtes ADD et KEEP/REMOVE), 1 seule nouvelle vraie hauteur et 10 fausses additions piano ; S524 a validé le rejeu des 24 poids entraînés. La cause peut être **en amont** du sélecteur, puisqu'un générateur incapable d'émettre un événement ou une nouvelle fondamentale impose une borne structurelle au réseau.

Sources : [S519](https://github.com/Andriamarosoa/note/actions/runs/38643896675), [S524](https://github.com/Andriamarosoa/note/actions/runs/38645298381).

## S525 — le générateur physique alternatif S511 ne corrige pas l'absence d'événement

[CI S525 SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/38646669720). Même corpus historique, appariement temporel un-à-un ±80 ms, événements du générateur S513 audio-only gelés avant la lecture des MIDI/XML, sans réentraînement.

| | IDMT 5 WAV guitare | MAESTRO 3 WAV piano |
|---|---:|---:|
| Vraies attaques annotées évaluées | 45 | 322 |
| Appariées S498 canonique BP | 45 | 220 |
| Appariées à une attaque physique S511 | 42 | 19 |
| BP manquée mais S511 la récupère | 0 | **0 / 102** |
| Accord complet contenu dans les 12 hauteurs candidates S511, oracle | 8 | 15 |

S511 est donc **un mauvais remplaçant de la génération d'attaques piano**. Ce score top12 est couverture oracle, non inférence note. Les événements non appariés ne sont pas des négatifs certifiés.

## S526 — 102 attaques MIDI existaient déjà parmi les candidats temporels V88 avant filtrage BP

[CI S526 SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/38646978013). Sur les 322 vraies attaques MAESTRO S476 **déjà appariées au V88 originel**, 220 figurent dans le graphe canonique Basic Pitch S498, et **102 en sont absentes**. Cela représente **116 hauteurs frappées** (88 attaques monophoniques et 14 polyphoniques). Parmi les 102, le V88 originel prédisait :
- K0 : **89** ;
- K1 : **9** ;
- K2 : **4**.

Même lorsqu'un candidat V88 a K0, **son horodatage d'attaque peut être pertinent pour faire naître une note**. La perte ne doit pas être confondue avec une absence d'événement V88. Parmi les 220 attaques retenues par S498, 64 manquent au moins une nouvelle vraie hauteur de note.

Prudence : **les 322 attaques avaient été sélectionnées parce qu'elles correspondaient déjà à un candidat V88**. La cohorte d'attaques MIDI originale comporte 330 événements (8 non appariés à V88). Ne pas présenter le 322/322 comme un score indépendant de rappel V88.

## S527 — vraie banque de preuves WAV pour tous les événements V88, y compris K0

[CI S527 SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/38647316407). Relecture des vrais WAV MAESTRO, empreintes SHA vérifiées contre le gel S470, récupération Basic Pitch S485 sans labels MIDI. Pour chacun des **575 événements V88 d'origine**, calcul de **49 hauteurs MIDI40–88 × 11 caractéristiques** S515 : fondamentales spectrales, montée, sustain, nouveauté, présence BP, support BP, soupçon harmonique et flux d'attaque. L'algorithme accepte un événement même sans note BP et avec source V88 K0. Cela inclut les **473 événements prédits K0**.

| Enregistrement | Événements observés | Dont K0 |
|---|---:|---:|
| A | 211 | 177 |
| B | 210 | 184 |
| C | 154 | 112 |
| **Total** | **575** | **473** |

Il s'agit **d'entrées acoustiques non supervisées / préparées pour l'apprentissage**, pas d'une sortie de détecteur validée.

## S528 — six modèles réellement entraînés hors enregistrement tenu à part

[CI S528 SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/38647706630). Trois folds par enregistrement tenu à part, apprentissage ExtraTrees uniquement sur les **deux autres enregistrements**, positives = hauteurs MIDI frappées sur événements d'attaque déjà appariés (322 au total), négatives = hauteurs individuelles absentes de ces **attaques positives**. Les 253 événements V88 restants sans annotation associée **ne sont jamais déclarés faux onsets**. Les 8 attaques MIDI sans V88 restent manquées.

| Variante 49 sorties indépendantes | F1 pitch conditionnel sur les 322 attaques positives | Accords complets /322 |
|---|---:|---:|
| 7 indices WAV seuls, seuil 0,50 | 39,96 % | 42 |
| WAV seuls, seuil 0,70 | 39,83 % | 50 |
| 11 indices WAV + BP, seuil 0,50 | 44,64 % | 47 |
| WAV + BP, seuil 0,70 | **50,14 %** | **82** |

WAV seul p>=0,50 retrouve **13/102** accords complets sur les onsets historiquement supprimés par S498, au prix de **30 régressions** sur les attaques autrefois retenues par S498 : ce n'est **pas** une nouvelle référence.

Les résultats sont des **F1 de hauteur conditionnels sur un jeu d'attaques réelles positives**, ni F1 NOTE_ON de flux entier, ni généralisation aveugle aux pièces inédites. Les trois morceaux ont déjà été exposés au développement historique.

## S529 — politique d'ajout sans suppression, audit indépendant, aucune régression *sur les accords déjà exacts*

[CI S529 SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/38648092726). Sans nouvel apprentissage, utiliser exactement les probabilités S528 gelées. Garder chaque note **née** dans le BP préexistant lorsque le groupe possède au moins une note BP, et autoriser les nouvelles hauteurs uniquement pour les événements où la naissance BP est vide. Aucune étiquette MIDI n'entre dans la décision d'inférence. Les seuils 0,50/0,70 sont ceux déjà définis, pas des seuils optimisés sur cette validation ; **la politique d'ajout est néanmoins conçue après l'inspection du corpus historique**, ce qui exclut une revendication de validation prospective.

| Variante | Exact accords /322 vs BP naissance 37 | Corrections complètes | Régressions complètes | Accords récupérés parmi les 102 | Naissances proposées sur 253 événements sans étiquette de vérité |
|---|---:|---:|---:|---:|---:|
| WAV seul p>=0,50 | **68** | +31 | **0** | 13 | **170 inconnues** |
| WAV seul p>=0,70 | **75** | +38 | **0** | 10 | **71 inconnues** |
| WAV+BP p>=0,50 | **66** | +29 | **0** | 5 | **170 inconnues** |
| WAV+BP p>=0,70 | **86** | +49 | **0** | 5 | **93 inconnues** |

Un résultat **0 régression** est surtout une **garantie structurelle de non-remplacement de notes BP déjà présentes** au sein de cette évaluation, pas la preuve que la politique est sans risque sur un flux réel. Les 71–170 naissances sur candidats sans correspondance parmi les 253 sont des sorties de modèle dont la vérité est INCONNUE, et il serait trompeur de les annoncer fausses ou vraies.

À noter : la référence de comparaison S529 (**37/322**) est la **naissance des noms de notes Basic Pitch aux timestamps V88**. Elle est différente des **125/220 accords** de l'évaluation Basic Pitch canonique à la fenêtre +30ms de S523, du V88 K historique, du score S80 et de YourMT3+ ; ne pas comparer les pourcentages directement.

## Architecture recommandée pour la suite (sans promotion)

```
WAV réel
   |-- têtes candidates temporelles V88 (TOUS K0..K6, état K seulement prior souple)
   |-- tête note acoustique 49 hauteurs / fondamentale vs harmonique
   |-- têtes frame / onset / sustain / NEW_NOTE vs CONTINUATION
   |-- indice Basic Pitch (possible mais pas obligatoire)
   v
Sélecteur conditionnel de note à nouvel onset
   |-- KEEP des notes déjà démontrées
   |-- ADD des nouvelles hauteurs avec confiance + contrôle harmonique
   |-- protection poly et audit par K véritable
   v
NOTE_ON hauteur E3/G3/B3 + horodatage (décodeur Live inchangé)
```

Priorité : collecter **vrais négatifs d'attaque** vérifiés (les 253 événements non appariés ne suffisent pas en tant que vérité négative), multiplier les morceaux / compositeurs / instruments et geler l'entraînement, les seuils et le protocole **avant** une évaluation réellement neuve. Ensuite seulement mesurer précision/rappel/F1 des NOTE_ON dans le flux continu, Exact-K, attaques manquées, accords complets et latence Live. **Cible 80 % F1 événementiel toujours non atteinte/non démontrée.**

## Conventions et sauvegardes

- Aucun checkpoint V88/S80, `freeze_local_combo`, Live ou décodeur onset/offset n'est écrasé.
- Les scripts de cette branche sont de nouveaux modules de développement, pas de la référence.
- SHA des WAV, modèles source gelés, sorties et labels MIDI vérifiés dans les workflows.
- Tous les gains sont limités aux corpus historiques et ne peuvent déclencher de promotion.
