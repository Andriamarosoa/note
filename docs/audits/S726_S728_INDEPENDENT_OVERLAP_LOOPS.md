# S726–S728 — Boucles empiriques sur la séparation harmonique des notes

## Boucle 1 : S725 diagnostiqué sur le cas E2→E3 réel

[Run ciblé S725 — SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/39123937037) après correction de l'entrée `native_features` attendue par S707. **CAL32 contient 1 seule attaque annotée E3 alors que E2 résonne encore depuis au moins 60 ms** : S708, S725 sans mémoire et S725 avec mémoire trouvent **0/1** vrai E3 et émettent **2** fausses attaques E3 pendant qu'E2 résonne (mêmes trois modèles). Conclure que cette hypothèse marche sur E2/E3 serait donc faux.

## Boucle 2 : S726 — inventorier tous les vrais accords avec attaques successives

[Workflow S726 — SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/39124472618) ; mêmes compositions GuitarSet fixées, pas de TEST.

| Vrais Onsets indépendants pendant un grave déjà actif | FIT80 | CAL32 | HELD32 | DEV12 |
|---|---:|---:|---:|---:|
| E2→E3 strict | 50 | **1** | 21 | 0 |
| Intervalle 12 demi-tons (octave) | **791** | **151** | **467** | 23 |
| Intervalle 19 demi-tons (approx. 3e harmonique) | 286 | 67 | 119 | 38 |
| Toutes nouvelles attaques avec un grave déjà actif | 3210 | 875 | 1491 | 277 |

Les chevauchements sont définis par les *onsets distincts* avec le parent plus bas déjà annoté Sustain depuis ≥60 ms et encore actif au nouvel onset. Une seule fréquence élevée persistante n'est pas comptée comme nouvelle note. **Les 151 vrais cas CAL sont assez nombreux pour un diagnostic significativement plus riche que l'unique E2→E3**, sans créer de fausses prises synthétiques. Mais ils ne sont pas un nouveau test statistiquement indépendant.

## Boucle 3 : S727 — corriger la différence entre une harmonique E2 et une attaque E3

`src/causal_note/s727_harmonic_competition_loss.py` : perte supervisée **FIT uniquement**, avec 3 niveaux prédéclarés de pénalité de faux Onset `strength=1.0,1.8,2.6`, entraînés chacun pour 1 et 2 époques. La vraie attaque E3 si E2 est déjà active **reste un positif** (avec un léger facteur supplémentaire, 10% de la majoration); la fausse hypothèse E3 lorsqu'E2 est Sustain **devient un négatif plus coûteux**, idem 19 demi-tons. Les pertes Sustain, Release, count/Exact-K et la cohérence sont reprises du S706 d'origine. Le même checkpoint S708, audio/mémoire causale 320D, seed, données FIT, lots et paramètres sont utilisés pour toutes les branches. Le contrôle à `strength=1.0` reproduit exactement la fonction de perte historique S706.

[CI S727](https://github.com/Andriamarosoa/note/actions/runs/39124756179) mesure sur les 32 CAL officiels : Onset, K global, K actif, Offset, F1 notes complètes et F1 hauteur trame d'un seul flux NoteOn/NoteOff par candidat, avec même moteur de release S707. Aucun modèle promu si une des six métriques régresse ou si une ancienne attaque longue/fin de note correcte disparaît. On ne remplace jamais S708 LIVE sans validation.

## Boucle 4 : S728 — compter réellement les notes récupérées vs inventées

[CI S728](https://github.com/Andriamarosoa/note/actions/runs/39124936595) dépend du succès S727. Pour S708 et cinq checkpoints S727, on décode d'abord **sans vérité terrain** tous les streams sur les 32 CAL ; ensuite on ouvre les annotations pour mesurer, avec association unique pitch-identique et tolérance ±80 ms :
- nouvelles attaques d'octave correctement trouvées / manquées parmi les 151 ;
- fausses notes supérieures alors que la fondamentale inférieure est véritablement encore active ;
- la même mesure pour les 67 attaques à +19 demi-tons ;
- diagnostic par registre et latence.

Ces analyses CAL sont descriptives **et ne peuvent pas servir de nouvel oracle pour modifier les seuils CAL à l'infini**. HELD32 et DEV12 historiques restent des vérifications après le choix CAL et, pour une véritable promotion, il faudra une source inconnue supplémentaire. Exact-K continue de conditionner les hauteurs et de guider les attaques faibles, et la causalité LIVE n'est pas sacrifiée.

## Résultats

Les seuls gains acceptables proviennent des rapports signés après les runs S727/S728. **Aucun gain S727 n'est présumé ou garanti à l'avance.** L'échec à dépasser 80% ne doit pas être masqué par un changement de définition de F1 ou un mélange de corpus différents.
