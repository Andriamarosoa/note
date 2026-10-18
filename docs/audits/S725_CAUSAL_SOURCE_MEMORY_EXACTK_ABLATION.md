## Résultat **mesuré** après S725 : modèle non promu

[Run S725 39122894425 (SUCCESS)](https://github.com/Andriamarosoa/note/actions/runs/39122894425) — 21 tests réussis, deux époques FIT pour chaque bras, CAL32/HELD32/DEV12. **Seul `S708_frozen` satisfait les six gardes de non-régression CAL.** Comparaison principale :

| CAL | S708 frozen | Sans mémoire 1 époque + K | Avec mémoire 1 époque + K | Avec mémoire 2 époques + K |
|---|---:|---:|---:|---:|
| Onset F1 | 74,2639% | 74,3603% | **74,3876%** | 74,0156% |
| Exact-K | 62,0455% | 62,0677% | **62,1445%** | 62,3116% |
| Exact-K actif | 56,2762% | 56,4329% | 56,5217% | **56,8449%** |
| Offset conditionnel | **77,3292%** | 76,6621% | 76,7063% | 76,7528% |
| F1 notes complètes | **57,1319%** | 56,6984% | 56,7517% | 56,4869% |
| F1 frame-pitch | 76,4626% | 76,6269% | 76,6705% | **76,7586%** |

La mémoire apporte, **à époque 1 contre son témoin entraîné à budget identique**, seulement +0,0273 point d'Onset F1, +0,0768 point d'Exact-K et +0,0532 point de full-note F1. Ce n'est pas un gain substantiel ni robuste à davantage d'entraînement (époque 2 se dégrade relativement au contrôle sans mémoire sur plusieurs métriques). Même les petites améliorations de K/frame-pitch restent accompagnées d'une dégradation Offset et full-note contre la référence S708.

HELD/DEV ont donc rejoué **S708 seul**, avec ses scores identiques à l'historique. **`promoted=false`**, LIVE S708/S707 inchangé. Aucune généralisation de la séparation E2/E3 n'est démontrée par les scores agrégés.

Diagnostic E2/E3 ciblé : [premier run CI](https://github.com/Andriamarosoa/note/actions/runs/39123312790) interrompu sur `KeyError: native_features` dans la préparation de l'entrée S707, **sans diagnostic complet**. Corrigé selon l'ABI de la validation globale, puis relancé : [run 39123937037](https://github.com/Andriamarosoa/note/actions/runs/39123937037). Aucun chiffre de détection E2+E3 ne doit être revendiqué avant la fin de ce run.

---

# S725 — Mémoire acoustique des sources et coopération Exact-K

## Hypothèse : pourquoi entend-on E3 quand E2 continue ?

Un E2 contient déjà une harmonique à la fréquence fondamentale d'E3.
Une représentation fréquentielle **statique** ne suffit donc pas à
attribuer cette énergie à un E2 seul ou à deux cordes distinctes.
L'hypothèse expérimentale est qu'une **nouvelle excitation E3** apporte
des changements de partiels et d'attaque indépendants de la
décroissance précédemment observée pour E2.

Cette mémoire n'a pas accès à la vérité JAMS, ni à des échantillons
futurs. Elle conserve seulement un état acoustique et une probabilité
de possession issue du *Sustain natif de l'instant précédent*.

## Implémentation

`src/causal_note/s725_source_memory.py`

Le S708 original garde ses **288 caractéristiques par hauteur** et
son checkpoint intact. S725 ajoute **32 caractéristiques causales** :

- 8D résidus d'énergie spectrale positive depuis une *prévision
  de l'enveloppe des huit harmoniques* ; prévision basée uniquement
  sur la valeur précédente et la tendance amortie ;
- 8D résidus positifs normalisés par le niveau de chaque partiel,
  pour mieux confronter attaques douces/fortes ;
- 8D résidus négatifs (partiels qui continuent à décroître) ;
- 3D de nouveauté pairs – impairs dans les partiels du **parent octave**,
  pondérées par la *possession acoustique de ce parent* ;
- 3D de nouveauté multiple de trois – autre partiel pour
  le **parent situé 19 demi-tons plus bas** ;
- 2D indiquant la possession acoustique des deux parents.

La possession est une **EMA causale du posterior Sustain natif**
observé *auparavant*. Elle n'est ni l'Exact-K de vérité ni un
oracle de notes actives, et reste probabiliste.

La prévision de décroissance `last + clip(trend,-.35,0)`
ne suppose pas que la note doit continuer à monter après une attaque.
Les résidus calculés par différence expliquent les modifications
inattendues. Pas d'autogain des queues de notes. L'algorithme
n'émet **aucune nouvelle note de lui-même** : le réseau d'origine
avec Onset/Sustain/Release et **Exact-K K0–K6** doit interpréter
ses nouvelles caractéristiques, puis le sélecteur causal K guidé
détermine les NoteOn/NoteOff.

## Comparaison équitable (pas seulement nouveau modèle vs vieille référence)

[CI S725 — entraînement réel](https://github.com/Andriamarosoa/note/actions/runs/39122894425)

Sur exactement les mêmes **80 prises FIT**, entraînement
des réseaux 320D initialisés à partir du checkpoint signé S708 :

1. `no_memory_epoch1` et `no_memory_epoch2` : nouvelle architecture
   320D, mais les **32 entrées mémoire remplacées par leur moyenne FIT**
   : leurs contributions standardisées sont constantes.
2. `with_memory_epoch1` et `with_memory_epoch2` :
   mêmes poids initiaux, seed, AdamW, epochs, minibatches et même
   loss Onset+Sustain+Release+Exact-K, mais 32 entrées mémoire causales
   actives.

L'architecture 320D est identique dans les contrôles et les
candidats. La tête Exact-K reste entraînée dans **tous** les réseaux.
Pour chacun, comparer la politique originale `Kguided` et
`KweakOFF` : cette dernière désactive seulement le *rattrapage K
des attaques faibles*, **pas la tête K ni son conditionnement des
prédictions de hauteur**.

Le checkpoint `S708_frozen` reste un témoin indépendant exact.
Sélection uniquement sur **32 CAL** avec les **six métriques
individuellement non régressives**, sans perdre de corrects offsets
ou longs Onsets déjà acquis. Le modèle gagnant gelé est rejoué
sur 32 HELD et 12 DEV, **historiquement exposés**, jamais un
nouveau TEST ouvert. **Aucune promotion** sans revue et
aucun changement au LIVE.

## Tests techniques

- Séquence synthétique E2 seule + hausse uniforme de ses partiels :
  pas de fausse « propriété E3 » de parité octave.
- Séquence synthétique E2 qui résonne, puis nouvelles croissances
  de ses partiels pairs : résidu octave non nul si E2 précédemment
  reconnu en Sustain (indice d'une possible E3).
- Sans possession antérieure du parent, cet indice est nul :
  impossible de lui attribuer rétroactivement une note.
- Lecture complète vs `step()` streaming bit à bit identiques ;
  modification des frames futures sans conséquence sur les anciennes.
- À epoch0, 320D avec nouvelles colonnes initialement nulles :
  parité S708 des logits 3 têtes et logits Exact-K.
- Contrôle sans mémoire : 32 dimensions ajoutées constantes et
  les 288 dimensions S708 inchangées.

**Limite scientifique importante** : ces tests synthétiques
valident l'implémentation mais ne prouvent pas que E2+E3 soit
toujours séparable. Les instruments réels ont du bruit,
de l'inharmonicité et des variations de pincement/tonalité.
Le verdict doit venir des mesures GuitarSet et de nouveaux
enregistrements physiques de chevauchements.

## Résultats

Le seul verdict exploitable sera celui du `report.json` et des
logs de [S725](https://github.com/Andriamarosoa/note/actions/runs/39122894425).
Un run de code réussi n'est pas une amélioration prouvée ;
`promoted=false` par défaut.
