## Résultat final S730 — 15 variantes rejetées

[Run 39126100578 : SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/39126100578). Les 15 variantes de 5 spécialistes S727 × 3 confiances Exact-K (0,35, 0,50, 0,65) ont été évaluées sur CAL32 avec le NoteOff S707 figé. **Seul S708 original** satisfait les six scores et les gardes sur les événements corrects. HELD32/DEV12 reproduisent donc S708 uniquement, aucune promotion LIVE.

| CAL32 | S708 | S730, pénalité 2,6 époque 1 / K 0,65 |
|---|---:|---:|
| Onset F1 | **74,2639 %** | 73,7026 % |
| Exact-K | 62,0455 % | **62,3741 %** |
| Exact-K actif | 56,2762 % | **56,7591 %** |
| Offset | **77,3292 %** | 75,5032 % |
| F1 notes complètes | **57,1319 %** | 55,3458 % |
| F1 hauteurs/trame | 76,4626 % | **76,6925 %** |

La variante K 0,50 basée sur pénalité 1,8 époque 2 atteint 62,9798 % Exact-K et 77,0390 % frame-pitch F1, mais Onset diminue à 73,2294 %, Offset à 73,8031 % et full-note à 53,7375 %.

**Le fait de préserver tous les événements originaux comme sous-séquence ne préserve pas les appariements de vérité terrain** lorsqu'un autre événement est ajouté. Les scores S730 le démontrent. Aucun gain complet ni objectif 80 % validé.

---

# S730 — Rattrapage causal de notes indépendantes, sans remplacer les notes reconnues par S708

## Pourquoi une cinquième boucle ?

S727 montre sur CAL32 que changer l'apprentissage pour supprimer les fausses harmonies crée un compromis. L'analyse S728 des **151 attaques d'octave réelles** donne :

| Candidat | Octaves correctement retrouvées | Fausses notes enfant quand le parent résonne |
|---|---:|---:|
| S708 | 99 / 151 | 43 |
| Réentraînement témoin 1 époque | 102 / 151 | 55 |
| Pénalité fausses harmonies ×2,6, 1 époque | 95 / 151 | 37 |

Il manque une méthode préservant les événements d'origine et exploitant les **nouvelles** preuves de hauteur provenant des spécialistes.

## Approche S730

**Pas de remplacement de la prédiction S708 ni du moteur S707** :
on construit indépendamment le flux de NoteOn/NoteOff de S708
et celui de chaque spécialiste S727, à partir du même audio PCM
(strictement causal et sans labels JAMS).

Le nouveau rattrapage reçoit **à l'instant courant** seulement :

- les événements originaux S708 réellement émis jusque-là ;
- les événements spécialisés S727 réellement émis jusque-là ;
- la dernière distribution **Exact-K K0..K6** issue du spécialiste
  et disponible **avant ou au moment** de l'événement candidat ;
- les notes actuellement actives dans les deux flux.

Une note supplémentaire est admise uniquement si elle n'est
pas déjà active/émise récemment, et si le spécialiste estime
une cardinalité **strictement supérieure au nombre de notes
déjà actives**, avec confiance K au-dessus du seuil prédéclaré.
K n'invente jamais une note : il doit y avoir **un véritable
NoteOn émis par le spécialiste**.

Si une note original S708 du même pitch est émise alors qu'une
note supplémentaire était déjà active, la note supplémentaire
reçoit immédiatement un NoteOff causal au même instant puis
la note S708 passe **intacte**. Les autres événements S708 ne sont
ni supprimés ni modifiés. Chaque NoteOff additionnel provient
du même S707 causal ou de cette préemption instantanée :
**aucune décision rétrospective**.

### Contrats tests

`test/test_s730_additive_k_guard.py` prouve notamment que :

- sans proposition, le résultat est exactement S708 ;
- K=1 garde E2 seule et rejette E3 fantôme ;
- K=2 peut accepter une nouvelle E3 acoustiquement proposée
  pendant que E2 sonne ;
- une K future ne peut pas valider une attaque passée ;
- une NoteOn S708 originale préempte la note ajoutée de
  même pitch sans jamais supprimer son propre événement.

### Boucles CAL figées

[Run S730](https://github.com/Andriamarosoa/note/actions/runs/39126100578).

Les **15 variantes** sont la combinaison de 5 vrais checkpoints
S727 × 3 seuils de confiance Exact-K (0,35, 0,50, 0,65),
comparées au S708 figé sur les mêmes 32 enregistrements CAL.

Critères immuables :
- aucun recul des six métriques Onset, K total/actif,
  Offset, F1 notes complètes, F1 hauteurs par trame ;
- zéro ancienne attaque longue correcte ou NoteOff
  correct perdu ; Onset TP non inférieur à S708 ;
- modèle gagnant gelé avant relecture de HELD32 et DEV12,
  déjà exposés historiquement ; TEST reste fermé ;
- **`promoted=false`** tant qu'un audit indépendant et
  l'intégration microphone LIVE n'ont pas validé l'ensemble.

Important : préserver textuellement les événements S708 ne garantit
pas à lui seul des scores Offset et full-note non-régressifs,
car une note ajoutée peut modifier l'association des événements.
Les métriques seront recalculées à partir du **seul flux fusionné**
et le candidat sera rejeté si un garde-fou échoue.

## Résultats
Pas de mesure validée tant que le run n'a pas émis
`S730 CAL FROZEN` et `S730 FINAL`. Aucune amélioration
imaginaire attribuée à S730.
