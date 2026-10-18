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
