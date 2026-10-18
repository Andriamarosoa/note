# S720 — Premier vrai réentraînement des têtes Onset/Sustain polyphoniques

## Motif mesuré

[S718 (16 tests PASS)](https://github.com/Andriamarosoa/note/actions/runs/39111634032) : 525 Onset FN DEV exacts, dont
321 sans preuve neuronale conjointe et 106 avec score Onset constamment
inférieur à .15. Les accords représentent 446/525 FN. Le réarmement
conditionnel S719 a émis zéro nouvelle attaque et n'a amélioré
aucune métrique : [S719](https://github.com/Andriamarosoa/note/actions/runs/39111883787).

## Expérience S720

**Vrai entraînement neuronal**, pas une modification cosmétique
de seuils. On reprend le checkpoint `s708_selected.pt` figé, SHA
`c81d98dbce8d096b54d71dedd88f521162ea95b19d0c059ab13baf69d5cf0588`.
Les mêmes paramètres S708, ses 49×288 features causales natives+sharp,
S707 75D Offset, et le même décodeur sont utilisés.

Trois candidatures contrôlées à budget identique :

| Nom | Origine | Apprentissage supplémentaire |
|---|---|---|
| `S708_frozen` | Checkpoint original | Aucun |
| `plain_control_epoch1` | Même S708 initial | 1 époque BCE historique, AdamW LR 8e-5 |
| `poly_bass_epoch1` | Même S708 initial | 1 époque BCE; poids ×3 attaques vraies avec K>=2 ; facteur supplémentaire 1.5 sur notes 40..64; légère pondération Sustain (+20% de l'augmentation) **seulement pour les mêmes pitches attaqués** |

Les données négatives, les silences, les solos et toutes les 49 hauteurs
restent présents dans les fenêtres d'entraînement, sans
sur-échantillonnage caché. Les cibles `y,K` sont extraites des notes
**FIT seulement** (80 prises de 5 familles de compositions). Le réseau
nécessite uniquement le son causal lors de l'inférence.

Le contrôle non pondéré est un véritable réentraînement depuis les
mêmes poids et avec le même nombre de passes (et non une comparaison
biaisée avec S708 non réentraîné uniquement).

## Sélection et garanties

1. **FIT** : entraîner les deux modèles; sauvegarder poids et hashes.
2. **CAL** : comparer depuis le **même flux NoteOn/NoteOff réellement émis**
   avec politique S708 et release S707 figés. Les six scores
   (`onset_f1`, `exact_k`, `active_exact_k`, `offset_accuracy`,
   `full_note_f1`, `frame_pitch_f1`) doivent être individuellement
   au moins ceux de S708; de plus, zéro perte de vrais Onsets longs
   précédemment reconnus, zéro perte des corrects NoteOff antérieurs,
   et TP total >= S708.
3. **HELD/DEV** : diagnostic seulement **après gel** CAL, sans retoucher
   le modèle d'après leurs scores. Splits historiquement exposés,
   PAS un nouveau TEST prospectif.
4. `promoted=false` sans revue indépendante. Aucun changement au
   `live_strict_notes.py` ou aux références S80/S613/S675.

## Preuves et résultat

[Workflow GitHub S720](https://github.com/Andriamarosoa/note/actions/workflows/note-s720-poly-onset-fit-cal.yml).

Archive `S720_POLY_ONSET_FIT_CAUSAL_CAL_HELD_DEV_NO_PROMOTION` :
`report.json` (6 scores/tous les candidats sur CAL et checks HELD/DEV)
et checkpoints expérimentaux non promus.
La CI annonce séparément l'échec de l'entraînement, les résultats de
CAL et les validations, sans prétendre dépasser 80% si ce n'est pas
mesuré.

Un verdict négatif est valide : il indique qu'une repondération
des pertes ne suffit pas et qu'il faut étudier de **nouvelles preuves
acoustiques indépendantes** (fondamentales graves et harmoniques
superposées) plutôt que retoucher l'arbitrage de seuils.
