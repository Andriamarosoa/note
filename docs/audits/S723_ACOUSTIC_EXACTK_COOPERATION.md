# S723 — Coopération des preuves acoustiques et d'Exact-K (nouvelle ABI 320D)

**Statut** : expérience autonome sur branche Draft, aucun changement à la référence Live. L'apprentissage et les performances sont évalués dans [le workflow GitHub S723](https://github.com/Andriamarosoa/note/actions/runs/39117029844). Tant que le rapport final n'existe pas, ne pas annoncer de gain.

## Motivation mesurée
- S718 DEV : 525 attaques manquées, dont 427 sans preuve neuronale conjointe exploitable dans ±80ms et 446 attaques manquées sur les pistes d'accords.
- S720 : renforcer les positifs polyphoniques augmente les vrais Onset (+159) mais aussi les faux Onset (+472) sur CAL. L'Exact-K et le F1 des hauteurs progressent, mais Onset/Offset régressent.
- S721 : même une fusion délicate n'a pas passé toutes les protections individuelles. La référence S708 reste active.

## Ce qui reste de NOTE
- **49 hauteurs MIDI40–88, trois têtes Onset/Sustain/Release plus 7 logits Exact-K K0–K6**;
- modèle latent temporel causal, attention harmonique inter-hauteurs ±12/±19 et tête K globale;
- tête K softmax reste injectée dans la prédiction per-pitch : elle contribue au choix des notes; elle n'est pas jetée;
- `GuidedSelector` avec sélection causale K des attaques faibles et limite physique à six notes actives;
- règle Offset S707 figée `gate=.8,persistence=1,grace=24` pour une comparaison à un unique flux NoteOn/NoteOff.

## Nouvelles caractéristiques acoustiques : append-only
Module `src/causal_note/s723_polyphonic_novelty.py`.

L'ABI **ne modifie aucun des 288 champs S708**. Elle ajoute 32 valeurs par pitch :
- 24 variations positives d'amplitude log des 8 partiels × 3 résolutions depuis le dernier bloc causal;
- 2 différences de montée `f_rise` corrigées aux frontières du registre quand les notes inférieures octave/19 demi-tons n'existent pas;
- 2 masques de validité des voisins inférieurs, pour éviter la fausse comparaison avec E2;
- 4 scores de synchronie nouvelle attaque : croissance fondamentale, partiels 2 et 3 et minimum commun.

**Important** : ce sont des descriptions d'entrées; elles ne prouvent pas une distinction parfaite entre deux notes superposées. En particulier, certains partiels de E2 et E3 coïncident physiquement, et la confirmation devra venir des ablations et erreurs par registre.

Le modèle 320D est obtenu par **copie exacte des poids** du S708 288D dans la première projection, avec **32 colonnes initialement nulles**. Il reproduit le réseau S708 au départ (test de parité) ; l'information supplémentaire ne peut intervenir qu'à la suite d'un vrai entraînement sur FIT. La normalisation des 32 ajouts est calculée **uniquement sur les 80 sources FIT**.

## Ablation contrôlée Exact-K
Deux checkpoints 320D, après 1 et 2 époques FIT seulement. Candidatures CAL :
1. **S708 original non modifié**, politique K d'origine;
2. S723 époque 1 et 2 avec guidage Exact-K des faibles Onset **activé**;
3. Les mêmes modèles avec le **rattrapage guidé par K désactivé** (`guide_confidence=1.1`). La tête K neuronale reste calculée et injectée dans la tête de hauteur même dans cette ablation — ce test ne mesure donc que la contribution du guidage K côté sélection, et non la suppression totale d'Exact-K.

La sélection est **uniquement sur CAL (32 enregistrements)**; HELD (32) et DEV (12) seulement après gel. HELD/DEV sont historiquement exposés : aucune prétention de validation sur jeu externe inédit.

**Protection stricte** : aucune des six métriques (Onset, Exact-K, Exact-K actif, Offset, F1 note complète, F1 hauteur par trame) ne doit être inférieure à S708; aucun NoteOff antérieurement correct, aucune attaque longue reconnue ne doit être perdue, aucun TP global en recul. Pas de promotion sans revue indépendante.

## Preuves techniques
- `test/test_s723_polyphonic_novelty.py` : causalité (changer les données futures ne change jamais les caractéristiques précédentes), réponse de partiel spécifique à une hauteur, masquage du bas du spectre, copie réseau S708→320D équivalente aux logits originaux avant entraînement.
- `scripts/s723_acoustic_exactk_cooperation.py` : prépare la transformation, entraîne sur FIT, reconstruit les émissions S707 depuis le signal et archive les métriques.
- `.github/workflows/note-s723-audio-exactk-cooperation.yml` : tests, chargement checkpoints signés, entraînement et gel CAL, diagnostic HELD/DEV, artefacts de preuve.
- Référence S708/S707 LIVE inchangée.
