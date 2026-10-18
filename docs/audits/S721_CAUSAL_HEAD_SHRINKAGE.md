# S721 — Fusion par tête S708 / S720, calibrée sans fuite

## Motif

S720 a récupéré +159 vrais Onset CAL mais ajouté +472 faux Onset. Le
réentraînement pondéré améliore aussi Exact-K (+1,92 pt) et frame-pitch
F1 (+2,04 pt) mais dégrade Onset F1 (-1,78 pt), Offset (-2,66 pt)
et F1 notes complètes (-3,31 pt) par rapport au checkpoint S708 original.

## Expérience reproductible

- Checkpoint original S708 et checkpoint S720 `poly_bass_epoch1.pt`
  entraîné sur FIT, tous deux avec SHA256 vérifié par leurs rapports.
- Même famille de 49 pitches, 288 caractéristiques causales,
  3 probabilités Onset/Sustain/Release et 7 probabilités K par décision.
- **15 variantes explicites**, dont témoin `S708_frozen` :
  mélanges à poids 10% / 25% / 50% appliqués séparément à
  Onset, Sustain, K, ou aux 4 sorties, plus un cas Sustain+K
  et un cas Onset+Sustain à 25%. Un mélange linéaire de
  probabilités courantes/causales ne regarde jamais le futur,
  ne lit aucune annotation et ne modifie aucun paramètre.
- Décodeur S708 figé, spécialisation S707 historique
  `gate=0.8,persistence=1,grace=24`. Chaque candidat émet
  vraiment un unique flux NoteOn/NoteOff ; toutes les
  comparaisons proviennent de ces événements, pas d'un oracle K.
- Les 32 pistes **CAL** sont les seules à choisir; **six métriques
  non-régressives**, zéro correct Offset perdu, zéro attaque longue
  perdue et TP global non inférieur au témoin S708.
- La politique est gelée avant l'évaluation HELD (32) et DEV (12),
  **historiquement exposés** et non un nouveau TEST indépendant.
  Pas de promotion ni changement de LIVE automatique.

## Résultats attendus

[Workflow S721](https://github.com/Andriamarosoa/note/actions/runs/39114792725).

Les tests initiaux prouvent la parité bit-à-bit de la variante S708,
le mélange uniquement des têtes spécifiées, la normalisation des
probabilités K, l'absence de NaN et l'impossibilité de passer la
promotion si une seule des six métriques ou un Offset
correct antérieur est perdu.

Un run réussi ne signifie **pas** que le mélange améliore
la détection. Les résultats chiffrés ne deviennent vérifiables
qu'une fois `S721 CA.../report.json` et l'étape finale contrôlés.

## Si la fusion échoue

Ne pas chercher la réponse en optimisant DEV. Il faut apprendre à
distinguer les vrais nouveaux onset polyphoniques des fausses attaques
(harmoniques, reprises partielles, résonances) **dans les caractéristiques
et la fonction de perte FIT**. Démontrer cela dans un contrôle
à budget/initialisation identiques puis choisir sur CAL.
