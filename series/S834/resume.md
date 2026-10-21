---
serie: S834
statut: en_cours
type: experience
reference: S708
branche: codex/note-s834-s809-six-axes-native256
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S834 — Portage fidèle S809, six architectures isolées vers PCM256 strict

## Objectif
Construire **une version de S809 à 256 échantillons**, et non un dérivé Exact-K S829–S833. Préserver intégralement les six chaînes originales HEAD → SELECTOR → DECODER, chacune avec son sélecteur, son état et ses hypothèses MIDI privées. Aucun correctif dans un axe ne peut changer une décision ou une sélection des cinq autres. Appliquer la cadence **256 échantillons à 44 100 Hz = 5,805 ms par décision**, sans attendre quatre blocs ni observer des échantillons futurs.

## Hypothèse
Les six axes S809 ne sont pas essentiellement incompatibles avec PCM256 : c'est `s763_independent_decoders.DECISION_SAMPLES=4*256` et l'usage de `S800` postérieurs historiques espacés de 1024 qui imposaient l'ancienne grille. Une copie indépendante avec `DECISION_SAMPLES=256`, alimentée par **S708 recalculé sur les vrais PCM256**, conserve le chemin S809 de prédictions déjà élaborées, notamment son modèle d'harmoniques et ses informations temporelles. En conservant une unique source **immuable** et six copies privés, les 30 non-interférences directionnelles restent vérifiables.

## Modifications
- `src/causal_note/s834_native256_decoders.py` : copie à horloge 256 de `s763_independent_decoders.py` sans modifier l'ancien fichier ni les poids. Tous les six décodeurs strictement isolés ; `FullNoteDecoder.min_age_samples=0` interdit tout hold supplémentaire de 2 anciennes fenêtres, ouverture/fermeture à la frontière actuelle ; onset refractory n'introduit pas d'attente sur une attaque acceptée.
- `src/causal_note/s834_s809_six_axis_pcm256.py` : copie du code S809, les 6 banques de 2 propositions, leurs 6 sélecteurs et décodeurs, avec vérification du même timestamp à chaque 256, sans changement des interfaces de production.
- `src/causal_note/s834_s809_native256_audio.py` : adaptation à l'acoustique **S827/S708 authentique** : PCM44.1k → frontend natif → 288D S708 recalculées → S708 temporel causal original gelé une fois par 256 → `Fragment(post=[49,3], k=[7])` entièrement immuable → six chaînes S809 ; ni sortie ni état d'une autre chaîne en entrée.
- `scripts/s834_s809_native256_fit80_replay.py` : vérifie SHA sources, 80 WAV, 306 536 nouveaux postérieurs S708, six métriques SHADOW sur GT, témoin historique H0 inchangé, **6×5 comparaisons de non-interférence sur 64 premières frontières par piste**. Sauvegarde diagnostics et six scores isolés, sans fausse revendication sur un seul MIDI final.
- `test/test_s834_s809_native256_axes.py` : 11+ tests adversariaux S809 originaux recalculés sur 256, y compris attaques d'une tête, d'un sélecteur ou d'un décodeur sur chaque axe.

## Protocole
Test préalable de non-interférence sur des signaux synthétiques, puis replay des cinq compositions FIT80 signées d'origine (16 × 5 pistes) avec véritables PCM. Exactement une décision par bloc complet de 256, y compris K0. Vérification des 30 relations non-interférentes sur 80 pistes × 64 frames = **153 600 contre-exemples**. Pour tout axe modifié, les cinq autres restent strictement identiques aux autres propositions sans modification et tous les événements MIDI publics S708 restent à l'identique. Compare les six diagnostics SHADOW aux valeurs H0 **sur la même cohorte**, mais les 84,24 % S809 à 1024 proviennent d'un protocole différent ; ne pas fabriquer un score natif de 84,24 %.

## Audits obligatoires
- Données : SHA checkpoint S708 et 80 audios GuitarSet + provenance S800 intacte.
- Acoustique : frontal temporel recalculé sur PCM256, S708 288D, pas d'as-of1024, causalité future et cadence stricte.
- Architecture : six propriétaires distincts, 30/30 relations protégées, aucune mutation S708/S828/MIDI.
- Apprentissage : source S708 gelée, entraînée historiquement à 1024 ; **aucun entraînement S834 nouveau** revendiqué.
- Métriques : séparées par axe, dénominateurs exacts K0–K6, métriques source signées non modifiées.
- LIVE : pas d'augmentation algorithmique du lookahead/hold mais **p95/p99 physique iRig à mesurer** ; 5,805 ms est une période de décision et non une latence mesurée.
- Agents NVIDIA / Work : non exécutés, preuves CI seulement.

## Statut
En cours, zéro promotion. Les scores S708 officiels et la référence de développement S828 restent protégés ; une amélioration ne pourra être revendiquée que sur métriques strictement comparables, pas uniquement sur le fait que l'architecture existe.
