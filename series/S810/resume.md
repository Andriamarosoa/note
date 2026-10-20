---
serie: S810
statut: en_cours
type: experience
reference: S708
branche: codex/note-s810-six-trained-axis-oof
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S810 — premier entraînement réel des six caractéristiques indépendantes

## Objectif

Après la séparation de S809, entraîner des poids distincts pour chacun des six axes : `exact_k`, `active_exact_k`, `onset`, `offset`, `frame_pitch`, `full_note`. Aucune caractéristique n'est fusionnée avec une autre au moment de la prédiction ; les six métriques finales ne sont pas recalculées sur des sorties imaginaires et S708 est entièrement immuable.

## Hypothèse

L'utilisation de cibles et de fonctions de pertes indépendantes avec un nouveau classifieur et son scaler par caractéristique peut fournir de nouvelles décisions SHADOW comparables aux anciennes propositions S809 sans régression d'un autre axe lorsque son modèle est remplacé. Cela ne garantit pas qu'un arbitre MIDI final puisse bénéficier de ces propositions sans introduire des faux NoteOn/NoteOff.

## Modifications

Entraînement local authentique sur 80 pistes GuitarSet FIT80 du [run source S800 #39352880035](https://github.com/Andriamarosoa/note/actions/runs/39352880035). Pour chacun des cinq folds de compositions BN1/Funk2/Jazz1/Rock3/SS1, six modèles `StandardScaler → LogisticRegression` distincts sont entraînés UNIQUEMENT sur les quatre autres compositions : **30 checkpoints joblib** avec empreinte SHA256 individuelle et six poids différents par fold. Les labels exact_k et active_exact_k sont des comptes K0–K6 ; l'apprentissage Active-K pondère indépendamment ses frames actives. Les labels Onset et Offset sont des événements par hauteur remis à la grille courante 1024 échantillons, Frame Pitch des occupations par hauteur, Full Note quatre états exclusifs (inactive/attack/sustain/release). Les cibles Full Note sont ici seulement des états de trame, pas un score complet de notes MIDI appariées.

Le programme complet de formation, les 30 checkpoints, les sorties scientifiques et les journaux sont conservés **ensemble dans l'archive de conversation** `S810_six_trained_independent_models_FIT80.zip`, SHA256 `003de0121fe033d81b82c16dddf7f537e3e21fbe197d0aa5b260e4e86282654e`. Le dépôt GitHub reçoit en plus un chargeur SHADOW `src/causal_note/s810_trained_six_axis_shadow.py` qui exige les SHA256 exacts, six classifieurs différents, les dimensions d'entrée et quatre compositions d'entraînement disjointes avant toute inférence ; le module ne fusionne jamais les six prédictions et ne produit aucun événement MIDI officiel. Le programme complet de formation reste dans l'archive téléchargable ; tant que le ZIP n'est pas versé au dépôt, une exécution GitHub autonome ne peut pas reconstruire les 30 checkpoints, limite explicitement déclarée.

## Protocole

Signatures SHA256 des 80 NPZ et JSON FIT80 vérifiées à partir du manifeste S800, compositions et compteurs natifs identiques. Les seules entrées au runtime des classifieurs sont les 23 variables déjà causales de S795 et leurs agrégations au même instant ; aucune vérité terrain n'entre dans l'inférence. Pour chaque composition d'évaluation, aucun de ses labels n'apparaît dans le train ni dans la normalisation du modèle correspondant. Chacun des 30 modèles est entraîné indépendamment et exporté, les poids sont tous différents. Chaque axe est muté séparément et les cinq autres tableaux prédits restent identiques.

Le test mesure les six axes *shadow* à l'horloge de 1024 échantillons. **Il ne mesure pas les six scores officiels S704 du MIDI final**, qui exigent une transcription appariée sur les fenêtres de 256 échantillons. Le flux MIDI S708 n'est pas modifié. Ne pas comparer directement les nombres S810 à CAL32 ni à un score Full Note F1 officiel. L'ancienne politique de veto s'appliquera à un futur modèle fusionné, et doit encore être satisfaite.

## Audits obligatoires

Données : source FIT80 authentique 80/80 et 30 fichiers signés. Acoustique : uniquement les données courantes/passées. Architecture : six modèles/scalers/sélecteurs privés, distincts, indépendants de S809. Apprentissage : leave-one-composition-out des six tâches. Métriques : diagnostics *shadow* et non six scores MIDI256. LIVE : aucune modification ni mesure matérielle. Les validations CI de cette nouvelle série sont à exécuter : les contrôles locaux ne deviennent pas artificiellement « 6 agents Work », ni avis NVIDIA. Le bundle est sans CAL32/HELD/TEST.

## Statut

**Entraînement local FIT80 achevé, non promu.** S810 reste `en_cours` jusqu'à la vérification reproductible des checkpoints dans la CI et à une véritable mesure MIDI256. Un score Active-K SHADOW amélioré n'est pas une promotion de S708.
