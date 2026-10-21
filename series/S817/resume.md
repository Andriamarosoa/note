---
serie: S817
statut: en_cours
type: experience
reference: S708
branche: codex/note-s817-activek-temporal-highpoly
run_id: en_attente
commit_sha: en_attente
promotion: false
---
# S817 — Spécialistes temporels privés Active Exact-K K4 à K6

## Objectif

Traiter directement les polyphonies de quatre à six notes, restées faibles en S815, et améliorer Active Exact-K avec de nouvelles têtes indépendantes sans toucher aux cinq autres caractéristiques ou au MIDI LIVE. Viser >85 % après preuves réelles MIDI256 et généralisation indépendante, sans promouvoir d'essai exploratoire FIT80.

## Hypothèse

S815 a progressé K5 7,28→29,78 % et K6 2,13→17,40 % par rapport à S814, mais a baissé K4 44,20→36,93 %. Un décodeur global mono-trame, même avec trois têtes indépendantes, ne traite pas assez bien la différenciation temporelle de notes acoustiquement superposées. De véritables observations au pas MIDI256 avec historique acoustique jusqu'à 16 pas précédents pourraient distinguer K4/K5/K6, sans exiger d'attente après l'attaque.

## Modifications

Deux nouvelles têtes propres à Active-K : détection K>=4 par acoustique passée/courante et classification K4/K5/K6 lorsque l'hypothèse haute polyphonie est crédible. Trois têtes indépendantes S815 gelées, cinq têtes Active-K au total. Historique temps réel privé et borné, aucune information future, deux nouvelles têtes entraînées uniquement sur 48 morceaux de trois compositions. Sélection entre base/override/max_high avec seuils 0,50/0,70/0,90 sur 16 morceaux de calibration. Rejet si le nouveau score régresse. Le score d'une tête Active-K n'est pas le score final MIDI unifié.

## Protocole

80 morceaux GuitarSet FIT80 SHA authentifiés, PCM original mono 44,1kHz, 306536 frames acoustiques réellement calculées à256, 5 compositions 48 entraînement /16 calibration/16 évaluation. Les 15 modèles S815 restent bitwise signés ; la nouvelle architecture reçoit seulement les observations acoustiques 256 et les prédictions privées de l'axe Active-K. Aucune annotation d'évaluation lors de l'entraînement/choix du sélecteur. Mesures S704 H0 et S815 obligatoirement répliquées à l'identique avant le nouveau score, par vrai K et composition.

## Audits obligatoires

Causalité par mutation du futur, streaming découpé invariant, absence de poids de caractère voisin, deux poids spécialistes distincts, contrôle labels sans fuite, classification véritable K4–K6, audit parité S704, métriques exactes K0–K6 et signature WAV/metadata. CPU total et iRig physiques requis pour toute promotion. Consultation NVIDIA pas présumée effectuée ; six audits complets distincts encore à fournir.

## Statut

Conception et scripts créés sur branche S817, résultats sur vraies pistes non encore obtenus. S816 56,4241 % est rejeté, S815 56,9125 % reste comparaison SHADOW, S708 CAL32 officiel préservé.
