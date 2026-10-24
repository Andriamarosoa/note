---
serie: S876
date: 2026-10-24
statut: termine_local_non_promu
source: S847_GuitarSet_reel
reference: S873_hybride_recherche
promotion: false
---

# S876 — GRU correcteur multibande ±40%

## Méthode et audit
Entraînement distinct, quatre bandes, gain borné ±40 %, régularisation réduite; toujours uniquement DSP8192 comme générateur de note.

## Résultats observés
6/6 hauteurs CAL; distance CAL 0,16548 → 0,14896; anciennes AUDITED 0,20692 → 0,17829. Phrase réelle 3,5 s : 0,44501 contre 0,40664 pour S873, dégradation nette; dernière seconde 0,60925 contre 0,52626.

## Décision de sélection
**Ne pas remplacer S873 : la fidélité se dégrade sur la phrase.** Ne pas transformer ces scores relatifs en « pourcentage de bonne transposition ».

## Contraintes, provenance, limites
**Contraintes** : transposition de guitare aiguë vers grave −12 demi-tons, 256 nouveaux PCM à 44,1 kHz par appel, audio passé autorisé mais aucun futur. La latence intrinsèque du DSP standard n'est PAS nulle ni auditée ici; la performance CPU a été volontairement ignorée. Six têtes NOTE et MIDI inchangés, aucune promotion. 39 extraits réels GuitarSet S847 : 27 TRAIN, 6 CAL, 6 anciennes AUDITED déjà consultées, donc pas de généralisation indépendante. L'enseignant offline librosa −12 n'est jamais présent à l'inférence.
