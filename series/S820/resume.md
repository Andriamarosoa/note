---
serie: S820
statut: en_cours
type: experience
reference: S708
branche: codex/note-s820-ss1-truek-spectral-forensics
run_id: en_attente
commit_sha: en_attente
promotion: false
---
# S820 — Audit vrai K1–K6 et indices spectraux de la régression SS1

## Objectif

Déterminer exactement quelles cardinalités K1–K6, erreurs sous/surcomptage et transitions de cardinalité causent la perte **−558 bonnes décisions sur SS1** avec la politique rétrospective `max_high@0.70` des experts S817, et si les indices spectraux disponibles dans le PCM causal diffèrent entre corrections et faux ajouts. Répondre à l'audit S819 sans inventer de cause acoustique.

## Hypothèse

Les résultats vérifiés de S819 sur les 80 pistes GuitarSet : baseline S815 141946/249411 (56,9125%), politique hypothétique `max_high@0.70` 141263/249411 (56,6386%) : 4867 décisions modifiées, 1637 sous-comptages corrigés mais 2320 surcomptages supplémentaires. SS1 perd −558, dont −487 sur le groupe `SS1-100-C#`; son accompagnement perd −505. Les taux de K>=4 de SS1 et Rock3 sont proches. Il faut calculer les transitions exactes `(vrai K, avant K, après K)` et les indices acoustiques conditionnels, pas conclure immédiatement à des harmoniques ou à un timbre.

## Modifications

Script `scripts/s820_truek_spectral_forensics.py` : matrices entières 7×7×7 par piste et composition ; gains/pertes exactes, K0 diagnostique mais exclu du score Active-K ; scores K1–K6, transitions K1→K4, faux K>vrai, correction exacte et désaccord de spécialistes. Audit de quelques indices acoustiques `sharp` du vecteur [T,49,222] (126/134/142 pour bin fondamental-like et 127/135/143 pour harmonique 2-like), moyenne, écart, contraste de distribution à travers 49 hauteurs. Ces métriques sont des **proxies de DSP**, pas des preuves d'identité d'une fondamentale ; aucune hauteur MIDI n'est attribuée à partir d'un simple compte K.

## Protocole

Restaurer 80 traces S800 originales signées, archive mono PCM GuitarSet à SHA épinglé 44,1 kHz, 306536 frames native MIDI256. Restaurer les 5 checkpoints S815 et 5 checkpoints S817 SHA signés, les données réelles complètes S819. Extraction strictement causale [T,49,222] et GT K0–K6 uniquement pour scorer. 5 plis par composition originelle, train/cal/evaluation conservés, aucune modification du modèle. Contre-factuel `max_high@0.70` **fixé a priori pour reproduire S819** : pas d'optimisation ni promotion. Vérifier la parité piste-par-piste et composition-par-composition avec S819 avant de publier une matrice.

## Audits obligatoires

- 80/80 sources et SHA, 5/5 poids originaux, 5 compositions de 16 pistes, 249411 vrais K>0.
- Chaque frame contribue exactement une fois à la matrice `true_K x old_K x new_K`, K0 conservé.
- Identité arithmétique `gain exact = nouveaux vrais positifs − anciens vrais positifs`, réconciliation des compteurs sur/sous.
- Statistiques DSP conditionnelles sur classes de décision avec prudence sur les harmoniques.
- Aucune re-sélection à partir des résultats outer, aucune fuites CAL32/HELD/TEST, zéro poids entraîné, zéro futur, zéro hold.
- Six axes S708 et LIVE inchangés ; objectif >85% reste ouvert. Les trois modèles NVIDIA S819 ont réellement fourni leurs avis, mais leurs hypothèses nécessitent preuve.

## Statut

Expérience read-only, score et matrices S820 non encore mesurés. L'exécution GitHub validera ou rejettera les vérifications ; les tableaux précédents sont des chiffres S819, pas des résultats S820.
