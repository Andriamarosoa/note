---
serie: S820
statut: termine
type: experience
reference: S708
branche: codex/note-s820-ss1-truek-spectral-forensics
run_id: 39559635752
commit_sha: 677dc6f486d0a347cde8d6d155b3aa07301ae822
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


## Clôture : résultat mesuré sur 80 enregistrements GuitarSet (21 octobre 2026)

Exécution réelle [GitHub Actions S820 #39559635752](https://github.com/Andriamarosoa/note/actions/runs/39559635752) **SUCCESS**. L'artefact [#12326639460](https://github.com/Andriamarosoa/note/actions/runs/39559635752/artifacts/12326639460) contient 80 matrices exactes (GT K0–K6, S815 K0–K6, politique S819 max_high@0.70 K0–K6), métadonnées SHA PCM MIDI256 et proxies acoustiques. La politique n'est pas promouvable : elle avait été examinée auparavant sur FIT80. Aucune nouvelle tête entraînée, aucun changement S708/LIVE.

Parité confirmée : S815 141946/249411 = **56,9125 %**, max_high@0.70 141263/249411 = **56,6386 %**. Sur SS1, 1783 frames modifiées : 355 corrections contre 913 décisions correctes détruites, perte nette **−558**. Répartition des variations de décisions exactes par vérité : K1 −1, K2 −8, K3 −163, **K4 −555**, K5 +152, K6 +17. Sur les vrais K4, le sélecteur transforme **597 K4 correctement prédits en K5** et **68 autres en K6** : **665 faux ajouts** alors que 5 et 6 notes restent trop peu détectées.

L'analyse des moyennes spectrales DSP donne un écart faible pour le proxy fondamental−harmonique2 entre les frames corrigées (**0,280527**, 355 frames) et les frames détériorées (**0,278595**, 913 frames), soit **+0,001932**. Ce simple indicateur agrégé n'attribue pas les erreurs à un timbre, une harmonique ou une octave : aucune note ajoutée précise n'est identifiée par ce décodeur de cardinalité K.

Autres compositions, corrections / décisions détériorées : BN1 248/269, Funk2 354/346, Jazz1 99/233, Rock3 222/200. Les différences par composition restent posthoc et ne peuvent servir à sélectionner des portes sur les outer folds.

**S820 TERMINÉ EN AUDIT, AUCUN GAIN MUSICAL NI PROMOTION.** Cause décisionnelle immédiate identifiée : confusion du vrai K4 avec K5/K6 dans SS1, surtout les passages 4→5. Prochaine investigation : preuves fondamentales par hauteur et confusion harmonique réelle sur les transitions individuelles, puis protocole de validation externe pour toute correction.
