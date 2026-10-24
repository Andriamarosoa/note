# Audit du code — Onset PCM256, référence de travail 69,2584331 % F1

**Date d'audit :** 24 octobre 2026  
**Base immuable :** `codex/note-onset-learned-hgb-arbiter256`, commit `e26cf43535c61795e5c2b1a63fbde85841edb79d`.  
**Branche d'audit :** `codex/note-onset-code-audit-256-20261024` ; seules de nouvelles preuves de tests et ces documents sont ajoutés. Aucune modification du modèle, des six têtes, de la chaîne Live ni du MIDI.
**Source principale :** https://github.com/Andriamarosoa/note/actions/runs/39957354552 ; [run S834 indépendant](https://github.com/Andriamarosoa/note/actions/runs/39658093988).

## Verdict synthétique
**Audit de code : blocages scientifiques CRITIQUES et duplication CPU CONFIRMÉE ; arbitre HGB REJETÉ ; aucune promotion.** Le F1 69,2584331 % est le contrôle effectivement réentraîné et rejoué sur 80 WAV de **cette architecture**. Le modèle historique archivé différent donne 69,2062044 %. Le HGB ajouté donne 69,2489759 % et régresse. Le chiffre 71,382285 % S834 appartient à une autre chaîne utilisant le professeur S708 entraîné à 1024; ne pas mélanger ni promouvoir ces mesures.

## Exécution / chiffres vérifiés
Sur **80 vrais WAV GuitarSet** et **306 536 décisions PCM256** :
| Condition | TP | FP | FN | F1 |
|---|---:|---:|---:|---:|
| Acoustique 195D, réentraînée/rejouée (référence actuelle de cette branche) | 7 607 | 3 093 | 3 660 | **69,2584331 %** |
| Acoustique + arbitre HGB 207D | 7 607 | 3 096 | 3 660 | **69,2489759 %** |
| Acoustique 195D archivée (modèles historiques distincts) | 7 585 | 3 068 | 3 682 | **69,2062044 %** |

L'étape GitHub `full_80_wav` finit en **FAIL scientifique attendu** car `candidate_f1 <= reference_f1` malgré l'entraînement et le replay intégral ; ne pas assimiler ce FAIL à un crash d'inférence. Les scores finaux viennent des journaux `CANDIDATE_FULL_WAV`/`CONTROLS` et de l'artefact `ONSET_LEARNED_HGB_ARBITER_80_WAV_PCM256`, ID 12506429382.

## Audits de code, triés par gravité

### A01 — CRITIQUE : les candidats HGB sont mathématiquement insuffisants pour 80 %
**Preuves :** `src/onset256/learned_hgb_arbiter.py` lignes 91-119 et 141-152 ; `scripts/onset_learned_hgb_arbiter256.py` lignes 107-128. Le HGB ne propose **que les pitchs déjà dans `baseline_pitches`**, issus des sorties HGB de référence. Le gate peut accepter un candidat ou le refuser, jamais créer une note hors de cet ensemble ni annuler une note acoustique directe.

`HGB_GATE_CANDIDATES` du run donne les labels positifs incrémentaux : **BN1 0, Funk2 4, Jazz1 0, Rock3 631, SS1 167, total 802** ; 99,50 % sont dans Rock3/SS1. Sur 3 660 FN, un oracle idéal acceptant ces 802 sans aucune nouvelle FP plafonnerait à **73,863586 % F1** `(TP=8409, FP=3093, FN=2858)`. C'est **6,14 points sous 80 %**. Le plafond est conditionnel à cet ensemble de candidats et au témoin courant, pas à toute future tête acoustique.

**Action vérifiable :** auditer les 2 858 FN non couverts, par voix/pitch, hauteur, polyphonie, solo/accompagnement, attaque récente/restrike et spectres, puis entraîner un **générateur d'attaques causal nouveau et indépendant** à partir de l'acoustique. Un nouveau seuil HGB seul ne suffit pas.

### A02 — CRITIQUE scientifique : les 80 FIT ne constituent pas une validation externe indépendante
**Preuves :** `series/ONSET_S701_256/diagnostic_scientifique.md` et `training_ledger.json` ; `scripts/onset_acoustic_direct256.py` lignes 213-232 et 289-295 ; `scripts/onset_learned_hgb_arbiter256.py` lignes 190-255 (`past_fitted_reference_overlap=True`).

S701 TRAIN contient **48/80** enregistrements identiques de FIT80, S701 HELD contient les **32/80** autres, et S613 TRAIN recouvre **20/80**. Les nouveaux modèles d'arbitrage excluent l'OUTER de leur propre ajustement, mais leur *chaîne amont* a déjà vu une partie du benchmark. **Aucune fuite d'annotation au moment de l'inférence n'est détectée**, mais ces scores ne constituent pas une mesure de généralisation extérieure ou un vrai OOF end-to-end.

**Action vérifiable :** gel des hyperparamètres et modèles sur TRAIN/CAL distincts au niveau prise/composition/joueur ; score finalement sur corpus indépendant jamais exploré. Étiqueter FIT80 comme benchmark de développement historique.

### A03 — MAJEUR code CPU : la même prédiction directe est calculée deux fois par bloc dans l'enveloppe HGB
**Preuves :** `src/onset256/acoustic_live.py` lignes 63-74 appelle `self.model.predict_proba` ; `src/onset256/learned_hgb_live.py` lignes 30-40 appelle d'abord `self.acoustic.push`, **puis** `self.acoustic.model.predict_proba` sur la même `self.acoustic.reference.risk.acoustic_evidence`.

Même si le gate HGB est désactivé, l'enveloppe procède à un second calcul quand `acoustic.model` n'est pas `None`. Ce code garde les mêmes décisions si le modèle déterministe renvoie les mêmes probabilités ; il consomme des ressources inutiles, ce qui doit être mesuré avant tout p99 Live. Un test de comptage de `predict_proba` prouve deux appels pour chaque hop dans cette configuration.

**Correction suggérée hors référence officielle :** conserver en lecture seule `last_acoustic_probabilities` après `AcousticOnsetStream.push`, puis les réutiliser dans `ArbitratedAcousticOnsetStream.push`. Tester identité bit-à-bit sur 80 WAV + CPU p95/p99 sur matériel; ne pas annoncer de gain avant mesure.

### A04 — MAJEUR reproductibilité : les chiffres archivés et réentraînés divergent
**Preuves :** `series/ONSET_LEARNED_HGB_ARBITER256/resultat.md`, `scripts/onset_learned_hgb_arbiter256.py` lignes 47-69, 263-282 ; [run #39957354552](https://github.com/Andriamarosoa/note/actions/runs/39957354552).

Archived **7585/3068/3682** (69,206204 %) ; nouveau TRAIN+REPLAY **7607/3093/3660** (69,258433 %). Soit +22 TP, +25 FP et −22 FN. Le code reconnaît explicitement la dérive et traite le deuxième résultat comme **variante distincte**, mais la cause racine n'est pas démontrée. Ne jamais attribuer les 825 pertes d'une variante à l'autre sans nouveau matching événement par événement.

**Action vérifiable :** comparer par composition les SHA de données, matrices de features 195D, échantillonnage des négatifs, versions sklearn/numba, hyperparamètres et checkpoints de chaque fold. Enregistrer des hashes et scores par piste et non uniquement des totaux.

### A05 — MAJEUR conformité d'architecture : cadence 256 ≠ fenêtre acoustique limitée à 256
**Preuves :** `src/onset256/acoustics.py` lignes 9-12, 46-49, 81-100. Les blocs reçus par `AcousticOnsetStream` sont de **256 échantillons nouveaux**, mais `CausalPitchFeatures` garde un historique et calcule des FFT Hann de **2048/4096/8192 échantillons passés**. Ceci est causal et ne prouve pas un ajout d'attente ; cela utilise cependant jusqu'à ~185,76 ms d'historique pour chaque inférence.

**Point de conformité :** si « tout doit être à 256 » interdit **toute** fenêtre supérieure à 256 (pas seulement le pas de décision/entraînement), le chemin actuel **n'est pas conforme**. Si la règle autorise un historique strictement passé tout en imposant des décisions à pas 256, alors l'historique multi-résolution peut être licite. Aucun modèle basé sur 1024 entraîné (S708) n'est utilisé dans cette chaîne Onset, mais elle exploite plus que 256 échantillons *passés*. Ne jamais dissimuler cette nuance.

### A06 — MAJEUR déploiement : cinq modèles OOF séparés, pas un modèle Onset de production validé
**Preuves :** `scripts/onset_learned_hgb_arbiter256.py` lignes 190-262, rapport `model_count=5` ; `src/onset256/learned_hgb_live.py` `load_arbitrated_stream(..., fold=...)`. Une composition OUTER possède son checkpoint et ses seuils. Une guitare Live inconnue ne fournit pas de `fold` GT/composition ; il faut choisir/entraîner un checkpoint de déploiement unique **avant** de mesurer le Live. Le moteur MIDI officiel n'a reçu aucune promotion de ce HGB.

### A07 — MODÉRÉ : métrique F1 ±80 ms ne prouve pas les 5,805 ms de latence physique
**Preuves :** `scripts/replay_native256.py` `match_onsets(..., tolerance=.080)` ; `src/onset256/acoustic_attack.py` `DirectAttackSelector.gap_hops=14`; `src/onset256/learned_hgb_arbiter.py` `GatePolicy.gap_hops=7`.

L'attaque est émise sur le front courant (sans buffer futur), mais une tolérance de ±80 ms peut compter TP une attaque musicalement tardive. Les `gap_hops` sont des refus de réattaque, pas une attente de confirmation, et peuvent supprimer certains Onset rapprochés. À mesurer séparément : F1 événement, écart temporel médian/p95 entre GT et décision, temps CPU p99 + acquisition iRig + rendu piano Windows.

### A08 — POINT POSITIF : vrai replay WAV causal et modèle 256 SHA protégés
**Preuves :** `src/onset256/model.py` hash S613/S701 et absence S708 ; `src/onset256/acoustic_attack.py` `push` refuse les blocs non-256/non contigus ; `scripts/onset_learned_hgb_arbiter256.py` lignes 288-369 vérifie l'identité des événements Live/caches et les 80 WAV réels, avec checkpoints gelés par fold. `learned_hgb_arbiter.py` applique les événements au front courant uniquement, sans NoteOff ni écriture du MIDI de production.

Les 80 WAV prouvent un système de recherche PCM256 **causal et reproductible à l'intérieur de ce run**, même si les modèles amont ne garantissent pas une généralisation extérieure. Ils ne prouvent pas une stabilité p99 sur le périphérique cible.

## Plan correctif ordonné, sans changer les autres axes
1. **En priorité scientifique** : générer des candidats Onset indépendants non limités aux 802 propositions HGB et évaluer leur couverture sur une composition OUTER réellement hors apprentissage amont.
2. **En priorité code CPU** : dédupliquer la prédiction 195D (petite modification isolée) avec test de parité événementielle stricte ; conserver la référence 69,258433 % inchangée avant preuve.
3. **En priorité protocole** : résoudre la dérive des deux variantes 69,206/69,258, SHA et métriques par piste, puis calculer les FN/FP gagnés-perdus par identité et texture.
4. **En priorité Live** : clarifier le contrat fenêtre maximale/historique causal, geler une tête de déploiement unique, mesurer p99 de bout en bout Windows/iRig sans ajout d'attente.
5. Dans toutes les boucles, garder Exact-K, Active Exact-K, Offset, Frame Pitch et Full Note **strictement gelées**. Un gain Onset avec une régression d'un autre axe ou hors budget ne doit pas être promu.

**Conclusion :** le problème scientifique principal est la couverture des candidats et leur déséquilibre TRAIN/CAL ; un défaut CPU concret ajoute une surcharge mais ne peut pas expliquer, à lui seul, les 3 660 FN. Le passage à >80 % n'est pas prouvé, pas promis et les modèles historiques sont conservés.
