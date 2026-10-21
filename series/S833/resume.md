---
serie: S833
statut: en_cours
type: experience
reference: S708
branche: codex/note-s833-crossfit-risk256
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S833 — Sélection locale Exact-K avec cross-validation CAL par piste

## Objectif
Tester si quatre sous-validations CAL par piste peuvent mieux estimer la stabilité des corrections polyphoniques que l'unique séparation 12/4 de S832, sans régression du modèle référence S829 ni interaction entre caractéristiques. Garder une décision strictement causale tous les 256 échantillons 44,1 kHz, sans hold ni lookahead ajouté.

## Hypothèse
S832 a écarté tous les candidats avec un veto très strict sur la pire piste, alors que plusieurs règles augmentaient leur nombre de bonnes décisions dans une sous-validation de calibration. Une répétition avec quatre quarts CAL distincts peut distinguer les règles robustes des illusions de sous-échantillon. Réussite uniquement si le score apparié OUTER et la répartition des régressions sont démontrés.

## Modifications
Le sélecteur privé `src/causal_note/s833_crossfit_risk_selector256.py` reprend les 20 candidats Exact-K de S832 (quatre modes × cinq seuils) et les cellules (mode, seuil, K source, K proposé, confiance musicale positive). Il calibre cinq familles de limites (minimum d'observations, rapport corrections/régressions, borne Wilson et gain minimal), sur quatre sous-folds track-disjoint de la seule composition CAL. Il exige un gain sur au moins trois quarts CAL, un gain agrégé, une proportion raisonnable de pistes détériorées et une limite au pire taux de régression par piste. En cas de doute, le sélecteur laisse les règles vides et les décisions S829 bit-identiques. Aucun modèle des cinq autres axes n'est importé.

`scripts/s833_fit80_crossfit_local256.py` est un replay isolé et conserve le score apparié S829/H0. `test/test_s833_crossfit_risk256.py` vérifie quatre quarts distincts, acceptation d'un vrai candidat synthétique, repli exact, noms dupliqués et inférence sans étiquettes.

## Protocole
S800 GuitarSet FIT80 de 80 WAV, cinq compositions. Par fold 3 compositions TRAIN (48 pistes) pour les cinq modèles privés S831, 1 inner-CAL (16 pistes) pour calibrer le risque en 4 × 4 pistes, 1 OUTER (16 pistes) pour noter sans calibrer. 306 536 frontières MIDI256. Compare S833/S829/H0 sur exactement les mêmes trames.

**Limite scientifique majeure :** les résultats de FIT80 de S829–S832 ont déjà été examinés pour orienter cette architecture S833. Même si l'évaluation OUTER reste séparée mathématiquement dans chaque fold, ce test de R&D n'est **pas** un HELD inédit et n'est pas une preuve externe de généralisation indépendante.

## Audits obligatoires
- Données : contrôles S800 et SHA/80 pistes.
- Acoustique : PCM44,1kHz 256 et fenêtres causales, aucune correction à partir de GT lors de l'inférence.
- Architecture : ensembles de modèles propres à Exact-K ; aucune modification des cinq autres chemins, ni MIDI officiel.
- Apprentissage : quatre validations CAL disjointes, sélection de paramètres uniquement CAL, entraînement TRAIN des têtes.
- Métriques : toutes les trames K0–K6 (silence inclus), gains/régressions et veto vs S829.
- LIVE : aucune mesure p95/p99 réelle iRig ; pas de promotion, pas de prétention de latence globale.
- Agents NVIDIA/Work : non invoqués.

## Statut
En cours, expérimentation seulement. S829 57,5247 % et S832 57,5247 % sur FIT80 restent les témoins. S831 56,9943 %. Aucun gain S833 annoncé avant replay signé ; pas de promotion.
