---
serie: S828
statut: en_cours
type: experience
reference: S708
branche: codex/note-s828-activek-s708-private-readout-256
run_id: en_attente
commit_sha: en_attente
promotion: false
---
# S828 — entraîner une tête privée Active Exact-K sur l'original S708 exécuté à 256

## Objectif

Conserver et certifier la vraie architecture de reconnaissance S708 portée à **une inférence par 256 échantillons @44,1kHz**, tout en entraînant **seulement** une tête Active Exact-K (K0–K6) dédiée sur des représentations authentiques de ce moteur S708. Les cinq autres caractéristiques Exact-K, Onset, Offset, Frame Pitch, Full Note sont strictement gelées, donc pas de modification de leur architecture, état, sélecteur, poids, ni flux MIDI signé. Le moteur historique reste l'unique source du MIDI de production.

## Hypothèse

S827 a démontré que le moteur S708 originel fournit réellement ses 288D et ses sorties à 256, mais qu'un modèle initialement appris à 1024 échoue encore au niveau de précision souhaité. Une tête entraînée et calibrée à 256 exclusivement pour Active-K peut éviter ce décalage temporel sans réentraîner ni partager les paramètres des cinq autres axes. **Il ne s'agit pas d'un réentraînement de toutes les couches neuronales S708**, dont le checkpoint original reste gelé. Le score ne sera accepté que si confirmé par de vraies données, pas par les tests synthétiques.

## Modifications

`src/causal_note/s828_activek_s708_private_readout.py` contient un nouveau lecteur de vecteurs natifs `S828Stream` qui dérive de S827: frontend original [49,126], `latest_s708_acoustic_features` [49,192], `sharp_track` [49,96], réseau original 288D TemporalHeads S708 inféré au nouveau hop 256, extraction de statistiques causales [trame actuelle + 4 dernières décisions] et classifieur **privé** `HistGradientBoostingClassifier` K0–K6. Le sélecteur ne lit que ces prédictions, la sortie Active-K S708 privé et sa CAL interne ; il peut choisir le S708 natif, le readout privé ou le routeur basé sur la confiance, et des décalages K prédéfinis. Aucun patch des classes d'origine.

`scripts/s828_train_s708_private_activek_midi256.py` vérifie les SHA historiques, rejoue de vrais WAV GuitarSet chaque 256 et effectue cinq splits disjoints par composition (3 TRAIN, 1 CAL, 1 TEST, 16 morceaux par composition en FIT80 complet). Comparaison sur EXACTEMENT les mêmes trames GT-actives du S704 signé, audit K1–K6 et fuite train/test/cal. Refus absolu de CAL32/HELD/TEST et de tout apprentissage sur le fold TEST. `test/test_s828_activek_s708_private_readout.py` teste les contrats.

## Protocole

1. Contrôle Git diff : seuls fichiers propres à S828, workflow et rapports ajoutés. S708/S827 et cinq autres architectures inchangés.
2. Tests unitaires synthétiques, causalité par suffixe modifié et invariance S708 à 256.
3. Relecture de 5 pistes originales signées SHA de GuitarSet (1 par composition) avec 5 splits TRAIN/CAL/TEST. Mesurer les comptes private-S828, private-S827, H0 et diagnostiquer les décisions par vraie classe. Pilote seulement, non recevable comme performance généralisée.
4. Si pilote confirme les contrats, répéter la même procédure sur les 80 pistes FIT et 249411 trames GT-actives avec cinq splits composition-disjoints et choix de sélecteur depuis CAL uniquement ; ne jamais promouvoir automatiquement. Validation externe indépendante et contraintes iRig requises pour production.

## Audits obligatoires

Données: audio GuitarSet SHA, annotations S800 signées et vérité uniquement au scoring/entraînement. Acoustique: 288D S708 réellement calculées à chaque nouvel hop 256. Architecture: tous les fichiers des autres axes inchangés et pas de MIDI émis par ActiveK. Apprentissage: seule tête privée K entraînée à 256; S708 original reste gelé. Métriques: score ActiveK sur GT>0, comptes par classe, calibration TRAIN/CAL/TEST distincte. LIVE: zéro lookahead nouveau, pas de latence matérielle certifiée, pas de déploiement.

## Statut

En cours, non promu. S827 validé et protégé comme base. Aucune prétention >85 %, aucun résultat GuitarSet S828 avant un run scientifique réussi. Une évaluation sur FIT80 ne constitue pas une généralisation externe indépendante.
