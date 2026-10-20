---
serie: S811
statut: en_cours
type: experience
reference: S708
branche: codex/note-s811-blind-nvidia-s810-error-attribution
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S811 — consultation NVIDIA NIM indépendante : diagnostics S810

## Objectif

Comprendre sans induction de réponse les faibles F1 SHADOW Onset, Offset et début de Full Note observés en S810. Interroger trois modèles distincts, séparément, sans leur transmettre les hypothèses de l'assistant ni les opinions des deux autres. Après réception, confronter les conclusions au code réellement exécuté puis les rejeter si les preuves sont insuffisantes.

## Hypothèse

Plusieurs causes, y compris interactions de code, de labels, échantillonnage, entraînement, métriques ou calibrage, sont possibles. Aucune cause n'est imposée a priori. La tâche des modèles est un diagnostic **falsifiable** ; un consensus linguistique même unanime ne suffit pas comme preuve d'amélioration.

## Modifications

`series/S811/fit80_blind_evidence.md` contient les six axes, comptes FIT80 et extraits du véritable entraîneur S810 (source SHA256 `687f731fa255d3aaba33fa07beb04b41d836c3d48d6b8c2e98581a2ecf693059`). La tâche n'utilise ni contenu CAL32/HELD/TEST, ni audio, ni secrets, ni labels piste-par-piste. Script `scripts/s811_blind_nvidia.py` : `google/gemma-4-31b-it`, `nvidia/nemotron-3.5-lightning-30b-a3b`, `moonshotai/kimi-k3`; prompt système et preuves identiques pour les trois. Paramètres fournisseur spécifiques uniquement nécessaires à la production de la réponse : Nemotron no-think, Kimi reasoning=low. Aucun partage inter-modèles.

`scripts/s811_review_critique.py` vérifie les empreintes identiques de prompt et preuves et sépare qualité textuelle/validité scientifique. Les réponses externes sont conservées séparément, même en échec HTTP, avec empreintes; elles restent des **opinions non validées**. Un seul run scientifique est autorisé par le marqueur explicite `[blind-s810]` après succès du smoke, sans répétition de modèles automatiquement.

## Protocole

Référence factuelle : 80 pistes GuitarSet FIT80, 5 folds de compositions, 30 checkpoints distincts S810. Onset TP=10002 FP=154245 FN=1247 ; Offset TP=10479 FP=142827 FN=647 ; Full Note proxy attaque TP=8518 FP=94655 FN=2731. Toutes ces valeurs sont des diagnostics par hauteur/trame à 1024 échantillons, et ne mesurent pas directement Onset/Offset/Full Note des événements S704 MIDI256. Le code natif S708 reste entièrement inchangé.

Le protocole de discussion est sans piste présélectionnée : causes concurrentes, éléments pour/contre, test falsifiable, références précises à la fonction et à la ligne, aucune affirmation de progression non mesurée. Après consultation, audit calculatoire indépendant des propositions et éventuelles ablations sur FIT80, sans regarder le CAL ou dégrader le système LIVE.

## Audits obligatoires

Données : provenance de la source signée FIT80 ; Acoustique : absence de fuite future ; Architecture : chaque sortie S810 privée ; Apprentissage : labels et distributions réellement utilisées ; Métriques : vérifier toutes les divisions de précision, rappel et F1, empêcher confusion SHADOW/S708 ; LIVE : refuser toute attente ou confirmation artificielle et ne pas déduire un gain de latence CPU non mesurée. Les trois avis NVIDIA ne sont pas les six agents Work, et trois succès API ne sont pas un 9/9 PASS scientifique.

## Statut

Consultation préparée, sans prétendre que les modèles ont répondu avant la vérification de leurs artefacts. Toute conclusion scientifique devra être distinguée d'une simple hypothèse proposée. Pas de promotion S708.
