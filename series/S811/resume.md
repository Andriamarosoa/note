---
serie: S811
statut: termine
type: experience
reference: S708
branche: codex/note-s811-blind-nvidia-s810-error-attribution
run_id: 39391755628
commit_sha: c0268b86424280b696f079213543e32709c43002
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

## Résultat contradictoire réellement obtenu — deux avis, quatre contre-mesures

GitHub [consultation indépendante #39391755628](https://github.com/Andriamarosoa/note/actions/runs/39391755628) **SUCCESS comme workflow**, avec réponses reçues de Gemma 4 31B et Nemotron 3.5 Lightning 30B. Gemma a produit un texte détaillé mais insuffisamment ancré dans les noms littéraux des fonctions selon le vérificateur ; Nemotron a produit un rapport détaillé et satisfait le filtre textuel. Kimi K3 **HTTP410**. La tentative ciblée d'un troisième avis [#39392258041](https://github.com/Andriamarosoa/note/actions/runs/39392258041) en remplaçant seulement Kimi par `z-ai/glm-5.3` a atteint le délai API, aucun texte reçu ; les deux premiers modèles n'ont **pas** été relancés. **2 réponses externes, 1/3 substantielle selon contrôles automatisés, 0/3 opinions validées comme preuve scientifique autonome**. L'input SHA256 `aeca6dd1545fa49f0a293ad6d40d87e512f91177ae683c34dcf8b2b85a592e18` et le prompt SHA256 `d000bbbbaed1e9f5c85cdf866ec5a638d8386e963856a5ca4fd15e00f36e87d3` sont identiques entre les tentatives.

Contre-vérifications **effectivement mesurées par l'assistant**, sans demander aux modèles de confirmer une idée :
- Rejeu FIT80 des 15 checkpoints Onset, Offset et Full Note originaux, **80/80** empreintes de traces et **15/15** empreintes de modèles vérifiées ; parité exacte des TP/FP/FN S810 pour Onset/Offset. Seuil fixe .5/.7/.9/.95/.99 sans apprentissage : F1 Onset **11,40/16,62/22,81/26,14/34,51 %**, Offset **12,75/16,09/24,12/27,36/32,72 %**. Les seuils sont **exploratoires**, ajuster après avoir observé FIT80 serait une sélection qui nécessite validation indépendamment.
- **Ablation causale du class_weight**, seule variable modifiée lors de **10 nouveaux entraînements contrôlés** (5 folds × Onset/Offset) : conserver les mêmes exemples, transformations, régressions, hyperparamètres, comparateurs OOF et seuil 0,5, supprimer uniquement `balanced`. Onset **11,40→16,01 % F1**, faux positifs **154245→98630** ; Offset **12,75→15,16 % F1**, faux positifs **142827→113655**. Prouve que la pondération contribue matériellement au nombre de FP, sans expliquer tous les échecs.
- **Full Note 4 états**, matrice de confusion complète obtenue sur les vrais 80 FIT et sur 5 checkpoints SHA signés : lignes vraies 0/1/2/3, colonnes prédites 0/1/2/3 : `[[3481217,81890,14198,27769],[992,8518,841,898],[4383,11871,98064,14667],[375,894,1170,5653]]`. Parmi **94 655 faux Onset Full Note**, **81 890** sont du silence véritable (86,5 %), **11 871** du sustain, **894** du release. **Accuracy 95,7386 % inférieure au trivial silence constant 96,0482 %** sur ces labels ; l'accuracy n'est donc pas une preuve de bonne détection des notes.
- **Falsification étiquetage** : les lignes `searchsorted`/fenêtre [t−1024,t] de l'entraîneur S810 acceptent **11 264/11 264** débuts et **11 129/11 129** fins éligibles de ces 80 pistes : **0 événement éligible rejeté** par la condition de fenêtre. Les avis suggérant qu'une « tolérance trop stricte » laisse tomber ces événements sont incorrects sur ces données.
- Les proportions positives S810 naturelles sont **0,30 %** pour Onset/Offset par couple (frame,pitch), mais les caps de classes les portent à environ **33–35 %** avant encore `class_weight='balanced'`, ce qui modifie fortement les priors du problème.

Erreurs de lecture rejetées : Nemotron a appelé « 20 pitches » les colonnes `FEATURE_COLUMNS`, alors que ce sont 20 **variables par hauteur** parmi 23 ; plusieurs références au soi-disant problème de label temporel ne concordent pas avec 0 perte de label. Le fait que l'architecture séparée conserve les autres **sorties SHADOW** ne garantit pas que les six métriques **du même MIDI final** resteront indépendantes. Aucune progression de S708 officielle déduite.

Tous scripts exécutés, données d'audit et réponses réelles externes sont remis dans l'archive conversationnelle vérifiée `S811_NVIDIA_contre_audit_FIT80.zip`, SHA256 `0832f0719d5d3b6ae30db2a25018163f44b3408b82ba0f53c7bcc08a42cb0c1e`. **Aucun entraînement de tous les six axes sur une nouvelle configuration** ni mesure iRig/CPU, aucune intégration MIDI finale.
