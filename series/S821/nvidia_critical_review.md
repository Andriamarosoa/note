# S821 — Trois avis NVIDIA réels et revue contradictoire

## Réponses attestées

GitHub [workflow #39561989215](https://github.com/Andriamarosoa/note/actions/runs/39561989215) **SUCCESS** : trois appels **réels et indépendants** NVIDIA NIM, [artefact intégral #12327798915](https://github.com/Andriamarosoa/note/actions/runs/39561989215/artifacts/12327798915). Identités **Gemma 4 31B**, **Nemotron 3.5 Lightning 30B**, **GLM 5.3 Flash** vérifiées ; briefing et prompt SHA identiques. Tous les retours sont des **hypothèses sans accès direct aux WAV ni exécution de code**, pas des modèles entraînés ni six agents personnels Work.

## Convergences vérifiables

Tous proposent de mieux distinguer **K4 réellement correct** de **K5/K6 réellement présent**, parce que S820 observe sur SS1 665 anciens K4 corrects transformés en faux K5/K6, contre K5 +152 et K6 +17. Ils recommandent de ne pas utiliser les **moyennes globales** fondamentales/harmoniques sur 49 hauteurs comme preuve d'une fondamentale ; préférer des profils par hauteur et des ablations mesurées. Leurs autres hypothèses (harmoniques/octave, décalage SS1/Rock3, annotations, calibration par style) ne sont **pas confirmées**.

## Contradictions importantes identifiées par contre-audit

- **Gemma** demande d'utiliser les sorties S815/S817 comme entrées du discriminant entraîné sur la partition TRAIN. Risque de biais **in-sample**, puisque ces checkpoints ont déjà été entraînés sur les 48 pistes TRAIN. La correction S821 entraîne ses nouvelles têtes **exclusivement sur les observations acoustiques et labels TRAIN**, sans sorties d'autres modèles comme features ; S815/S817 interviennent uniquement comme propositions **gelées** lors du choix inner-CAL / de l'inférence.
- **Nemotron** parle d'une prétendue « surcharge des poids indépendants » : aucune addition ou partage de leurs poids n'existe dans les workflows. Il suggère de calibrer sur les partitions de **TEST**, ce qui constituerait une fuite : strictement rejeté. Il déduit de l'écart moyen 0,001932 que les indices acoustiques sont non discriminants, mais une moyenne de 49 hauteurs ne permet pas de conclure sur une morphologie par hauteur.
- **GLM** suppose que « SS1 = solo fingerstyle avec résonances longues » et évoque un accordage particulier de la piste C# : **non établi** par les données auditées. Ses hypothèses d'erreurs d'annotation restent une possibilité, non un diagnostic. Il suggère d'ajuster la calibration par style SS1 spécifique, impossible en outer test SS1 sans contaminer la validation.
- Aucune des trois réponses n'a mesuré la latence CPU ou matérielle ; le pas de 256 samples = 5,805 ms de résolution n'est jamais une mesure de latence bout-en-bout.

## Correction falsifiable S821 choisie

Deux **nouvelles têtes privées** Active-K, (P(K\geq5\mid K\geq3)) et (P(K\geq6\mid K\geq4)), entraînées sur 48 morceaux de trois compositions TRAIN, avec échantillonnage par classe inversement pondéré, observations spectrales **par hauteur** (49 bins et compétition avec parents 12/19/24 demi-tons), sélection de la politique sur une **autre** composition inner-CAL de 16 pistes. Le dernier groupe de 16 pistes n'est jamais consulté avant la mesure finale. Aucun modèle d'Onset/Offset/FramePitch/Exact-K/FullNote n'est chargé.

Le garde-fou **ne peut modifier que les trames pour lesquelles S815 prédit K4**. Les politiques fixées sont : BASE, CONFIRM5 (expert S817 K>=5 et nouvelle tête concordent), VERIFY5 (nouvelle tête), VERIFY56 (vérificateur K6 sous confirmation plus forte). Seuils prédéclarés 0,40/0,55/0,70/0,85 ; une seule politique est choisie par pli par **meilleure précision Active Exact-K sur inner-CAL**, avec BASE inclus en alternative. Aucun sur-ajustement sur SS1 ou Rock3 outer n'est permis.

L'expérience est **shadow only**, MIDI256 causal sans futur/hold, compare S815/S817 historiques bit-exact et teste les cinq compositions GuitarSet, avec preuve par vrai K4–K6. Puisqu'il s'agit de FIT80 très exploré, un gain éventuel reste un **résultat de recherche**, pas une certification sur un jeu inédit. Pas d'unification ni de promotion sans validation des six axes et mesure LIVE.
