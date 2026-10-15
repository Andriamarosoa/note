# S611 — retour aux têtes et sélections de NOTE

S610 avait contourné l'architecture demandée en important directement le modèle et le décodeur de `midi`. **Cette intégration est retirée.** S611 améliore l'exécution de composants de NOTE, sans modifier ses modèles de recherche ou présenter un frontal microphone S80 qui n'existe pas encore.

## Modifications concrètes

- Suppression du réseau TFLite de `midi`, du code importé de reconnaissance/décodage, du moteur S610, du lanceur et de sa CI. Le rapport et les résultats S610 restent archivés, explicitement marqués comme retirés.
- `NativeFeatureStream` : tampon audio borné, événements natifs ordonnés, contexte acoustique réel −80/+160 ms, mêmes calculs S97. Les vues flow/morphologie/physique et la géométrie PR16 sont disponibles pour porter les têtes de comptage.
- PR16 : l'appartenance des trames à l'événement conserve la comparaison avec les voisins. Une certification du producteur indique que tous les événements jusqu'à +240 ms ont été transmis ; aucun prochain événement n'est supposé absent avant cette limite.
- `NativePitchHeads` : chargement unique des quatre S402, quatre S416 et quatre S428, fonctions de caractéristiques et de sélection d'origine. S419 choisit la référence S402/S416 ; S428 utilise cette référence et décide, selon le K, si son classement remplace la référence.
- `NativeNoteStream` : réception des événements et K originaux, production des noms de notes avec K inchangé, tête sélectionnée et disponibilité du contexte. Aucun comptage de pics ne remplace S80, aucune fin de note n'est inventée.
- Checkpoints et deux rapports de sélection inclus, copiés octet pour octet ; SHA256 vérifiées avant leur chargement. [Provenance](../model/live/native_note/provenance.json).

Le chemin de comptage complet (producteurs V84→V88, familles S5/S18/S35/S45/S46, sélections S69→S80 et garde S397) reste inchangé. Les caractéristiques nécessaires à certaines de ces têtes sont maintenant disponibles en flux ; **leur exécution complète depuis le microphone reste à intégrer**. La protection S397 s'applique aux propositions S70 avant S71/S72, pas comme une correction arbitraire du K final.

## Vérifications réalisées

| Contrôle | Résultat |
|---|---|
| Contrats natifs, modèles réellement entraînés, pont MIDI historique et routeur | 30 tests réussis |
| Extraction originale contre tampon circulaire, plusieurs découpages de blocs | Identité bit à bit des sept vues testées |
| S430 original contre prédiction persistante et événement par événement, K0…K6 | Même sélection de têtes et mêmes hauteurs |
| Extrait GuitarSet MIC original de S434, 10 s | 30 événements, 22 notes prédites ; sorties identiques aux archives et au S430 inchangé |
| Rejeu de cet extrait en blocs de 256 et 997 échantillons | 0 différence de noms de notes, 0 modification du K fourni |
| Tous les voisins natifs de l'enregistrement, géométrie et acoustique sur le préfixe mûr | 109 événements identiques bit à bit |
| Sources d'origine protégées | 38 fichiers inchangés, dont les 20 étapes de reconstruction S80, S397, les têtes/routages comparés et les scripts gelés S601 |

Les tests vérifient aussi l'attente du futur réellement reçu, le refus des trous et chevauchements audio, des groupes invalides, de l'historique expiré et du débordement. Une fin de fichier ne provoque pas d'inférence sur un futur rempli de zéros. Les tests K0 et réattaque du pont MIDI historique passent toujours.

[Rapport mesuré](evidence/v273-s611-native-stream/report.json) · [30 tests](evidence/v273-s611-native-stream/tests.txt) · [Sources inchangées](evidence/v273-s611-native-stream/protected_original_files.json) · [Provenance du cas réel](evidence/v273-s611-native-stream/fixture_provenance.json).

## Délai et périmètre exact

Sur le rejeu local en blocs de 256, les fenêtres sont disponibles **160,34 à 165,60 ms après l'événement**. Le maximum de calcul observé pour un bloc est **20,70 ms**, supérieur aux 5,80 ms de durée d'un bloc. Ce test non cadencé ne mesure ni les files d'attente d'un vrai microphone, ni le coût du producteur S80, ni la latence physique. Il ne certifie pas un budget temps réel.

Le délai acoustique de 160 ms vient des modèles natifs actuels, pas d'un réglage de taille de bloc. La géométrie 46D exige jusqu'à 240 ms d'information sur les événements ; le producteur peut encore ajouter son propre retard. La validation emploie une certification conservatrice retardée de 40 ms, durée maximale d'un groupe natif. Les derniers événements dont le contexte n'est pas complet sont laissés en attente, sans score.

Le cas réel est `00_BN2-166-Ab_comp_mic.wav`, réextrait à l'identique de S434, SHA256 de l'extrait `cda97b6199e03a17af3c6fdd7d3569dae221917426215f40f24c818a85ba2ee8`. Les 30 K viennent des prédictions S80/S397 archivées par S434. Les identités, débuts et fins viennent des métadonnées natives du fold 0 ; le champ de vraies classes n'est pas lu. La vérification de géométrie utilise tous les voisins natifs, et non les seuls 30 événements sélectionnés pour les notes.

**Ce n'est pas une validation de précision musicale indépendante.** Le corpus GuitarSet a déjà servi au développement ; l'audio MIC diffère du pickup utilisé pour les événements/K. Aucun score IDMT, MAESTRO ou multi-guitares n'est obtenu par ce portage. L'égalité des sorties ne prouve pas leur justesse musicale. Les scores S601 précédents restent hors ligne et inchangés.

## Suite de l'intégration native

1. Extraire l'ordonnanceur d'inférence S80 de ses scripts de cohortes, avec poids, règles de vote et garde S397 inchangés ; vérifier chaque tête et chaque sélection sur les mêmes entrées.
2. Brancher le producteur natif d'événements et sa certification temporelle sur la capture, puis raccorder les K réellement calculés à ce chemin de hauteurs.
3. Adapter et comparer les têtes plus récentes S569/S578/S601, avec budget de contexte réduit, sans les remplacer par un moteur externe ni promouvoir S430 par défaut.
4. Intégrer les têtes d'attaque/réattaque/fin au pont MIDI, puis mesurer précision, régressions et latence sur enregistrements indépendants et matériel réel.

La promotion de S80 ou une modification du Live historique reste conditionnée à une validation indépendante et à l'audit des régressions. [Commandes reproductibles](../LIVE.md).
