# S81 — pré-enregistrement du test indépendant S80 (11 octobre 2026)

**Statut réel : DATASET IDENTIFIÉ, TEST EXTERNE NON ENCORE EXÉCUTÉ. Aucune précision externe annoncée.** Les résultats actuels de S80 (83,09 % global ; 43,87 % poly) proviennent des **59 309 fragments historiques** et non de musique nouvelle.

## Les deux sources indépendantes

- **Benchmark principal** : [IDMT-SMT-Guitar V2](https://zenodo.org/records/7544110), publié par Fraunhofer IDMT, archive 1,3 Go `IDMT-SMT-GUITAR_V2.zip`, MD5 `06796e08731bccffaed6ae59361486e4`. Guitaristes/instruments différents de GuitarSet, notes individuelles et polyphoniques, XML et audio 44,1 kHz. Ne pas consulter les labels pour calibrer S80.
- **Mini-test technique** : [GOAT Dataset](https://github.com/JackJamesLoth/GOAT-Dataset), `audio-examples/example1,2` avec audio DI et MIDI, commit externe figé `55c6e6668ef4546e4635c5e7eaad3062cf7f036b`. Seulement deux extraits : ce n'est PAS une évaluation suffisamment large pour promouvoir officiellement le modèle.

Les métadonnées et les empreintes du corpus choisi sont figées avant l'analyse des annotations. Le fichier `analysis/s81_external_source_manifest.json` enregistre la source et la décision de non-promotion.

## Unité mesurée et garde contre la fuite

K = nombre de **débuts de notes attribués au groupe de candidats**, et non nombre de notes qui résonnent encore. Le groupe est déterminé par les *seuls audios* et dure au plus 40 ms (1764 échantillons à 44,1 kHz). Chaque onset d'annotation est attribué une seule fois à l'intervalle le plus proche, à au plus 20 ms (882 échantillons), tie-break premier groupe ; K plafonné à 6.

Les entrées labels ne doivent jamais atteindre les fonctions d'inférence. **Avant d'ouvrir les MIDI/XML** : geler les poids, produire et hasher `row_id, member, starts, ends, sample_rate, incumbent_K, candidate_K`, sans `true_K`. Un audit indépendant compte ensuite les débuts et mesure les résultats appariés en K0–K6, poly K≥2, corrections/régressions et pistes.

**Promotion préannoncée** : gain strictement positif global ET polyphonique, toutes les classes et enregistrements détaillés, images/versions du modèle identiques pour toutes les actions, absence de composants entraînés sur le test. La taille et la robustesse de l'échantillon restent à examiner. Le test miniature GOAT n'est qu'un smoke test.

## Audit d'inférence des artefacts existants

- PR16 `37837235723` fournit quatre jeux de modèles/normalisations et leurs probabilités sur les quatre folds historiques. Les poids PR16 sont conservés séparément.
- S46 `37891336823` fournit 193 politiques **sur les mêmes fragments historiques** ; cela ne constitue pas un extracteur de politiques pour un fichier audio externe.
- S72 `38155427761` n'a jamais produit de sortie hors GuitarSet folds 0/1/2/4.
- S80 `38170647237` conserve quatre `K1_gate_fold*.joblib` et les décisions S80, mais consomme en entrée les probabilités PR16, votes S46 et le **K parent S72 pré-calculé**.
- Conséquence : **le pipeline complet S80 pour une nouvelle entrée WAV n'est pas encore disponible**, malgré la reproductibilité du rejeu de ses anciennes prédictions. Faire semblant d'évaluer S80 à partir d'un tableau historique serait une fuite et non une validation indépendante.

## Travail effectué

- Protocole enregistré ; contrôle du calcul K0..6 et des garde-fous testé **localement sur quatre cas unitaires synthétiques**, jamais présenté comme un résultat indépendant.
- Job GitHub S81 : inventaire de véritables fichiers WAV/MIDI externes et hashes, sans lire les étiquettes MIDI, puis preuve explicite de blocage faute de prédictions externes. Aucun seuil recalibré.
- L'audit complet MIDI indépendant, avec tests, est livré en archive de travail `S81_EXTERNAL_VALIDATION_PREPARATION.zip`.

**Statut officiel** : `freeze_local_combo` inchangé ; S80 candidat expérimental ; aucune promotion de modèle ni gain généralisation attesté avant vraie inférence audio externe et audit pairé.
