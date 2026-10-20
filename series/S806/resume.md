---
serie: S806
statut: en_cours
type: experience
reference: S708
branche: codex/note-s806-physical-fundamental-contrast-oof
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S806 — nouvelle représentation acoustique inter-octaves

## Objectif

Améliorer Onset F1 et Full Note F1 des notes polyphoniques sans dégrader Exact-K, Active Exact-K, Offset accuracy et Frame Pitch F1, sans augmenter la latence algorithmique ou retoucher S708. Investiguer les faux NoteOn SS1/Jazz1 et les fausses harmoniques avec un nouvel expert spectral acoustique.

## Hypothèse

Les architectures S803 et S805 n'ont pas assez de discrimination : leurs seuils et deux têtes de risque construits principalement sur les probabilités déjà existantes ont régressé. Une **nouvelle représentation** tirée uniquement des mesures acoustiques natives des fondamentales et de leurs relations d'octaves, indépendamment des cinq sorties des experts probabilistes, peut mieux distinguer une vraie note d'un partiel voisin.

## Modifications

Créer `scripts/s806_fundamental_contrast_oof.py` avec seize variables entièrement causales : niveaux des fondamentales aux résolutions 2 048 et 4 096, fondamentale relative, contraste de résolutions, compétition des fondamentales une octave/quinte au-dessous et une octave au-dessus, évolution passée du fondamental, historique acoustique physique et position de hauteur. Un classifieur `LogisticRegression` indépendant est entraîné hors composition : 5 modèles croisés, 4 compositions d'apprentissage et 1 composition d'évaluation. Les experts S795 `general, bass, polyphonic, transient, harmonic` restent gelés et sélectionnés dynamiquement par trame/hauteur. Trois bras gelés `fundamental_consensus` (seuil .50), `fundamental_adaptive` (.35), `fundamental_strict` (.70) n'agissent que sur des suggestions de NoteOn additionnels. Source et NoteOff natifs inchangés.

**Limite de preuve :** les traces S800 contiennent certains coefficients spectro-acoustiques réels issus de S613, mais ne contiennent NI les fichiers WAV originaux NI la totalité des partiels harmoniques. Il s'agit d'un **nouveau contraste acoustique inter-hauteurs sur signaux enregistrés**, PAS d'une véritable séparation fondamentale/harmoniques par spectre complet. Toute revendication d'extraction de nouveaux bins FFT bruts serait mensongère. Une étape future travaillera le WAV pour fournir des données complémentaires.

## Protocole

Évaluation sur les 80 vraies pistes GuitarSet FIT OOF conservées avec SHA256 dans S800. Ne jamais ouvrir CAL32, HELD ou TEST. Le classifieur croisé ne peut voir aucune annotation de la composition à évaluer ; les labels S800 `onset_targets_fit` des quatre autres compositions servent uniquement à la supervision des têtes spectrales, jamais au décodeur. Le décodeur S795 produit les événements MIDI256 réels, évalués avec les six métriques S704 et les 12 compteurs natifs d'origine. Contrôle de référence S803 et 5 comparaisons par composition : veto d'un candidat si **une seule** des six métriques recule globalement ou dans une composition, même en cas de petits gains ailleurs. Aucun seuil ajusté d'après les résultats FIT.

Un premier commit sans `[run-fit80]` déclenche uniquement smoke (tests causaux et syntaxe). **Une seule** exécution scientifique réelle est déclenchée avec un commit explicitement marqué `[run-fit80]` après validation smoke. Éviter les duplications S804. Tout push ultérieur revient à smoke-only ; pas de run NVIDIA déclenché automatiquement.

## Audits obligatoires

Six audits CI indépendants : Données (signatures et 80 pistes), Acoustique (16 dimensions natives uniquement, limites des partiels), Architecture (5 têtes d'origine intactes), Apprentissage (5 modèles sans fuite de composition), Métriques (six ratios/12 compteurs et veto par composition), LIVE (zéro buffer ajouté, performances physiques non mesurées). Consensus local strict 6/6. Trois modèles NVIDIA distincts reçoivent les résultats réels et du code source dans des jobs **manuels exclusivement**, avec consensus 9/9 fail-closed ; sans leur exécution, 9/9 est UNKNOWN, jamais PASS. Les six agents Work personnalisés ne sont pas simulés par les scripts CI.

## Statut

En cours : aucune mesure S806 vérifiée avant le replay réel. S708 demeure référence officielle. Aucune nouvelle tête n'est intégrée en LIVE, latence CPU réelle non testée.
