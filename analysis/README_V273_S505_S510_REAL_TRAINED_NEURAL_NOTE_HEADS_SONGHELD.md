# S505–S510 — Entraînement réel des sélections neuronales de notes

**Statut le 14 octobre 2026 : R&D seulement ; S80, V88 officiel, Live, onsets et offsets préservés.** Aucun déploiement, aucune promotion et aucun score revendiqué comme validation indépendante et aveugle.

## Architecture / objectif

Nous avons repris l'architecture Exact-K `ClassConditionalAuditSelector` et les **13 têtes** S502–S504 pour entraîner les sélections des notes réelles : `E3`, `G3`, `B3`, etc. Les premières têtes choisissent une proposition **d'accord complet** ; un deuxième réseau sélectionne les hauteurs **individuellement** en regroupant les preuves de plusieurs têtes.

Un résultat doit être audité sur deux cibles séparées :
- **Exact-K** : nombre de notes correspondant à l'annotation à l'instant et selon la convention de l'instrument. Sur guitare IDMT, on évalue les notes actives à +30 ms ; sur piano MAESTRO, les nouvelles notes commençant à l'attaque.
- **Ensemble exact de notes** : toutes les hauteurs correctes, sans omission ni fausse note. Un accord de deux notes avec une hauteur incorrecte est une **régression de notes**, même si K=2 reste correct.

Les prédictions des têtes étaient **gelées avant les XML/MIDI**. Les vrais labels ne sont utilisés que lors de l'entraînement sur les enregistrements autorisés et lors de l'audit final des enregistrements exclus. Mais les cinq morceaux IDMT et les trois MAESTRO avaient été examinés dans les boucles précédentes : **ceci n'est pas un nouveau test aveugle**. Il faut aussi vérifier à terme la provenance d'entraînement des modèles acoustiques pré-entraînés.

## S505 — Construction réelle, sans négatifs inventés

[Run S505 SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/38596480284).

Les matrices comprennent 13 ensembles de hauteurs MIDI observées par événement, `head_mask [N,13]`, `head_k_mask [N,13,7]`, logits, audio 16D, types de têtes, identité de l'enregistrement et label du véritable ensemble MIDI réservé à l'apprentissage/audit.

| Source | Vraies attaques annotées | Appariées à un événement canonique | Notes exactes BP origine | Exact-K du modèle source |
|---|---:|---:|---:|---:|
| Guitare IDMT, modèle K = S80 final | 45 | 45 | 27/45 | 25/45 |
| Piano MAESTRO, modèle K = V88 seul | 322 | 220 | 125/322 | 42/322 |
| **Total** | **367** | **265** | **152/367** | Source différente, ne pas agréger comme un seul modèle |

Les 102 véritables attaques de piano sans correspondant canonique restent **des faux négatifs de détection** ; nous ne les retirons jamais des dénominateurs. Les 15 groupes de notes IDMT et 14 piano sans vérité positive appariée sont **inconnus**, pas de véritables négatifs démontrés. Il n'existe donc pas de mesure de précision sur les faux déclenchements à partir de ces artefacts.

## S506–S507 — Premier vrai entraînement du routeur Exact-K neuronal

[Run S506 SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/38597630105) et [rejeu indépendant S507 SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/38597962024).

Le réseau original `ClassConditionalAuditSelector` TensorFlow a été **effectivement entraîné huit fois**, en excluant entièrement à chaque fois le morceau évalué. Chaque exemple d'entraînement reçoit ses profils d'audit calculés sur les **autres morceaux d'entraînement**. Le morceau tenu à part ne sert à calculer aucun seuil, poids, moyenne de corrections ou risque. Huit jeux de poids ont été sauvegardés avec leur SHA et rejoués indépendamment.

Deux politiques de choix : `KEEP` sauf forte preuve de correction ; ou variante expérimentale moins prudente. Sur cette cohorte, **les deux prédisent zéro modification** : guitar 27/45 accords corrects ; piano 125/322, Exact-K du BP identique à la baseline (guitare 32/45, piano 151/322 quand on ne compte que les 220 attaques appariées). **0 corrections, 0 régressions.**

Cause : la supervision en tête entière est extrêmement rare et les logits de `KEEP` dominent ; une tête ne peut pas prélever E3 de H4 et G3 de H1 tout en rejetant leur faux harmonique.

## S508 — Diagnostic d'une borne supérieure ORACLE

[Run S508 SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/38598303045).

Cette borne utilise volontairement la vraie annotation pour **mesurer la capacité des propositions**. Elle ne doit jamais servir à prendre une décision en inférence, ni être annoncée comme un score du réseau.

| Sur événements canoniques appariés | IDMT guitare 45 | MAESTRO piano 220 |
|---|---:|---:|
| BP originel entièrement correct | 27 | 125 |
| Meilleur cas théorique en choisissant **une seule tête** | 31 | 136 |
| Meilleur cas théorique en recomposant **chaque note** dans l'union des têtes | 34 | 172 |
| Attaques où au moins une note réelle manque de toutes les têtes | 11 | 48 |
| Vrais poly parmi les attaques évaluées | 21 | 59 |
| Meilleur cas théorique d'accords poly exacts en recombinant les notes candidates | 12 | 27 |

Pour les 102 véritables onsets de piano **sans groupe acoustique**, aucune de ces stratégies n'a de proposition musicale à sélectionner. Ces résultats prouvent que **le problème ne se résout pas seulement par davantage d'entraînement du même sélecteur**. Il faut à la fois des sélections note par note et de **nouvelles têtes génératrices de notes candidates** à partir de l'audio (hauteurs absentes, attaques manquantes).

## S509 — Deuxième réseau, attention neuronale par note candidate

[Run S509 SUCCESS](https://github.com/Andriamarosoa/note/actions/runs/38599016032).

`scripts/s509_train_neural_note_by_note_head_attention_recording_held.py` :
- produit une entrée d'attention pour **chaque hauteur effectivement observée** par au moins une des 13 têtes disponibles ; encode les preuves/masks de chaque tête, le contexte audio, la hauteur MIDI observée, l'existence de cette hauteur dans la référence BP, les indices d'harmoniques et le domaine connu ;
- `MultiHeadAttention` TensorFlow avec apprentissage supervisé binaire **présence de cette note** ; les labels d'un morceau ne servent pas à entraîner la sortie de ce morceau ;
- huit modèles entraînés, poids sauvegardés avec SHA ; décisions regroupées ensuite par événement musical canonique ; aucune hauteur hors des notes observées n'est ajoutée ;
- deux politiques **pré-déclarées** : décision diagnostique `p >= 0,50`, décision prudente `ajout p >= 0,85 / suppression p <= 0,15`; en cas de sortie vide le décodeur garde le jeu de notes BP.

| Sur événements annotés et appariés | BP original | S509 diagnostic p≥0,50 | S509 prudent |
|---|---:|---:|---:|
| Guitare, ensembles complets exacts /45 | 27 | **26** | **26** |
| Guitare, K exact /45 | 32 | 30 | 30 |
| Guitare, vrais poly exacts /21 | 7 | 6 | 6 |
| Piano, ensembles complets exacts /220 | 125 | **128** | 125 |
| Piano, K exact /220 | 151 | **156** | 150 |
| Piano, vrais poly exacts sur 59 groupes appariés | 18 | 17 | 18 |

Sur le piano, la variante diagnostique : **4 corrections, 1 régression**, 4 événements erronés→erronés ; sur la guitare : **0 correction, 1 régression**, 3 erronés→erronés. La variante prudente : piano **0 correction/0 régression**, guitare **0 correction/1 régression**.

Ces scores **ne couvrent pas les 102 attaques MIDI de piano sans prédiction** : sur les 322 vrais onsets, la variante diagnostique fait 128/322 notes complètes exactes, contre 125/322 pour BP. Aucun résultat n'est une validation sans régression sur le flux continu, car les groupes inconnus ne sont pas annotés négatifs.

**Verdict :** nous avons entraîné un vrai réseau capable de combiner les hauteurs des têtes, et observé un gain positif mais très modeste sur une source, avec régression sur l'autre. **Les huit poids et prédictions sont experimentaux et NON PROMUS.**

## S510 — Rejeu indépendant des erreurs note par note

`scripts/s510_independent_neural_per_pitch_add_remove_regression_audit.py` vérifie sans TensorFlow, sans retraining, chaque SHA des huit checkpoints, chaque probabilité de hauteur gelée et chaque accord initial/final contre S505. Exporte un journal de toutes les notes **ajoutées, supprimées**, les pitches vrais (uniquement pour le diagnostic), le vrai K et le K source. Un accord avec une note incorrecte reste une **régression** même si Exact-K ne change pas.

## Décision et suite nécessaire

1. Ne **jamais** promouvoir S506 ou S509 sur S80, Live, les onsets/offsets ou les autres modèles officiels. S509 comporte encore des **régressions**.
2. Créer et entraîner de vraies nouvelles têtes **génératrices** de fondamentales manquantes et de nouvelles attaques, plutôt que multiplier les sélections sur les mêmes 13 listes acoustiques. Les 11/45 et 48/220 attaques appariées incomplètement couvertes représentent un plafond.
3. Étendre les propositions des têtes à `ADD_BIRTH/REMOVE_NOTE/CHANGE_PITCH/REATTACK/MERGE/KEEP/ABSTAIN`, en gardant le graphe S498 unique par naissance acoustique et un veto de duplication/accord poly.
4. Constituer **de nouveaux enregistrements** de guitare, piano et autres instruments, avec véritables annotations de notes actives / naissantes, vrais négatifs de début de note, per-instrument/same-K, splits par morceau/joueur/prise, et test extérieur non observé. Les audits S503 rétrospectifs ne sont **pas** des audits OOF de ces nouvelles têtes et ne doivent pas alimenter les mêmes événements.
5. Mesurer sous protocoles harmonisés : exactitude des notes, Exact-K par K0–K6, précision/rappel onset, offset, faux positifs, latence causale Live et régressions par instrument. Sans gain démontré sur ces axes, le système officiel reste strictement inchangé.
