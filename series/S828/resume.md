---
serie: S828
statut: termine
type: experience
reference: S708
branche: codex/note-s828-activek-s708-private-readout-256
run_id: 39604850899
commit_sha: bec403685379a581554f4d6d2d1e6629370e477f
promotion: false
---
# S828 — S708 original natif256, apprentissage privé Active Exact-K (80 pistes)

## Objectif

Reproduire des informations S708 neuves toutes les **256 valeurs @44,1 kHz (5,805 ms)** et entraîner **uniquement** une tête de cardinalité Active Exact-K K0..K6. Ne modifier aucune tête, aucun sélecteur, aucun décodeur, aucune sortie ni état des cinq autres axes (**Exact-K, Onset, Offset, Frame Pitch, Full Note**). Conserver la production MIDI S708 originale intouchée. Mesurer sur les **mêmes 80 pistes FIT80** que S823; interdire tout bilan de performance fondé sur seulement 5 pistes.

## Hypothèse

Le réemploi direct à 256 d'un réseau S708 entraîné à 1024 peut produire des scores Active-K dégradés même si les 288D sont recalculées. Un nouveau readout de cardinalité, entraîné sur les observations S708 authentiques à 256, doit corriger ce décalage sans modifier les paramètres ni les outputs de cinq autres axes. Les couches temporelles/harmoniques d'origine restent gelées. Le score doit être jugé **sur 80 pistes et 249 411 trames GT actives**, pas sur le pilote.

## Modifications

Nouvelle classe `S828Stream` dans `src/causal_note/s828_activek_s708_private_readout.py`: extraction native S708 288D (192D contexte + 96D sharp), inférence originale S708 à chaque 256, nouveaux vecteurs causaux de trame courante et historique ActiveK **privé**, tête `HistGradientBoostingClassifier` K0..K6 entraînée et sélecteur réservé à Active-K (S708 privé / candidat appris / routeur). **Aucune mise à jour de poids neuronaux S708**; nouvelle tête seule apprise, aucun input MIDI H0 ni sortie d'une autre caractéristique à l'inférence.

Nouveau script `scripts/s828_train_s708_private_activek_midi256.py` et tests `test/test_s828_activek_s708_private_readout.py`. Seuls des fichiers S828 ont été ajoutés dans une branche dédiée au-dessus de S827 ; contrôle automatisé du Git diff contre la base S827, sans patch d'un seul composant commun ni du LIVE.

## Protocole

Le [run GitHub #39604850899](https://github.com/Andriamarosoa/note/actions/runs/39604850899), commit de calcul `bec403685379a581554f4d6d2d1e6629370e477f`, a réussi. Audio et métadonnées signés SHA, exactement **80** enregistrements des cinq compositions. Cinq plis composition-disjoints : **48 TRAIN / 16 inner-CAL / 16 OUTER-TEST** ; choix du sélecteur depuis CAL seulement, sans labels OUTER. 249 411 trames actives sur grille PCM256 à 44,1 kHz, **identiques à S823**. Prouver la reproduction du témoin historique H0 `142213/249411` avant de publier les scores. L'entraînement du readout uniquement est composition-disjoint ; le checkpoint S708 préentraîné n'est pas prouvé end-to-end OOF sur ce corpus : pas de revendication de généralisation nouvelle externe.

**Scores FINALS (sur 80 pistes identiques)** : S823 ancien S795 1024 as-of **191741/249411 = 76,877523 %** ; S828 nouveau readout natif256 **184143/249411 = 73,831146 %** ; S827 original gelé natif256 **152555/249411 = 61,166107 %** ; H0 original **142213/249411 = 57,019538 %**. Écart S828 vs S823 : **−7 598 bons comptages, −3,046377 points**. S828 sous-comptages **41 647** (S823 **34 611**), surcomptages **23 621** (S823 **23 059**). Les écarts les plus lourds sont SS1 −2 821, BN1 −2 061, Funk2 −1 696. Le **44,414368 % sur cinq pistes** était un smoke pilote **non recevable** comme score de référence FIT80, à ne plus afficher comme bilan principal.

## Audits obligatoires

Données PASS (80 WAV SHA + H0 signé), acoustique PASS (288D fraîche PCM256), architecture PASS (5 axes et MIDI productif inchangés), apprentissage PASS (readout privé entraîné à 256, sélection inner-CAL mais upstream S708 pas certifié OOF), métriques PASS (ActiveK GT>0, 249411, scores par composition, K1..K6 dans l'artefact), LIVE UNKNOWN (aucun passage production, ni CPU p99 ni iRig prouvés). Aucun ajout de délai d'observation future; le décodeur privé S708 conserve ses anciennes règles de garde temporelle, donc aucune affirmation de latence absolue 5,80 ms.

## Statut

**S828 TERMINÉ, NON PROMU**, objectif 85 % non atteint et score à −3,05 points de S823 sur FIT80 comparable. [PR #132](https://github.com/Andriamarosoa/note/pull/132) reste expérimentale et DRAFT. **Les 5 autres architectures et le LIVE officiel sont intacts**. Le succès d'un run n'autorise pas la promotion. Pour S829 : audits profonds des sous-comptages par classe et composition, tests isolés ActiveK seulement, sélection CAL uniquement, 80 pistes obligatoires avant conclusion.
