---
serie: S827
statut: en_cours
type: experience
reference: S708
branche: codex/note-s827-isolated-s708-activek-midi256
run_id: en_attente
commit_sha: en_attente
promotion: false
---
# S827 — restauration des cinq axes et réintégration S708 **uniquement Active Exact-K**, strict MIDI256

## Consigne utilisateur non négociable

**Seul Active Exact-K est modifiable.** Toutes les cinq autres caractéristiques sont gelées : Exact-K, Onset, Offset, Frame Pitch, Full Note. Aucun nouveau décodage musical global, aucune altération de leurs têtes, poids, sélecteurs, états, entraînements, ni MIDI signé S708/S707. La cadence de décision, d'inférence, et d'extraction de la tête Active-K doit être **exactement 256 échantillons @44,1kHz** (5,805ms par décision). Zéro lookahead, zéro hold ajouté, jamais de rétrodatage. Interdiction de distillation et remplacement de S708 par une tête acoustique arbitraire.

## État antérieur et retour à la base sûre

S826 (#130) a changé un sélecteur et la cadence du réseau S708 **communs** et a fait régresser Onset (−1,67pt), Offset (−1,09pt) et Full Note (−1,96pt) dans un essai FIT80 complet. **S826 n'est pas promu**. La branche S827 est fondée sur **S825**, pas sur S826 : elle n'hérite pas des modifications de S826 dans les fichiers S704, S708, S712. Les cinq projections originales protégées par S812 sont inchangées. Aucun patch dans `scripts/live_strict_notes.py`, aucun patch d'une tête d'un autre axe et aucune intégration au flux MIDI public. La branche dédiée ne crée que des fichiers `s827_*`, plus l'Action et les rapports.

## Architecture privée

`src/causal_note/s827_s708_activek_only.py` reconstruit **exactement les entrées neuronales S708** : frontend original `CausalPitchFeatures(hop=256)` [49,126], postérieurs natifs S613, `latest_s708_acoustic_features` [49,192], `sharp_track` [49,96] refait à chaque *nouvel* hop 256, puis checkpoint **TemporalHeads S708 288D original** via `IncrementalS708TemporalHead` à chaque 256. Une copie **entièrement privée** de l'état `GuidedSelector`, dérivée mot pour mot de ses règles Onset/Sustain/Release/K, calcule uniquement un compte Active-K K0..K6, sans envoyer de NoteOn/Off externe et sans donner accès aux cinq autres décodeurs. Les paramètres de priorité/probabilités S708 sont identiques à la politique S708 SHA gelée. Les poids d'origine ont été entraînés à 1024 : ce port est un témoin, pas un entraînement MIDI256. Aucun résultat de score n'est revendiqué avant la CI.

## Preuves et contrôles

Tests unitaires : parité **état par état** du sélecteur privé contre le `GuidedSelector` original à 1024, identité exacte de la représentation 288D reconstruite sur un signal synthétique chaque 256 avec la fonction S708 offline, une vraie inférence PyTorch par 256, invariance aux tailles callbacks, invariance de préfixe à suffixe futur modifié, refus des discontinuités, conservation bit-identique du MIDI signé H0 sur les cinq autres axes S812. Test **réel** GuitarSet : archive audio SHA épinglée, originales traces S800 signées, vrai checkpoint S708 d'artefact GitHub, relecture de 5 (puis si autorisé des 80) pistes; score de la sortie **privée Active-K seulement** et parité bit-à-bit de chaque compte contre l'exécution authentique native256 de S826, conservée en *oracle de comparaison* (jamais entrée d'inférence). Ce score ne prouve pas une généralisation externe fraîche.

## Interdictions et statut

Aucun merge ni promotion avant parité de sécurité ; on ne prétend pas avoir restauré 76,88 % de S823 (ce score dérive de features S795 as-of 1024 et MIDI historique), ni validé 85 % ni les temps CPU p99/latence iRig. Les temporisations internes héritées de S708 (35,65,120ms de garde dans le décodeur) restent à auditer : il serait faux d'appeler ce mécanisme *zéro délai de confirmation*, même si aucune attente **nouvelle** n'est ajoutée. La production LIVE et les cinq autres têtes restent gelées.
