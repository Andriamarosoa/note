# S177 — source de validation polyphonique indépendante enregistrée (annotations closes)

[Workflow S177 validé](https://github.com/Andriamarosoa/note/actions/runs/38207396480).

L'API de [Zenodo IDMT-SMT-GUITAR V2](https://zenodo.org/records/7544110) a confirmé la présence d'une archive publique :
- `IDMT-SMT-GUITAR_V2.zip`
- taille : **1 329 918 978 octets**
- MD5 publiée : `06796e08731bccffaed6ae59361486e4`
- API archive : `https://zenodo.org/api/records/7544110/files/IDMT-SMT-GUITAR_V2.zip/content`.

**Contrôle effectué : métadonnées uniquement.** Les fichiers audio n'ont PAS été téléchargés ni analysés. Aucune annotation XML/MIDI, ni aucun vrai K, n'a été ouvert. Le hash ci-dessus a été constaté comme valeur publiée : il **n'est pas** encore un hash recalculé sur les 1,33 Go du ZIP.

## Protocole de validation à préserver

1. Récupérer et contrôler intégralement l'archive par MD5, sélectionner les WAV et leurs identités musicales sans lire les annotations ;
2. Restaurer les producteurs S1/S2/S3/S4/`repair_coherent` sur ces audios, en reconstituant leurs **vrais** tenseurs et non des colonnes substituées ;
3. Figer `S80-deploy-v1` si adaptations nécessaires (identité de modèle différente du S80 historique), sorties audio inédites, code, poids et paramètres, avec SHA-256 ;
4. Ensuite seulement ouvrir les annotations indépendantes et calculer Exact-K global, K0..K6, K2+ polyphonique, matrices corrections/régressions appariées et intervalles ;
5. Promotion seulement si la supériorité sur `freeze_local_combo` est indépendamment démontrée en global **et** polyphonique.

Les 57 événements GOAT précédents comportent **45 K0, 10 K1 et seulement 2 K2 prédits** par le S18 adapté (pas leurs étiquettes). Ils sont insuffisamment diversifiés pour tester S1–S4 en polyphonie.

**Statut au terme S177 :** 11/16 producteurs S45 adaptés ayant été exécutés sur nouvel audio ; S1/S2/S3/S4 historiques rejoués exactement sur 7 493 cas anciens ; quatre poids `repair_coherent` récupérés, mais aucune de ces cinq familles n'est encore exécutable en source complète sur nouveaux WAV. La référence officielle reste `freeze_local_combo`, S80 non promu.
