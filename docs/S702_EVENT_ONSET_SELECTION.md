# S702 — sélection causale des attaques et état de réarmement

## Problème reproduit

S701 augmente des probabilités Onset sous seuil avant de réutiliser le même
sélecteur. Cela ne préserve pas ses événements : un ajout peut désarmer une
hauteur avant une attaque native, ou occuper une des six places de polyphonie.
Le test synthétique montre un ajout à 63,855 ms qui fait disparaître l'attaque
native à 208,980 ms. Ce cas prouve un risque logiciel, pas sa fréquence réelle.

## Correction et expérience

S702 entretient deux sélecteurs indépendants, l'un alimenté par les probabilités
natives, l'autre par les propositions S701. Une sélection de sortie fusionne
leurs candidats au moment de leur émission, sans antidater les NoteOn.

- `protected_S702` garde tous les NoteOn natifs. Un ajout ne peut pas couper
  une note native ; une note ajoutée cède sa place si le budget de six notes
  empêcherait l'émission d'un NoteOn natif.
- La tête expérimentale de qualité apprend sur des **événements candidats**,
  avec un appariement un à un aux annotations TRAIN, et non sur toutes les
  trames positives d'une fenêtre d'attaque.
- Ses 49 caractéristiques utilisent les 39 variables causales S701, le score
  S701, l'origine du candidat, les notes natives actives et les temps écoulés
  depuis les candidats/attaques précédents.
- Les seuils d'acceptation peuvent rejeter des propositions natives fausses et
  ajouter des attaques manquées. Cette variante n'offre pas une garantie de
  conservation de tous les NoteOn natifs : les pertes sont mesurées par note.

La politique neutre reproduit exactement les événements natifs. La causalité
est contrôlée en comparant une séquence complète avec son préfixe. Les tests
couvrent aussi le réarmement, les six places et les pertes cachées par un gain
net de vrais positifs.

## Protocole

Base : commit `1fdd97fcba80fb22883e99791e34ce7771c2a1af`.

| Étape | Prises GuitarSet | Usage |
|---|---:|---|
| FIT | 64 | Quatre familles de compositions TRAIN ; apprentissage de la tête d'événements |
| CAL | 16 | Cinquième famille TRAIN ; choix d'une seule politique avant HELD/DEV |
| HELD | 64 | Quatre autres familles ; évaluation du candidat figé |
| DEV | 12 | Diagnostic historique ; contrôle de reproduction 998 TP / 585 FP / 582 FN |
| TEST | 0 | Non ouvert |

S701 est réutilisé avec son apprentissage antérieur sur cinq familles TRAIN.
CAL est donc une calibration de la tête d'événements, pas une validation
indépendante de tous les composants. HELD/DEV ont déjà été examinés dans les
recherches antérieures : ils ne constituent pas un test final inédit.

## Mesures et limites

- Onset avec hauteur correcte à ±80 ms, précision, rappel et détail accords/solo.
- Références réellement gagnées/perdues, événements ajoutés/supprimés.
- Exact-K des intervalles décodés avec la fin native, comme dans S701.
- S675 figé : Offset à ±50 ms, F1 note complète et notes longues.
- Contrôle des fins sur les mêmes événements appariés, pour distinguer une
  régression d'une variation du dénominateur de l'Offset conditionnel.
- Fichiers JSONL des événements pour permettre de refaire l'appariement.

L'Offset S675 reste évalué par replay shadow séparé ; ces mesures ne constituent
pas une certification du streaming microphone ni une mesure commune de toutes
les meilleures têtes historiques. La référence LIVE officielle reste inchangée.

## Exécution

Workflow : `note-s702-event-onset-selection.yml`.
Run : https://github.com/Andriamarosoa/note/actions/runs/39089441825
Artefact attendu : `S702_CAUSAL_EVENT_ONSET_STATE_AND_QUALITY`.

```powershell
python -m unittest -v test.test_s702_event_selection test.test_s701_causal_onset
```

## Résultats vérifiés — run terminé avec succès

Durée du calcul : 942,01 s. Onze tests réussis sur le runner. Les TP/FP/FN
HELD et DEV ont été recomptés à partir des fichiers d'événements JSONL.

| Mesure | Natif | Réarmement protégé | Tête commune S702 |
|---|---:|---:|---:|
| F1 Onset DEV | 63,1046 % | 63,1080 % | 62,1711 % |
| TP / FP / FN DEV | 998 / 585 / 582 | 999 / 587 / 581 | 945 / 515 / 635 |
| Exact-K décodé DEV | 54,4721 % | 54,5191 % | 54,2067 % |
| Offset conditionnel S675 DEV | 78,0808 % | 78,0020 % | 77,2921 % |
| F1 note complète DEV | 48,8776 % | 48,8313 % | 47,6974 % |
| F1 Onset HELD | 67,5894 % | 67,6141 % | 67,1912 % |

La tête commune récupère 10 véritables attaques mais en perd 63 sur DEV
(30 graves MIDI 40–51 et 33 médiums MIDI 52–71). Sur HELD elle en récupère
29 et en perd 449, dont 26 notes longues. La baisse du nombre de faux positifs
ne compense pas ces pertes. La sélection CAL avait déjà retenu `native` ; le
candidat libre `event_native_0.2_rescue_0.65` a été évalué comme exploratoire.

La protection de l'état natif ne supprime aucun événement natif. Cependant,
S701 ne perdait aucune référence vraie sur ces HELD/DEV : le défaut synthétique
ne prouve donc pas la cause du plateau réel. Préserver aveuglément tous les
événements natifs garde aussi des faux positifs.

**Décision : aucune promotion.** Le correctif reste expérimental ; le LIVE et
ses checkpoints sont inchangés. Les résultats complets sont conservés dans
l'artefact du run et la synthèse vérifiée dans
`analysis/evidence/s702-event-onset-selection/verified_summary.json`.

La boucle suivante S703 sépare les cibles natives des cibles résiduelles :
un candidat mieux placé ne doit pas transformer une attaque native correcte
en exemple négatif d'apprentissage. Elle conserve aussi l'horloge rythmique
native causale pour les entrées S675 communes.
