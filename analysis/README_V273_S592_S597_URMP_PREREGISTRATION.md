# S592–S597 — Transfert figé vers cinq ensembles URMP

Protocole enregistré avant l'ouverture des WAV et des annotations des cinq
enregistrements. Aucun entraînement supplémentaire, aucune adaptation des
seuils et aucune promotion. Les résultats doivent être publiés même en cas
de régression. Cette expérience ne remplace pas une validation de S80 ni un
test Live.

## Question

S585 améliore S578 de 72,46 à 75,31 % de F1 NoteOn sur les deux extraits piano
S591, mais perd quatre attaques d'Enescu. Est-ce que son gain se retrouve sur
des mélanges de cordes et de vents ? Le score S591 reste un résultat sur deux
extraits de 12 secondes. Il n'est pas une mesure générale de performance.

Les modèles restent ceux entraînés sur les mêmes 28 sources MAESTRO. Le test
ci-dessous n'entraîne pas une nouvelle tête pour réparer les quatre notes
désormais exposées. Il mesure le transfert des têtes existantes.

## Sources et sélection sans contenu musical

Source scientifique : [URMP, Université de Rochester](https://labsites.rochester.edu/air/projects/URMP.html)
et [documentation des annotations](https://labsites.rochester.edu/air/projects/URMP/URMP_doc.pdf).
L'archive préparée [YourMT3 sur Zenodo](https://zenodo.org/records/10009959)
propose URMP à **16 kHz**. Nous utilisons les fichiers individuels du miroir
[MIMuT Data v2](https://huggingface.co/datasets/J1mmymm/MIMuT_Data_v2/tree/3a62b3a1c8bb263a6a6ac37559070339ddb81080/data/urmp/urmp_yourmt3_16k),
à la révision immuable `3a62b3a1c8bb263a6a6ac37559070339ddb81080`.
Le manifeste conserve les SHA256 LFS des WAV et SHA1 Git des fichiers Notes_
issus de la seule liste des fichiers. Les octets doivent correspondre à ces
empreintes avant toute utilisation. Ce miroir n'est pas présenté comme la
distribution originale 48 kHz : la bande audio disponible est différente.

Les 44 noms AuMix_ terminant exactement par `.wav` sont classés selon les
instruments de leur nom. La sélection suit l'ordre de catégories cordes,
bois, cuivres, mixte, cordes ; dans chaque catégorie, on choisit le plus petit
SHA256 du chemin audio dont le titre abrégé n'a pas déjà été sélectionné.
Fichiers auxiliaires `.reapeaks` exclus. Aucun morceau de remplacement après
écoute ou évaluation.

| Enregistrement | Groupe | Instruments |
|---|---|---|
| 35 Rondeau | Cordes | 2 violons, alto, contrebasse |
| 03 Dance | Bois | Flûte, clarinette |
| 31 Slavonic | Cuivres | 2 trompettes, cor, trombone |
| 25 Pirates | Mixte | 2 violons, alto, saxophone |
| 26 King | Cordes | 2 violons, alto, violoncelle |

**Extrait fixe : de 5,000 à 29,000 secondes**, soit 24 secondes par source,
120 secondes et dix types d'instruments. Aucun compositeur inédit n'est
revendiqué : des œuvres/compositeurs peuvent recouper le répertoire piano.
La recherche dans les scripts, README et analyses accessibles de cette
branche n'a trouvé aucune utilisation URMP antérieure. Ce constat ne prouve
pas l'absence d'exposition dans toutes les branches ou tous les préentraînements.
URMP n'apparaît pas dans les jeux d'entraînement énumérés par l'article
[Basic Pitch, tableau 1](https://arxiv.org/pdf/2203.09893), mais aucun audit
exhaustif des corpus externes du checkpoint n'est revendiqué.

## Inférence avant annotations

1. S592 publie le manifeste des noms/empreintes, sans WAV ni Notes_.
2. S593 convertit les WAV mono PCM16 à 16 kHz vers 44,1 kHz par le rééchantillonnage
   polyphasé habituel, puis exécute les sources originales V84–V88 et le
   `freeze_local_combo` historique. Tous les sites K0–K6 sont conservés. La
   tête auxiliaire S466, inutilisée par ces décodeurs, n'est pas exécutée.
3. S594 exécute Basic Pitch 0.4.0 avec les mêmes paramètres : attaque 0,5,
   activité 0,3, durée minimale 127,7 ms. S595 conserve les 49×11 descripteurs
   S515 et les entrées temporelles 52D S569 sans adaptation.
4. S596 recharge les poids S578 et S585 déjà produits. Le SHA S585 doit être
   `555cb092accb6ea0b42394280f646c90ccac0b48746040811ac57da382837d36`.
   Les probabilités et notes sont téléversées en artefact avant S597.
5. S597, dans un job distinct, vérifie toutes les empreintes et ouvre les
   annotations des notes jouées pour la première fois.

Politiques préfixées : `S578_global28`, `S585_state_strict100`,
`S585_state_restrike`, et le témoin diagnostique `BasicPitch_direct`.
Le seuil d'attaque reste 0,7 ; la déduplication reste 100 ms et le veto
harmonique 0,8. La variante réattaque déjà rejetée en S591 est publiée à titre
de contrôle, sans sélection rétrospective. Basic Pitch direct conserve ses
propres horodatages ; il sert à évaluer les pertes dues aux propositions V88.

## Définition du score et audits

La vérité vient des fichiers **Notes_** : temps de début en secondes,
fréquence en Hz, durée en secondes. Le MIDI **Sco_** décrit la partition et
n'est jamais utilisé pour aligner le jeu réel. La hauteur est arrondie à la
note MIDI la plus proche selon `69 + 12 log2(Hz / 440)`. Le décalage de
l'extrait est exactement 5 secondes, sans réalignement appris sur la cible.

Score principal : F1 micro NoteOn, MIDI **40 à 88**, appariement de même
hauteur un à un à ±80 ms ; sensibilités préfixées à ±30 et ±50 ms.
Toutes les prédictions et toutes les notes vraies admissibles comptent, y
compris sur une source sans prédiction. Les unissons entre pistes ne sont
fusionnés que si hauteur **et temps exact** sont identiques. Les attaques
proches restent distinctes ; aucune déduplication temporelle de la vérité.
Le nombre de notes hors plage et d'unissons fusionnés est publié. Les notes
déjà actives à la coupure n'ont pas de véritable NoteOn dans l'extrait ; une
fausse attaque générée à la coupure compte comme FP.

Audits obligatoires : TP/FP/FN par morceau, registre et tolérance, notes
individuelles récupérées/perdues et FP retirés/ajoutés, toutes les notes
manquées de S585 strict, maximum de TP accessible aux temps V88 figés, et
réattaques de même hauteur **sur la même piste instrumentale** sous 100 ms.
La catégorie « probabilité insuffisante » est un diagnostic après labels,
pas une invitation à modifier le seuil puis à appeler le score indépendant.

Limites : cinq extraits, audio préparé 16 kHz, annotations humaines de débuts
parfois progressifs, plage 40–88, aucune reconnaissance du nom d'instrument,
aucun score NoteOff, aucun benchmark de latence. L'évaluation des hauteurs
utilise les mêmes entrées audio dans les deux têtes ; elle ne valide pas le
Live causal ni l'Exact-K du pipeline S80 complet. Les fichiers historiques
restent inchangés. Les nouveaux morceaux deviennent exposés après S597.
