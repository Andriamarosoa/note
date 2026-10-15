# S592–S597 — Gain transféré, mais pertes dans le raccordement à V88

**Verdict du 15 octobre 2026 : S585 strict atteint 64,78 % contre 62,23 % pour S578 sur les mêmes cinq ensembles URMP. Basic Pitch direct atteint 73,54 %. L’objectif global de 80 % reste non atteint. Aucun modèle n’est promu.**

Le gain de la tête à quatre états se retrouve sur chacun des cinq extraits. Cependant, cette chaîne de notes, qui émet aux temps V88, reste moins bonne que son propre composant Basic Pitch utilisé directement. L’audit identifie un problème concret de conservation des attaques Basic Pitch, en plus du manque de rappel de la tête. Il ne suffit donc pas de publier le gain relatif S585/S578.

## Protocole exécuté

[Préenregistrement](README_V273_S592_S597_URMP_PREREGISTRATION.md) · [Manifeste des noms et empreintes](v273_s592_urmp_metadata_only.json) · [CI réussie](https://github.com/Andriamarosoa/note/actions/runs/38681794058) · [Preuves JSON](evidence/v273-s592-s597/manifest.json).

Un seul run, sans réentraînement : sources V84–V88 historiques et deux modèles appris sur les mêmes 28 sources MAESTRO. Les cinq enregistrements ont été sélectionnés par nom/empreinte, avant lecture des WAV ou des annotations. Les extraits sont fixés à **5–29 s**, soit **120 secondes** au total, avec dix types d’instruments. Les fichiers du miroir YourMT3/MIMuT sont **mono PCM16 à 16 kHz**, rééchantillonnés à 44,1 kHz ; ce n’est pas la distribution originale URMP à 48 kHz.

Les annotations **Notes_** donnent les notes réellement jouées. Les partitions MIDI **Sco_** ne sont pas utilisées. On compte **1 119 NoteOn vrais MIDI40–88**, après exclusion documentée de 4 notes hors plage et fusion de 7 unissons strictement identiques en hauteur et en temps. Les autres attaques proches restent distinctes. Les 1 358 sites V88 couvrent les cinq intervalles entiers, y compris K0.

Le job audio se termine à **04:33:49 UTC**, le gel des prédictions à **04:34:40 UTC**, et le job ouvrant les annotations commence à **04:34:43 UTC**. Les empreintes ont été revérifiées lors du téléchargement des artefacts. Le second calcul local reproduit les scores, et **60 contrôles indépendants** d’appariement maximal confirment les TP : 4 politiques × 5 sources × 3 tolérances.

## Résultat apparié à ±80 ms

Hauteur exacte, appariement un à un, tous les faux positifs et notes manquées inclus. Les modèles sont comparés sur la même vérité.

| Politique | TP | FP | FN | Précision % | Rappel % | F1 % |
|---|---:|---:|---:|---:|---:|---:|
| S578 Global28 | 575 | 154 | 544 | 78,88 | 51,39 | **62,23** |
| S585 strict 100 ms | 595 | 123 | 524 | 82,87 | 53,17 | **64,78** |
| S585 réattaque apprise | 596 | 129 | 523 | 82,21 | 53,26 | **64,64** |
| Basic Pitch direct | 735 | 145 | 384 | 83,52 | 65,68 | **73,54** |

**S585 strict vs S578 : +2,55 points de F1 ; 35 vraies notes récupérées, 15 perdues ; 48 FP retirés, 17 ajoutés, soit 31 FP de moins.** Les 106 FP communs restent visibles dans le registre individuel.

La variante réattaque apporte seulement 1 TP et 6 FP supplémentaires par rapport à S585 strict. Elle n’est pas retenue. Une seule paire vraie de réattaques de même hauteur sur la même piste est présente : trompette 2, MIDI63, intervalle 84 ms dans Slavonic. **Les deux notes de cette paire sont manquées par les quatre politiques.** Le TP supplémentaire de la variante concerne MIDI67 à 20,287 s ; il ne valide pas la paire rapide réelle.

## Par morceau et tolérance

| Morceau | Notes vraies | S578 F1 % | S585 strict F1 % | Basic Pitch direct F1 % |
|---|---:|---:|---:|---:|
| 03_Dance_fl_cl | 107 | 77,60 | 81,97 | 92,46 |
| 25_Pirates_vn_vn_va_sax | 233 | 57,51 | 58,97 | 75,40 |
| 26_King_vn_vn_va_vc | 256 | 69,83 | 72,00 | 76,06 |
| 31_Slavonic_tpt_tpt_hn_tbn | 318 | 57,26 | 59,47 | 65,77 |
| 35_Rondeau_vn_vn_va_db | 205 | 57,45 | 60,92 | 69,23 |

S585 strict progresse sur les cinq morceaux, mais Basic Pitch direct reste supérieur sur chacun. Ces observations sur cinq extraits ne constituent pas une garantie statistique universelle.

| Tolérance | S578 F1 % | S585 strict F1 % | Basic Pitch direct F1 % |
|---|---:|---:|---:|
| ±30 ms | 41,13 | 42,13 | 58,93 |
| ±50 ms | 55,30 | 56,83 | 68,93 |
| ±80 ms | 62,23 | 64,78 | 73,54 |

Les trois registres progressent également face à S578 :

| Registre MIDI | S578 F1 % | S585 strict F1 % | Basic Pitch direct F1 % |
|---|---:|---:|---:|
| Grave 40–59 | 70,11 | 71,27 | 81,19 |
| Médium 60–71 | 58,03 | 61,91 | 71,28 |
| Aigu 72–88 | 58,16 | 59,82 | 64,20 |

Les agrégats cordes/bois/cuivres/mixte, avec leurs TP/FP/FN, figurent dans [l’audit après évaluation](evidence/v273-s592-s597/postlabel_transfer_diagnosis.json).

## Pourquoi Basic Pitch direct fait mieux

Par identité de note vraie, **180 notes correctement détectées par Basic Pitch direct ne sont pas retrouvées dans S585 strict**. Réciproquement, S585 strict récupère 40 notes manquées par Basic Pitch direct. Le bilan est donc de **140 TP de moins** pour la chaîne S585.

Parmi ces 180 pertes, **176 n’ont aucun raccordement protégé de leur attaque Basic Pitch à un site V88**. Le code historique ne reprend une attaque BP que si son instant est situé entre **t_V88 − 0,2 ms et t_V88 + 20,2 ms**. La présence d’un site V88 à ±80 ms ne suffit pas. Pour les 4 autres pertes, un raccordement existe, mais le déplacement d’horodatage ou la concurrence d’appariement fait perdre la correspondance vraie. Les instants de chaque cas sont publiés dans l’audit.

Cela démontre que « conserver les notes BP aux sites V88 » ne conserve pas toutes les notes de la sortie BP directe. La tête ajoute des notes utiles, mais ne compense pas les attaques BP qui ne passent pas ce raccordement. Aucune union ni nouveau seuil n’est évalué ici comme résultat indépendant : ce serait une variante conçue après exposition de cette cohorte.

## Diagnostic des 524 notes manquées de S585 strict

| Catégorie observable | Notes |
|---|---:|
| Aucun site V88 à ±80 ms | 21 |
| Probabilité d’attaque <0,7 sur tous les sites voisins où la tête peut intervenir | 410 |
| Seulement des sites réservés aux naissances Basic Pitch | 48 |
| Décodeur ou concurrence d’appariement un à un | 45 |

Un oracle de hauteurs aux seuls temps V88 pourrait apparier **1 094 / 1 119 notes**, soit **97,77 %**. C’est une borne optimiste de rappel, pas un modèle exécuté : elle ignore les erreurs de hauteur, les contraintes du décodeur et les faux positifs. Les 25 notes hors de cette borne comprennent 21 absences de site voisin et 4 conflits d’appariement. Le manque de sites temporels ne suffit donc pas à expliquer le faible rappel réel de 53,17 %.

Les **15 vraies notes perdues par rapport à S578** ont toutes une probabilité S585 d’attaque inférieure à 0,7 au site qui permettait à S578 de les reconnaître :

| Morceau | MIDI | Temps vrai relatif (s) | P(attaque) S578 % | P(attaque) S585 % | P(maintien) S585 % |
|---|---:|---:|---:|---:|---:|
| Rondeau | 57 | 14.226 | 74,19 | 29,61 | 68,92 |
| Rondeau | 70 | 1.808 | 70,26 | 44,97 | 51,93 |
| Rondeau | 70 | 6.825 | 73,00 | 63,85 | 34,28 |
| Rondeau | 74 | 4.109 | 76,29 | 49,98 | 47,87 |
| Rondeau | 76 | 10.081 | 73,85 | 66,85 | 1,66 |
| Slavonic | 43 | 0.433 | 71,67 | 66,96 | 13,45 |
| Slavonic | 71 | 21.767 | 71,91 | 69,82 | 27,43 |
| Pirates | 57 | 18.513 | 74,12 | 12,14 | 86,49 |
| Pirates | 59 | 11.367 | 74,61 | 66,90 | 30,57 |
| Pirates | 59 | 16.896 | 72,94 | 54,26 | 43,15 |
| Pirates | 61 | 8.820 | 72,44 | 69,08 | 27,40 |
| Pirates | 71 | 18.365 | 91,67 | 61,14 | 37,59 |
| Pirates | 76 | 23.758 | 87,06 | 69,47 | 11,66 |
| King | 52 | 20.594 | 77,77 | 69,89 | 28,07 |
| King | 69 | 16.699 | 71,40 | 65,88 | 1,84 |

Ces probabilités décrivent les rejets ; elles ne prouvent pas qu’une baisse de seuil améliorerait le F1. Les comptes de FP et les validations extérieures restent nécessaires. Le registre complet contient aussi les 35 corrections et les faux événements ajoutés/retirés.

## Décision et prochaine correction à valider

**Conserver S585 strict comme tête expérimentale, sans promouvoir la chaîne complète.** Son amélioration face à S578 se transfère à cette cohorte, mais il faut désormais comparer systématiquement le système entier à Basic Pitch direct. Le prochain travail utile est de préserver les attaques BP à leur propre instant, puis d’ajouter les propositions apprises avec un dédoublonnage audité. Ce comportement devra être développé sur les sources déjà exposées et testé sur d’autres enregistrements encore fermés. Il n’est pas implémenté ni validé par S597.

Les scores **75,31 % sur S591 piano**, **64,78 % sur S597 URMP** et **59,34 % sur S584 piano** concernent des cohortes différentes. Ils ne forment pas une courbe de progression à corpus constant. Les cinq extraits URMP deviennent exposés après ce test.

Pas de compositeurs inédits revendiqués, pas de score d’identification instrumentale, pas de NoteOff, pas de latence mesurée. La conversion 16→44,1 kHz n’ajoute pas la bande fréquentielle perdue. Les débuts des sons de cordes/vents sont annotés humainement. **S80, Exact-K et le Live historique ne sont pas modifiés ; leur rejeu intégral n’a pas été effectué.**

## Reproduction et preuves

- [Implémentation S592–S597](../scripts/s592_s597_urmp_prospective.py) et [workflow](../.github/workflows/v273-s592-s597-urmp-frozen-transfer.yml).
- [Vérification indépendante hors réseau](../scripts/audit_s597_paired_transfer.py) : `PYTHONPATH=.:src python -m scripts.audit_s597_paired_transfer --root model --out audit.json` après restauration des artefacts.
- [Score complet](evidence/v273-s592-s597/independent_score_report.json), [prédictions figées](evidence/v273-s592-s597/frozen_note_predictions.json), [vérité après gel](evidence/v273-s592-s597/truth_after_freeze.json), [annotations des extraits](evidence/v273-s592-s597/annotation_events_after_freeze.json).
- [Registre des corrections et régressions](evidence/v273-s592-s597/individual_note_audit.json), [diagnostic de toutes les FN](evidence/v273-s592-s597/missed_note_diagnosis.json), [audit du raccordement BP et des réattaques](evidence/v273-s592-s597/postlabel_transfer_diagnosis.json).
- [Manifeste des artefacts, dates et empreintes](evidence/v273-s592-s597/manifest.json). Les caractéristiques 49×11 et probabilités 4 états sont conservées dans les artefacts CI ; aucun poids ni audio brut ajouté au dépôt.

SHA256 des notes figées : `63464cca3052c52e12525730813898d89dfa8b19633134378769f95b762fa6a2`.  
SHA256 S578 : `21b234796441fa5594d3a748d5b9ca42377e1181ec6cc2fe5bf68816c4c687af`.  
SHA256 S585 : `555cb092accb6ea0b42394280f646c90ccac0b48746040811ac57da382837d36`.
