# S757 — Audit expérimental des collisions d'appariement Onset/Offset

Statut : expérience CAL32 de recherche ; aucune promotion automatique.

## S756, mesures signées terminées

S756 GitHub Actions SUCCESS :
https://github.com/Andriamarosoa/note/actions/runs/39215745309

- 36 tests unitaires réussis et 12/12 variantes GuitarSet CAL32 terminées.
- **0/12 variantes admissibles** : le préservateur de flux MIDI
  reproduit les événements originaux S708, mais les six scores peuvent
  encore régresser.
- Meilleure Active-K S756 `depth5_rare_risk40` : Onset 74,2959 %,
  Exact-K 62,2831 %, Active Exact-K 56,5975 %, Offset 77,1635 %,
  Full Note 57,0160 %, Frame Pitch 76,6315 % ;
  37 nouvelles attaques acceptées, 12 préemptées plus tard et
  **3 anciennes références Offset correctes non maintenues**.
- La métrique S704 s'appuie sur `event_score`, qui recalcule
  un appariement global de coût (Hungarian) pour les attaques
  de même hauteur dans la fenêtre ±80 ms. Ajouter un NoteOn peut donc
  déplacer l'appariement à un événement préexistant même si le S708
  signé est byte-to-byte inchangé.
- **Hypothèse à falsifier** : une partie des pertes d'Offset est
  due aux compétitions de NoteOn du même pitch, et non uniquement
  au mauvais relâchement d'une voix supplémentaire. Aucun audit de
  paires source/référence n'a encore confirmé le mécanisme
  pour chacune des trois pertes.

## S757 — variable isolée

Quatre modèles FIT80 S755 conservés, S708/S707 intégralement figés,
même gate d'Onset supplémentaire = 0,55 (classe non calibrée).
**Une seule dimension varie** : recul minimum depuis le dernier
NoteOn signé de la même hauteur (85 / 160 / 240 ms).
Quatre spécialistes × trois fenêtres = **12 flux MIDI réels**
évalués sur CAL32.

Aucune donnée du futur ni étiquette n'est passée aux sélecteurs,
le NoteOff candidat ne sert jamais à calculer la probabilité
d'admission à son NoteOn.

Contrôles : 6 métriques S704 sur même split CAL32 et même MIDI,
zéro ancien vrai Offset correct perdu,
zéro ancien Onset long correct perdu,
zéro ancien TP Onset perdu. Pas de TEST, HELD ni LIVE,
pas de promotion S80/S708/S707, pas de distillation.

Les tests synthétiques montrent :
- une attaque supplémentaire 139 ms après une attaque native
  est admise à 85 ms mais refusée à 160 ms ;
- une attaque ~232 ms après une attaque native est admise
  à 160 ms mais refusée à 240 ms ;
- l'arrivée future d'un NoteOn natif ne change pas
  la décision d'admission actuelle ;
- un futur NoteOff supplémentaire ne change pas non plus
  sa probabilité d'admission.

Le premier run CI S757 contenait un scénario synthétique
incohérent (posterior native K1=100 % après la fermeture de la
seule note, supprimant naturellement tout K-tail novelty). Le
scénario a été corrigé pour native K0 après relâchement.
Ne pas considérer l'échec du premier test comme une validation
sur de vrais enregistrements.

Run CAL32 S757 :
https://github.com/Andriamarosoa/note/actions/runs/39216827575
PR DRAFT :
https://github.com/Andriamarosoa/note/pull/88

Aucun nouveau F1 S757 n'est revendiqué avant lecture du
rapport complet `report.json` de ce run.

## Résultat réel S757 — exécution terminée

GitHub Actions `39216827575` : **SUCCESS**, 42 tests, 12/12
boucles CAL32 ; référence S708 parité exacte et 0 candidat sûr.
L'essai précédent `39216651039` a échoué seulement à cause d'une
fixture synthétique K0/K1 contradictoire ; corrigé avant le run validé.

Point crucial : **les trois gardes 85 / 160 / 240 ms donnent des
métriques et des nombres de propositions acceptées identiques pour
chaque famille** sur CAL32. L'hypothèse que les attaques
supplémentaires se présentaient entre ces trois seuils, causant la
régression, **n'est pas étayée par cette expérience**. Cela n'exclut
pas d'autres collisions Onset ni des offsets supplémentaires erronés.

Meilleure Active-K S757 `depth5_rare_guard85ms` (mêmes résultats
pour `guard160ms` et `guard240ms`) :
- Onset 74,3176336 %, Exact-K 62,2545604 %,
  Active-K 56,5575651 %, Offset 77,1556166 %,
  Full Note 57,0267868 %, Frame Pitch 76,6179303 %.
- 32 notes supplémentaires acceptées ; 12 préemptions.
- 3 anciennes fins de notes correctes non maintenues,
  zéro ancien long Onset perdu.
- 0/12 variante safe selon les six scores et les correspondances
  de notes précédemment correctes.

**Verdict P1 : FAIL / non-promotion.** L'augmentation du
refractory de même hauteur n'est pas un moyen utile de corriger
les pertes d'Offset dans le corpus CAL32. Prochaine expérience
requise : audit *des trois paires concrètes* source/truth perdues,
ainsi que des candidates ajoutées appariées aux mêmes vérités,
en différenciant réassignation Hungarian, notes supplémentaires
avec relâchement incorrect, et préemption par source.
