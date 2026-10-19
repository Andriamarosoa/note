# S756 — Ajout de notes polyphoniques sans décoder à nouveau S708

Date : 19 octobre 2026. Expérience strictement non promue.

## Résultats à l'origine

S755 (https://github.com/Andriamarosoa/note/actions/runs/39213327892)
a terminé 12/12 boucles et 36 tests unitaires, mais n'a produit
**aucun candidat non régressif**. La meilleure Active-K était
56,6523 % contre 56,2762 % pour S708 ; elle faisait baisser Offset
de 77,3292 % à 77,1791 % et perdait 8 anciennes fins correctes.
Les variantes améliorent souvent la cardinalité, mais peuvent modifier
des Notes originales lors du redécodage.

S753 (https://github.com/Andriamarosoa/note/actions/runs/39209836870)
a montré que la tête K6 n'est reconnue sur aucune des 189 décisions
CAL32 de cardinalité K6 dans S708, tandis que le décodeur MIDI
sous-compte aussi avec un K théoriquement suffisant.

## Architecture S756

**Ne pas greffer `midi` ; conserver les spécialistes NOTE.**

- Récupérer les quatre modèles K S755 réellement appris sur FIT80,
  vérifier leur signature S708 et la forme 240D. **Aucun nouvel
  entraînement sur CAL**.
- Pour chaque piste CAL32, calculer le flux S708 original, puis douze
  flux de propositions en lecture seule avec K du spécialiste S755
  interpolé à 55 %. Il s'agit de propositions, pas de substituts.
- Les doublons de notes présentes dans le S708 source sont filtrés par
  les seuls événements natifs disponibles **jusqu'à l'attaque**. On
  bloque aussi les nouvelles attaques de même hauteur que la source
  dans les 85 ms suivant une ancienne attaque native.
- L'admission utilise seulement l'Onset, le Sustain, la masse K
  supplémentaire et l'occupation native au fragment ; la raison, le
  moment et la longueur du **futur NoteOff** ne sont jamais utilisés
  pour accepter une attaque. Trois seuils expérimentaux 0,40 / 0,55 /
  0,70 × quatre spécialistes = **douze tests complets**.
- Le mux S746 est responsable de toutes les attaques supplémentaires
  et des fermetures : les événements S708 originaux sont recopiés dans
  l'ordre, avec les mêmes métadonnées. Si un futur NoteOn natif
  réclame une hauteur ou une place, il préempte une voix
  supplémentaire. **Le futur natif n'est pas pré-consulté pour
  accepter une attaque supplémentaire à t.**

### Limites à ne pas cacher

1. La confiance `0.35*onset + 0.30*sustain + 0.35*K_tail` est un
   **score causal exploratoire**, pas une probabilité calibrée d'une
   instance note complète. Aucun nouveau modèle appris.
2. Préserver les événements originaux S708 n'implique **pas**
   automatiquement la conservation des six scores : le matching
   des véritables notes change avec les attaques supplémentaires.
   La promotion exige explicitement *zéro ancien Offset correct
   perdu*, *zéro vrai long Onset perdu*, *pas de perte de TP Onset*,
   *zéro baisse de chacune des six métriques*.
3. `CAL32` est la sélection expérimentale, pas HELD ou TEST ;
   le score de tête K seul ne remplace jamais la métrique Exact-K
   décodée.
4. Mesure de latence Windows/iRig **non effectuée** ; il est
   interdit de brancher S756 dans le LIVE par défaut sur la base
   d'un résultat de recherche.

## Plan de mesure

- Scripts : `scripts/s756_source_protected_k_birth_mux.py`.
- Tests : `test/test_s756_source_protected_mux.py`.
- CI : https://github.com/Andriamarosoa/note/actions/runs/39215745309
- PR DRAFT : https://github.com/Andriamarosoa/note/pull/87

La référence officielle S80 et la référence signée de comparaison
S708 restent distinctes et ne sont pas touchées. HELD/TEST fermés,
aucune distillation étudiant avant un véritable dépassement des
six métriques au-dessus de 80 %.

**Statut scientifique** : seuls les tests unitaires de signature,
causalité et préemption peuvent être déclarés réussis avant la
lecture du rapport GitHub S756 ; aucune nouvelle valeur F1 validée
n'est écrite dans ce document.
