# Protocole — support harmonique exclusif de l'attaque

Date : 6 octobre 2026.

## Question

Les composantes F0 non appariées que le reconstructeur choisit à la place des
notes attendues gagnent-elles surtout parce qu'elles réutilisent des partiels
déjà expliqués par d'autres notes ?

Cette étape est diagnostique uniquement. Elle ne modifie aucune prédiction
Exact-K, aucun seuil et aucun réseau.

## Population

- cohorte acoustique figée : 488 cas ;
- K3_regressed=125, K3_preserved=147, K2_corrected=108, K2_missed=108 ;
- folds 0, 1, 2 et 4 uniquement ;
- fold 3 interdit ;
- chemin audio normal uniquement.

## Spectre d'attaque

Même PRE et POST1 que les audits précédents :

- PRE   : [s-2048,s)
- POST1 : [s,s+2048)

Puissance d'attaque :
`A = max(P_POST1 - P_PRE, 0)`.

Même Hann, FFT 8192, bande d'analyse, MAX_HARMONICS et KERNEL_HZ que le
reconstructeur existant.

## Masques harmoniques

Pour une F0 f :

`H(f) = union des bins autour de h*f, h=1..MAX_HARMONICS`.

Pour une composante f dans un ensemble S :

- support total : énergie A dans H(f) ;
- support partagé : énergie A dans H(f) intersect union(H(g), g dans S\{f}) ;
- support exclusif : énergie A dans H(f) moins cette union ;
- fraction exclusive : exclusif / total.

Deux diagnostics sont calculés.

### 1. Cohérence interne du jeu

- pour une note attendue : unicité par rapport aux autres notes attendues ;
- pour une composante du triplet choisi : unicité par rapport aux deux autres
  composantes choisies.

### 2. Chevauchement avec la vérité annotée

Pour chaque composante choisie non appariée :

- fraction de son support d'attaque qui tombe dans l'union des gabarits des
  notes attendues ;
- fraction située hors de cette union.

Les annotations restent strictement diagnostiques.

## Comparaisons

Priorité :

1. K3 : expected vs selected_unmatched ;
2. K3_regressed vs K3_preserved ;
3. K2_corrected vs K2_missed ;
4. cohérence du sens sur folds 0/1/2/4.

Mesures principales :

- unique_attack_fraction ;
- shared_attack_fraction ;
- attack_energy_total ;
- selected_unmatched_overlap_with_expected_fraction ;
- selected_unmatched_outside_expected_fraction.

## Condition pour autoriser un bras d'inférence ultérieur

Un correcteur de classement basé sur l'unicité ne sera testé que si :

- les vraies notes ont systématiquement plus de support exclusif que les
  composantes non appariées, ou
- les composantes non appariées tirent systématiquement une grande part de leur
  énergie d'attaque des mêmes bins que les notes attendues,
- avec le même sens dans plusieurs folds et sans dépendre de quelques extrêmes.

Sinon, la piste "support harmonique exclusif" est rejetée.

## Sorties

- cases.jsonl : mesures par composante ;
- report.json : agrégats globaux, groupes et folds ;
- report.md : résumé ;
- aucune prédiction modifiée ;
- aucun gain Exact-K annoncé ;
- empreintes des sources et du script.

## Limites

Les bandes harmoniques peuvent se recouvrir physiquement entre vraies sources.
Une fraction exclusive faible ne signifie pas qu'une note n'existe pas ; elle
mesure seulement l'identifiabilité spectrale sous les gabarits actuels.
