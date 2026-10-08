# V27.3 — Protocole de correction du sélecteur

**Protocole fixé avant lecture des nouveaux résultats, 8 octobre 2026.**

La revue de conception a confirmé des défauts d'entrée et d'arbitrage. Un
rejeu changeant seulement la formule, à poids fixes, a aggravé le bilan.
Cette expérience réentraîne les mécanismes corrigés et conserve un contrôle
de reproduction. Elle n'ajoute aucun spécialiste acoustique.

## Trois bras prévus

| Bras | Entrées / audits | Décision et objectif |
|---|---|---|
| `legacy` | Version complète précédente, identité + régimes, 26 caractéristiques | Modèle et pertes précédents ; zéro divergence requis avec le run 37758789956. |
| `contracts` | 43 caractéristiques explicites, Cxy réellement présent, H0/action et métadonnées de repli documentées, producteurs des audits indépendants du récepteur | Ancien arbitre réentraîné avec ses pertes précédentes. |
| `coherent` | Exactement les mêmes entrées réparées que `contracts` | Distribution unique sur les sept vrais K, log-vraisemblance catégorielle non pondérée, choix par gain Exact-K attendu. |

Les trois bras utilisent 30 epochs, le seed 27402, Adam à 0,002, des lots
de 192 et une largeur neuronale de 32. Les nombres de paramètres sont
rapportés : les dimensions d'entrée et la tête de sortie évoluent, il ne
s'agit pas d'une égalité de capacité revendiquée.

`contracts` teste un ensemble de corrections d'entrée et de provenance.
Il n'attribue pas son résultat à une seule d'entre elles. La comparaison
`contracts`/`coherent` garde les mêmes entrées, mais modifie ensemble la
paramétrisation probabiliste et la fonction de perte.

## Contrats réparés

- Le contexte contient les 14 caractéristiques spectrales, 11 de naissance,
  11 d'amortissement et 7 de persistance. La troncature alphabétique disparaît.
- Le canal de présence Cxy indique le bit réel du sous-ensemble, comme le
  vecteur d'identité, et non la disponibilité générale de la transition.
- H0 fournit une action de repli déterministe, encodée par la classe initiale.
  Ce vecteur n'est pas annoncé comme la probabilité que cette classe soit
  correcte. La confiance originale du réseau gelé est absente des artifacts.
- Les 14 métadonnées distinguent cette disponibilité, le soutien des cinq
  spécialistes à la classe initiale, l'évidence des adaptateurs Cxy et les
  indicateurs de repli. Les anciennes constantes F_keep=0,98 cessent d'être
  présentées comme une confiance acoustique.
- Pour un audit reçu par un fold R, les modèles produisant les votes de
  référence excluent à la fois R, le test externe et leur propre fold prédit.
  Le scaler et le KMeans de cet audit excluent aussi R et le test externe.
  Les producteurs sont mémorisés par leurs folds de fit ; leurs IDs réels
  et les empreintes de ces ensembles sont exportés et contrôlés.
- La normalisation du contexte brut du réseau reste une normalisation
  sans labels sur le train externe, distincte des producteurs de régimes
  utilisés pour les audits. Les lissages des audits restent ceux du protocole
  précédent ; le changement porte sur les observations admises au fit.

## Décision cohérente

Le réseau conserve l'attention sur les 32/64 combinaisons légales. Il produit
une seule distribution `P(Y=K | observations)` sur K0–K6.

```text
probabilité de correction vers k = P(Y=k)
probabilité de régression         = P(Y=K_initial), commune aux destinations
probabilité que KEEP soit juste   = P(Y=K_initial)
avantage(k)                       = P(Y=k) − P(Y=K_initial)
avantage(KEEP)                    = 0
```

Les égalités conservent la classe initiale. Il n'y a plus de ratio de risques
comparé à des odds KEEP, ni de label appelant « KEEP correct » un événement
K0/K1 dont la classe initiale est fausse. Les vrais K0/K1 participent à la
perte catégorielle, mais restent des destinations de sortie interdites pour
cette comparaison : cette expérience n'ouvre pas le périmètre de correction.

La normalisation probabiliste et une perte propre n'établissent pas, à elles
seules, la calibration empirique. Le résultat est à mesurer après entraînement.

## Évaluation et arrêt

- Cohorte native immuable : 59 309 événements, 7 385 vrais poly ; 7 493
  événements admissibles selon la prédiction gelée K2/K3/K4.
- Référence : 48 454 exacts globaux et 2 530 exacts poly, soit 81,6976 % et
  34,2586 %. Les 1 918 erreurs poly hors périmètre ne sont pas traitées ici.
- Folds 0/1/2/4 séparés par enregistrement, **déjà exposés à la recherche**.
  Aucun player05/fold3 et aucun nouveau jeu indépendant.
- Sorties : Exact-K global, poly, K0–K6 ; corrections/régressions/neutres ;
  résultats par fold ; nombres de paramètres ; provenance ; poids entraînés
  par fold et prédictions alignées.
- Aucun seuil, epoch ou architecture ne sera ajusté aux résultats de ce run.
  Le contrôle doit reproduire l'archive avant interprétation comparative.
- Une perte nette ou l'absence de gain poly entraîne le rejet comme
  remplacement. Même un résultat positif reste expérimental : une
  confirmation sur un jeu neuf est nécessaire avant toute promotion.

## Vérifications

Les tests contrôlent la présence des quatre familles, les sous-ensembles,
la suppression de la fausse confiance H0, le rejet des IDs interdits,
l'invariance des audits du récepteur quand ses labels sont modifiés,
l'invariance des entrées de test quand ses labels sont modifiés, une
probabilité unique de régression, le contre-exemple multiclasses, les
gradients TensorFlow et un véritable entraînement court.

[Workflow comparatif](../.github/workflows/v273-selector-contract-repair.yml).
