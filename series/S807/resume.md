---
serie: S807
statut: termine
type: experience
reference: S708
branche: codex/note-s807-blind-activek-three-model-review
run_id: 39378681127
commit_sha: 07aa63dbad13dba4922a16b22da6b2b920a06061
promotion: false
---

# S807 — consultation réellement indépendante et critique Active Exact-K

## Objectif

Obtenir trois diagnostics indépendants pour comprendre les échecs Active Exact-K du vrai MIDI émis et concevoir des expériences causales non régressives, sans demander aux modèles de confirmer l'explication précédemment avancée. Aucun modèle ne peut décider d'une promotion.

## Hypothèse

Les causes fondamentales d'une mauvaise cardinalité active doivent se mesurer et non se supposer. Les trois modèles doivent proposer plusieurs mécanismes concurrents, éventuellement contradictoires. Toute solution n'est recevable que si elle réussit une falsification sur des traces authentiques puis les six métriques exactes, sans aucun ajout de latence.

## Modifications

Créer un bundle signé FIT80 avec **uniquement** les agrégats actifs de S794 et les lignes numérotées des chemins critiques. Trois appels fournisseur NVIDIA NIM distincts : `google/gemma-4-31b-it`, `nvidia/nemotron-3.5-lightning-30b-a3b`, `poolside/laguna-xs-2.1`. Utiliser le **même prompt, même contenu de preuve, même température, mêmes paramètres de sortie** pour les trois. Aucune mention des hypothèses de l'assistant, aucune réponse d'un autre reviewer, aucun choix d'architecture favorisé, pas de recommandation préremplie.

Les réponses restent conservées intégralement, avec empreintes prompt/contexte/réponse, identification réelle de l'endpoint, qualité (volume, citations de deux fichiers), avertissement d'échec du fournisseur. Le validateur refuse d'appeler un modèle « réussi scientifiquement » sur la seule base de la réception d'un texte. Consultation une fois, pas de réessais automatiques multiples.

## Protocole

Source brute S794 run 39336923230, 80 pistes FIT, 249 411 trames actives, 142 213 correctes, 80 654 sous-comptées, 26 544 surcomptées, six scores du même corpus. La distribution K0 à K6, les quatre indices acoustiques avec vrais/fausses hauteurs et la décomposition par composition sont transmis à l'identique. Ne pas partager les vrais enregistrements, les étiquettes individuelles, CAL32, HELD, TEST, les secrets, les anciens avis NVIDIA ou le texte des suggestions précédentes. Les extraits numérotés n'impliquent pas une lecture exhaustive du dépôt.

Tous les modèles doivent rendre au moins trois hypothèses opposables, des contre-exemples, tests de réfutation, les effets sur six métriques et sur latence. Après les trois retours, contre-vérifier leurs citations, les comptes et leur argumentation dans le code exact. Rejeter les explications incohérentes. Préparer séparément les tests sur GuitarSet FIT80, sans entraîner sur la composition évaluée. Aucune promotion CAL32, HELD/TEST ou LIVE par consultation.

## Audits obligatoires

Contrôle de parité SHA de la preuve identique entre trois appels ; contrôle des trois identités de modèles réellement différents ; test des citations source et des textes non triviaux. Six angles locaux d'audit : données, acoustique, architecture, apprentissage, métriques, LIVE. Ce protocole n'affirme pas déclencher les six agents personnalisés du mode Work. Les trois avis externes sont des hypothèses et ne constituent jamais la preuve d'un gain. Une critique indépendante reste obligatoire.

## Statut

Préparation de l'expérience, résultats de modèles inconnus jusqu'à lecture des artefacts réels. S708 strictement conservée. Aucun entraînement ou retiming LIVE.

## Conclusion après deux consultations réelles et contre-audit

Le premier run de modèles [#39377951484](https://github.com/Andriamarosoa/note/actions/runs/39377951484) termine et signale : Gemma texte non suffisamment ancré (1 chemin cité exactement), Nemotron contenu final vide malgré 3300 tokens, Poolside HTTP503. Une unique tentative corrective [#39378681127](https://github.com/Andriamarosoa/note/actions/runs/39378681127) désactive le long raisonnement Nemotron, remplace le fournisseur 503 par `meta/llama-3.3-70b-instruct` et conserve EXACTEMENT le même prompt SHA et le même contexte numérique et source SHA. Nemotron rend 14 847 caractères, trois chemins valides ; Gemma rend un long texte mais zéro chemin exact vérifiable ; Llama répond HTTP410. Triage textuel 1/3, **validation scientifique 0/3** puisque Nemotron présente des fausses références et interprétations. Pas de consensus externe certifié, aucun modèle promu. Les deux runs ont passé leur smoke et leur collecte traçable.

Contre-audit : les hallucinations communes confondent le `ActiveKDecoder` S763 (shadow, non déployé) avec S708, attribuent à S797 (recherche, non promu) une régression de S708 et confondent K posterior `argmax` avec active-K calculé sur les événements MIDI effectivement émis. Le lissage gaussien de S741 n'a pas de biais directionnel intrinsèque sur l'argmax d'une source isolée. Déplacer la vérité ou modifier le dénominateur des frames n'est pas un gain musical ni un test admissible. Les modèles ont néanmoins raison sur la **statistique descriptive** : 80 654 sous-comptages sur 107 198 erreurs actives FIT80.

Analyse causale plus poussée sur vrais événements : le script `scripts/s807_midi_gap_diagnostic.py` analyse chaque trame 256 et chaque hauteur sur les **80 JSON S800 FIT** en reproduisant strictement les compteurs signés S708/S794, sans ouverture de CAL/HELD/TEST. Il partitionne 146 984 trames-pitch manquées : 50 442 notes-positions d'un segment totalement absent, 48 957 débuts non couverts, 29 525 fins non couvertes, 18 060 interruptions au milieu. Ces catégories sont des observations de la timeline, PAS des causes acoustiques déjà prouvées. 835 trames contiennent des unisons/hauteurs annotées en parallèle, dont 758 sous-comptées : à inspecter séparément sans gonfler artificiellement le plafond.

Statut **TERMINÉ comme audit scientifique**, pas comme amélioration. Le contrôle six métriques S708 demeure obligatoire pour toute S808. Aucun entraînement, aucun changement LIVE, aucune mesure de latence CPU/iRig ou gain Onset/ActiveK annoncé.
