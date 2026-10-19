---
serie: S792
statut: en_cours
type: experience
reference: S708
branche: codex/note-s792-framepitch-six-metric-midi-replay
run_id: en_attente
commit_sha: en_attente
promotion: false
---

# S792 — Résumé

## Objectif

Valider le candidat Frame Pitch S790 dans **un véritable flux causal d'événements MIDI NoteOn/NoteOff** et recomputer les six métriques officielles sur un même flux, à la cadence de base 256 échantillons.

## Hypothèse

Le gain SHADOW S790 85,12 % peut être partiellement transféré en prolongeant avec prudence certains NoteOff S708 jugés prématurés à partir des **preuves de Sustain déjà disponibles**. Cette hypothèse peut échouer : la tête SHADOW détecte aussi des notes jamais émises, qu'un simple ajustement de NoteOff ne peut récupérer.

## Modifications

- Rejeu causal par événements, **aucun oracle** : toujours conserver les NoteOn source et ne prolonger un NoteOff qu'après une nouvelle preuve S790 ; durée prolongée bornée.
- Vérification stricte du délai, des notes actives, des retriggers et de la cardinalité maximale six.
- Sélection de la durée maximale de rétention sur FIT80 uniquement, validation hors-composition FIT, CAL32 verrouillé.
- Validation de **parité avec le S708 signé sur les six métriques** avant de considérer les nouveaux résultats comme comparables.
- Mesure de six métriques sur les événements produits et rapport par composition/texture, y compris tous les rejets.
- Six audits indépendants + consensus, fichiers resume.md et resultat.md.

## Protocole

Sources S708 signé #39094682419 et S707 immuable ; GuitarSet FIT80 pour apprentissage et sélection, CAL32 uniquement pour évaluer la politique figée, sans lecture de HELD/TEST. La tête S790 est réentraînée à partir de FIT80 selon le code gelé, et la politique de tête est celle gelée `cut0.45_harm0.00`, sélectionnée auparavant uniquement sur FIT.

Les durées S792 candidates sont 0, 1, 2 et 4 fenêtres de 1024 échantillons ; ne sont conservées que les politiques qui respectent la non-régression à chaque composition FIT. Les mesures officielles utilisent **les mêmes intervalles MIDI** reconstruits sur la grille 256, et non les probabilités 1024.

Un score SHADOW supérieur à 80 % n'est pas une preuve de progrès MIDI ; la promotion n'est possible qu'après l'audit des six métriques intégrées.

## Audits obligatoires

- Données : SHA, split, compositions FIT/CAL, nombre d'enregistrements, vérité terrain après inférence.
- Acoustique : Sustain, harmonique et incertitude réelle des NoteOff prématurés.
- Architecture : événement réel, états MIDI isolés et causalité du retard.
- Apprentissage : 5 compositions OOF, aucune recherche sur CAL, protocole S790 préfigé.
- Métriques : parité six scores S708, recalcul des comptes 256, régressions croisées.
- LIVE : émetteur événementiel sans rétrodater, pas de modification du décodeur utilisé par le microphone.

## Statut

En cours. Aucun nouveau score MIDI validé. Aucune promotion.
