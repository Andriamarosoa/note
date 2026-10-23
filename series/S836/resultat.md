---
serie: S836
statut: en_cours
type: experience
branche: codex/note-s836-live-sharp-cache256
promotion: false
---

# S836 — Résultats et analyse d'intervention

## Situation de départ mesurée
S834 SHADOW 80 pistes réelles : 306536 frontières PCM256, même source S708 gelée, p95 CPU Linux par bloc entre 6,61 et 7,75 ms contre un budget strict de 5,805 ms. Aucune mesure micro→piano Windows concluant à une amélioration n'est fournie. Les scores musicaux et références sont conservés dans `series/S834/resultat.md`, et les meilleures références par axe dans `series/S835/resultat.md`.

## Boucle H1 : suppression du calcul FFT redondant S706
Statut initial : **implémentation en mode SHADOW facultatif**. L'ancienne voie reste sélectionnée par défaut. Aucune promotion.

Résultats à inscrire à partir de CI :
- TESTS : à mesurer
- Parité 96D sur signal même audio : à mesurer
- p50, p95, p99 du seul S706 synthétique : à mesurer
- Parité des six sorties sur modèles réels : **NON VALIDÉE tant qu'un replay acoustique signé n'a pas réussi**
- Gain CPU du pipeline complet S708 + six axes : **NON MESURÉ**
- Qualité guitare/microphone, NoteOn/NoteOff et polyphonie en Live : **NON MESURÉE**
- TP/FP/FN de cette intervention : en cas de parité parfaite = mêmes événements, sinon audit détaillé obligatoire.
- Budget global 5,805ms et Windows/iRig : **NON VALIDÉS**

Un gain de performance du seul composant FFT n'autorise pas à conclure que les six caractéristiques fonctionnent ensemble en vrai Live ; un calcul de p95 CPU sur runner Linux ne remplace pas le PC Windows/iRig.
