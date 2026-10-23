# Résultat — Onset Event Integrity / HGB256

Statut initial : protocole et code préparés ; nouveau replay audio non encore
terminé. Aucun nouveau score ni gain n'est revendiqué à cette étape.

Référence historique comparable : contrôle69, TP 8127 / FP 4519 / FN 3140,
F1 67,971396 %, run 39775357133.

Le contre-exemple unitaire « NoteOff puis NoteOn au même bloc » reproduit
la perte de l'attaque avec l'ancien arbitre. Le correctif laisse sortir le
NoteOn à cette même frontière de 256 échantillons.
