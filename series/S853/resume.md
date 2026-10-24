# S853 — Résumé

Boucle audio-vers-audio pour surmonter S852 : utiliser une perte magnitude STFT multi-résolution (512/1024/2048 échantillons) et la dynamique d'un son réel transposé standard.

**Échec confirmé** : la variante log_spectral conserve la fondamentale, erreur de hauteur médiane 11,87 demi-tons sur six notes jamais utilisées en entraînement. Une corrélation d'enveloppe de 0,946 ne suffit pas : elle ne vérifie pas la note.

Aucun changement de modèle NOTE ni déploiement. Voir [résultats S853](resultat.md). Prochaine boucle : réseau multiplicatif et supervision harmonique.
