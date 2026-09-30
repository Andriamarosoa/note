# Comparaison native du résidu — préparation validée

État avant lancement, 30 septembre 2026. Aucun résultat Exact K de cette
comparaison n’est encore disponible.

- Protocole fixé avant entraînement : deux bras, 12 époques, une graine.
- 43 357 groupes d’apprentissage et 15 952 de validation, 190 pistes internes.
- 59 309 cartes préparées ; aucun échantillon du fold externe évalué.
- Même architecture : 249 282 paramètres, poids et dropouts initiaux identiques.
- Ancienne initialisation à trois canaux préservée.
- 21 tests réussis sous TensorFlow 2.15.1 et NumPy 1.26.4.
- Une mise à jour sur un vrai lot de 128 groupes d’apprentissage est finie
  dans chaque bras ; inférence sur 256 groupes vérifiée. Ces poids sont jetés.
- Le workflow sauvegarde les époques 4/8/12 et recalcule les métriques à la fin.

[Protocole](v273-residual-native-protocol.md) ·
[Preuves locales](evidence/v273-residual-native/prelaunch.json) ·
[Workflow](../.github/workflows/v273-residual-native.yml)

La comparaison normal/bass rescue/high pitch rescue reste non mesurée : ces
catégories ne sont pas définies dans les sources retrouvées de ce dépôt. Les
origines des propositions V8.6 sont `peak` et `baseline_edge`. Une analyse par
hauteur MIDI ne serait pas équivalente à un audit de ces mécanismes de rescue.
