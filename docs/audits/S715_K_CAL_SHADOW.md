# S715 — Boucle de sélection K / attaques faibles en shadow

## Objet

L'audit [S714](S714_S708_EXACTK_ROOT_CAUSE.md) montre un sous-comptage
récurrent des accords K2–K5. La sélection officielle S708 a un
`count_confidence=1.1` ; ce nombre désactive volontairement le budget
strict K, tandis que `guide_confidence=0.5` garde une aide aux attaques
acoustiques faibles. **Changer le seuil directement en LIVE pourrait
dégrader les vrais accords : S715 ne le fait pas.**

## Ce que S715 teste réellement

- **Un seul checkpoint** S708/S706 `s708_selected.pt`, SHA verrouillé.
- Les variantes `count_confidence` dans `0.55, 0.70, 0.85, 1.10`,
  `guide_confidence` dans `0.50, 0.70` et
  `weak_sustain` dans `0.45, 0.55`, en plus de la référence S708.
- Réévaluation de chaque décision depuis les acoustiques NOTE et
  les distributions K prédites par le vrai modèle **sans toucher aux poids**.
- Un seul vrai flux NoteOn/NoteOff par variante ; replay de la politique
  research S707 `gate=0.80,persistence=1,grace=24`.
- **CAL 32 pistes** : seule phase autorisée à choisir, sur 6 métriques
  non régressives face au témoin S708, zéro perte d'attaques de notes
  longues et d'offsets auparavant corrects. S'il n'y a pas de progrès
  intégral, le témoin S708 reste sélectionné.
- **HELD 32 puis DEV 12** : uniquement après gel ; vérification que les
  scores du témoin reproduisent l'ancien S708, puis tests appariés de la
  variante choisie.
- **TEST non ouvert**, pas de promotion, pas de remplacement du LIVE,
  pas de futur PCM utilisé par l'inférence ; HELD/DEV historiquement
  déjà exposés, donc pas une nouvelle preuve indépendante.

## Exécution et preuves

[Workflow S715](https://github.com/Andriamarosoa/note/actions/workflows/note-s715-shadow-k-calibration.yml) dans
`codex/note-s713-independent-code-audit-agents`.
Le workflow télécharge l'artefact original S708 et le modèle hazard S675,
puis le corpus GuitarSet dont la taille et les SHA256 sont figés.
Il archive `report.json` comme preuve : CAL, candidat choisi et
comparaisons HELD/DEV. Les compteurs de scores sont vérifiés sur les
NoteOn/NoteOff effectivement émis.

L'état `success` de GitHub Actions atteste seulement que les étapes
du workflow ont abouti. **Un score >80 % doit être rapporté avec la
valeur exacte, le split et la paire de modèles dans ce rapport.**
Aucun gain chiffré S715 n'est affirmé avant qu'un `report.json` final
existe et ait été contrôlé.

## Si le modèle K ne s'améliore pas

Ne pas diminuer aveuglément les seuils. Priorités ensuite :
1. Vérifier l'information indépendante sur les fondamentales basses
   (E2/B2) et l'attaque soutenue, car S714 prouve que les trames-note
   MIDI 40–52 sont les plus souvent manquées sur DEV.
2. Apprendre un spécialiste attaques harmoniques/fondamentales sur
   compositions FIT uniquement, puis sélectionner sur CAL.
3. Coupler le relâchement S707 au vrai état des notes actives LIVE.
4. Valider les six scores sur des pistes nouvelles et les contraintes
   CPU/iRig avant toute promotion.
