# S746 — Sélection d'instances complètes sans toucher aux notes S708

Branche : `codex/note-s746-instance-safe-multimetric-loops`.

## Motif mesuré dans S745

Le run S745 39198255358 a exécuté 12 CAL32 : Active Exact-K 60,0840 % au
mieux contre 56,2762 % de S708, mais Onset 72,7824 %, Offset 76,1598 %
et Full Note F1 55,1113 %. 39 Offsets précédemment corrects perdus.
Tous les essais ont été rejetés et S708 reste référence.

**Cause logicielle à tester :** les naissances supplémentaires S743/S745
réexécutaient l'émetteur S707 sur un flux modifié, ce qui pouvait modifier
des instances et des fermetures déjà correctes. S746 isole d'abord les
instances proposées et garde les événements S708 d'origine, bit-à-bit.

## Architecture expérimentale native NOTE

1. Charger les mêmes checkpoints S708 signé, S734, S707 et le même protocole
   GuitarSet FIT80/CAL32 disjoint par composition.
2. Générer les propositions de naissances avec les **6 têtes acoustiques
   H0–H5** et une **sélection de masque H0 ancré parmi 32 possibilités par
   fragment**. S743 demeure le générateur causal ; pas de greffe de `midi`.
3. Terminer la **piste candidate uniquement** avec S707. Extraire ses
   NoteOn/NoteOff candidats. Ces Offsets ne servent **jamais à décider
   si un NoteOn doit être accepté à son instant**.
4. FIT80 uniquement : entraîner deux experts HGB de risque d'instance, sur
   les seules caractéristiques disponibles au fragment de NoteOn. Une
   cible positive exige une note de hauteur/onset correcte, non déjà
   reconnue par S708, avec une terminaison correcte à ±50 ms.
5. Un multiplexeur événementiel donne toujours priorité aux notes de S708,
   respecte au plus six notes simultanées et ne recalcule jamais les Offsets
   S708. Une nouvelle note qui interfère avec une note originale est
   fermée à l'instant de la collision, **jamais une note S708**.
6. Évaluer 12 expériences réelles : deux profondeurs HGB × trois risques
   (0,55/0,75/0,90) × deux contraintes K (0,20/0,55), toujours sur les
   mêmes 32 pistes CAL. Les six métriques proviennent d'un unique flux
   MIDI fusionné complet.

## Conditions de non-régression

- Événements S708 originaux conservés, nombre maximal de voix = 6.
- Aucun onset vrai perdu ; aucun NoteOff vrai auparavant correct perdu ;
  aucune note longue correctement attaquée perdue.
- Toutes les six métriques égales ou supérieures à S708 avec au moins
  une amélioration stricte, sur les 32 enregistrements.
- FIT uniquement pour l'apprentissage, CAL pour sélection, jamais HELD/TEST
  pour l'ajustement.
- Une éventuelle CAL admissible devra subir validation indépendante HELD et
  audit de causalité physique avant la moindre promotion.
- S708, S80/S397, microphone LIVE et S675 officiels sont inchangés.
- Aucun Student, aucune distillation avant l'objectif >80 % sur les six.

## État

**Lancement CI configuré ; résultats CAL non encore certifiés.** Les tests
unitaires incluent la préemption des notes ajoutées par la référence,
l'invariance du modèle vis-à-vis des Offsets futurs et l'identité des
événements source.

Workflow : `.github/workflows/note-s746-instance-safe-cal32.yml`.

Ne pas confondre succès de l'exécution avec promotion, ni une hausse
de Active Exact-K avec une progression validée des six caractéristiques.
