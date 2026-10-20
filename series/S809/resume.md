---
serie: S809
statut: termine
type: infrastructure
reference: S708
branche: codex/note-s809-six-isolated-axis-pipelines
run_id: 39384080646
commit_sha: c3970969558520c7e0941c1d85b8251d2453eae3
promotion: false
---
# S809 — indépendance effective des six chaînes têtes → sélecteur → décodeur

## Objectif

Corriger le couplage architectural de S808, qui réunissait les nouvelles têtes de naissance/continuité dans un seul flux MIDI. Définir et exécuter réellement une chaîne indépendante pour chaque objectif : Exact-K, Active Exact-K, Onset, Offset, Frame Pitch et Full Note. Aucun composant d'un axe n'est autorisé à écrire l'état ou à influencer les décisions d'un autre. S708 demeure référence officielle.

## Hypothèse

Séparer l'entraînement, l'espace de propositions, la sélection, le décodeur, son état et sa sortie peut permettre de mener des ablations sur une caractéristique sans changer les cinq autres sorties **shadow**. Cela ne signifie PAS que toutes les métriques du **MIDI final** pourront rester identiques après une future fusion. Il faudra un arbitre commun validé indépendamment, sans copier arbitrairement les six métriques shadow dans le bilan S704.

## Modifications

Nouvelle `src/causal_note/s809_isolated_axis_pipeline.py` : six objets `AxisPipeline` distincts, chacun possédant son `AxisHeadBank`, son `AxisSelector` et son décodeur privé repris du prototype vérifié S763. Les six axes possèdent douze noms de têtes de **signaux acoustiques** différents, six sélecteurs distincts, six instances de décodeurs et des états temporels non partagés. Les douze têtes sont de simples alternatives déterministes causales sur des preuves acoustiques déjà reçues : **elles ne sont pas présentées comme douze nouveaux réseaux entraînés**. Chaque axe est réservable pour son propre apprentissage ultérieur. Aucune de ces modifications n'affecte les composants LIVE/S708.

Les deux modes sont `native_only` (témoin) et `causal_contrast` (sélecteur alternatif calculé à la trame reçue). Exact-K sélectionne parmi distributions K0–K6 ; Active-K et Frame Pitch sélectionnent leur propre estimation d'occupation ; Onset sélectionne les nouvelles attaques ; Offset sélectionne son indice de terminaison ; Full Note produit ses propres NoteOn/NoteOff appariés. **Ne pas fusionner** ces propositions dans un flux MIDI synthétique pour déclarer les scores d'une seule caractéristique. Un observateur `ShadowArbiter` peut comparer les sorties mais ne peut ni les réécrire ni promouvoir S708.

## Protocole

`test/test_s809_axis_isolation.py` teste les six axes, l'indépendance des noms, objets/états, la modification isolée d'un sélecteur, d'une tête (même hostile), ou du décodeur, l'ordre d'exécution, la causalité (préfixe invariant face au futur), l'absence de partage de composants, et le caractère bit-à-bit préservé du MIDI natif.

`scripts/s809_fit80_independence_replay.py` vérifie les 80 pistes authentiques GuitarSet FIT80 S800, en signant les entrées SHA256 et en retrouvant les 12 compteurs du MIDI S708 pour chaque piste. Il rejoue une chaîne entière de six axes par piste et six perturbations « un seul sélecteur modifié à la fois » sur les 64 premières fenêtres acoustiques de **chaque** piste. Il exige l'identité stricte des cinq autres sorties et des cinq autres sélecteurs sous chaque mutation. Les sorties alternatives sont enregistrées par axe seulement, sans MIDI final modifié. Les six métriques S704 finales sont reproduites exclusivement sur les événements source S708 immuables. Un test par axe ne peut pas prétendre représenter cette même métrique officielle : son diagnostic est seulement SHADOW.

Limite instrumentale honnête : S800 ne stocke pas les distributions [7] K originales, mais uniquement la classe K argmax (et une entropie). Le replay utilise donc un proxy K one-hot et refuse explicitement toute revendication de nouvelles performances Exact-K à partir de cet artefact.

Un push ordinaire lance uniquement syntaxe, tests négatifs et vérification des rapports. Un unique commit portant `[run-fit80]`, ou un workflow_dispatch explicite, autorise l'exécution des 80 vrais FIT80 et des six audits locaux. Aucune API NVIDIA supplémentaire ni aucun run de doublon.

## Audits obligatoires

Données : 80/80 FIT authentiques et signatures+12 compteurs inchangés. Acoustique : trame reçue seulement, limitation du K proxy explicitée. Architecture : six objets, douze têtes et six sélecteurs nominativement distincts, 30 invariances croisées. Apprentissage : ne pas prétendre entraîner de nouveaux modèles. Métriques : distinguer strictement scores S704 de la sortie unique et diagnostics des axes. LIVE : aucune attente ni anticipation ; coût matériel CPU/iRig **non mesuré**. Six audits CI spécifiques, consensus 6/6 exigé avant résultat final de cette série d'infrastructure. Les six agents Work personnalisés sont distincts des six jobs CI, et les trois NVIDIA ne sont pas appelés.

## Statut

Démarré sur une branche de recherche indépendante, S708 intégralement protégée. Une fois les vrais tests terminés, reporter les preuves de séparation mesurée. Ne jamais annoncer amélioration Onset/Active-K ni promotion à partir de cette séparation structurelle seule.

## Finalisation vérifiée — 80 vraies pistes, 30 invariances

[Exécution scientifique S809 #39384080646](https://github.com/Andriamarosoa/note/actions/runs/39384080646) **SUCCESS** sur le SHA scientifique `c3970969558520c7e0941c1d85b8251d2453eae3`. **23 tests unitaires PASS**, y compris l'ancienne batterie S763. Replay des 80 pistes GuitarSet FIT80 signées, 5 compositions, parité stricte piste par piste avec tous les douze compteurs MIDI S708 et mêmes six scores source FIT80.

À chaque axe ont été substituées les décisions de son **sélecteur privé** avec une alternative causale, puis tous les **cinq autres axes** ont été comparés avec l'architecture témoin. Leurs sorties et choix de sélecteur sont exactement égaux pour les 30 couples distincts. Nombre de décisions effectivement modifiées sur l'axe visé : Exact-K **1 184**, Active Exact-K **1 341**, Onset **477**, Offset **225**, Frame Pitch **235**, Full Note **857** ; **zéro** décision des autres axes affectée. L’expérience a utilisé les 64 premières fenêtres de chacune des 80 pistes pour chaque mutation, tandis que la chaîne témoin complète a traité chaque piste entière. Ces chiffres sont des décisions SHADOW et **ne sont PAS des gains musicaux**.

Les six audits CI autonomes Données, Acoustique, Architecture, Apprentissage, Métriques et LIVE ont tous renvoyé **PASS** et le consensus local 6/6 est **PASS**. Cela confirme le contrat d'architecture et les preuves numériques, pas le niveau de 80 % ni la vitesse iRig. Les trois modèles NVIDIA n'ont pas été consultés et les six agents Work personnalisés ne sont pas des jobs CI.

S809 est donc terminée comme **correction d'infrastructure et séparation** ; six décodeurs individuels sont opérationnels en *shadow*. Aucune tête nouvellement entraînée n'a été promue ; la sortie MIDI LIVE et ses six métriques S708 restent inchangées. Pour un nouveau modèle musical promu, entraîner séparément les véritables têtes et mesurer ensemble les six métriques du MIDI final, avec preuve de non-régression, CAL32 indépendant et latence CPU/micro.
