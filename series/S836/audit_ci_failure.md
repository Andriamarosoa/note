# S836 — Audit obligatoire d'une boucle inattendue

## Incident
- Run https://github.com/Andriamarosoa/note/actions/runs/39987476143
- L'étape `strict_parity` a réussi, mais l'étape signée `signed_real_five` a échoué dès le premier hop S834 référence.
- Erreur observée dans le vrai stacktrace : `NameError: name 'nn_post' is not defined`, en `src/causal_note/s834_s809_native256_audio.py`, dans `_step`, lors de la construction du posterior.
- Le premier test de ce nouveau contrôle d'empreintes a introduit par erreur **des caractères littéraux `\\n`** sur une seule ligne commentée. La ligne commentait accidentellement `nn_post,nn_count=self.temporal.step(X)`.
- Le run antérieur https://github.com/Andriamarosoa/note/actions/runs/39986203235 **avait réellement réussi** les cinq pistes sans cette instrumentation. Ses scores/p95/p99 sont des mesures valides pour ce code antérieur, mais **ne valident pas** la version défectueuse.

## Pourquoi les tests initiaux n'ont pas détecté l'erreur
Les premiers tests vérifiaient la parité S706 96D et les six pipelines à partir de fragments déjà formés ; aucun ne traversait le vrai `S834S809Native256._step`. Un `compileall` valide une variable non définie utilisée plus tard ; il n'exécute pas le réseau. La lacune est donc une couverture de test insuffisante, pas une erreur du modèle guitar.

## Correction et preuve exigée
- Remettre les trois instructions Python sur trois vraies lignes : commentaire, `self.s836_last_input_288d=X`, puis `nn_post,nn_count=self.temporal.step(X)`.
- Ajouter `test_real_s834_step_invokes_temporal_net_in_both_engines` : vrais appels `_step` référence ET cache, chaque `temporal.step` exactement une fois et six décisions égales.
- Rerun du même 5×1 source GuitarSet signé ; comparer en plus les SHA256 **de chaque frame source [49,288], chaque frame spectral [49,96], et de chaque décision et sélection des six axes**.
- Mesurer séparément p95/p99 CPU. Les empreintes sont calculées après `push()`, hors du chronométrage CPU et ne peuvent expliquer artificiellement un ralentissement.
- Interdit d'ignorer l'échec de ce run ou de le prendre pour un gain/régression musicale. Il ne démontre que l'absence d'exécution du réseau au premier hop.

## Protections permanentes
Ce test d'exécution réelle est désormais obligatoire avec le test spectral; aucun changement d'Onset/Offset/Exact-K/Frame Pitch/Full Note, aucun MIDI public, aucune confirmation ajoutée, aucune promotion avant replay musical et mesure physique.
