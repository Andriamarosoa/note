# NOTE — S689–S692: Onset causal sans régression (état mesuré)

Périmètre : vraie piste MIC GuitarSet, 36 TRAIN (9 familles) pour figer le choix, puis 12 DEV (3 familles). Les enregistrements DEV ont déjà été exposés historiquement : aucun de ces scores ne prouve à lui seul la généralisation à une guitare inconnue. TEST reste fermé.

## Baseline native S630 + S675 inchangée

- Onset hauteur correcte (±80 ms) : **998 TP / 585 FP / 582 FN**, F1 **63,1046 %**.
- Exact-K **décodé en nombre de notes actives par trame** : **26 675 / 48 970 = 54,4721 %**. Ne pas confondre avec l'Exact-K S80 historique et son corpus.
- S675 NoteOff (±50ms sur Onsets appariés) : **773 / 990 = 78,0808 %**.
- F1 note entière (hauteur + Onset + Offset) : **48,8776 %**.

## S689 — preuve harmonique indépendante

Politique retenue uniquement sur TRAIN : `independent_floor_0.35_ratio_1.2`.
Sur DEV : Onset TP **1 021**, FP **631**, FN **559**, F1 **63,1807 %** ; Exact-K décodé **54,8213 %** ; S675 NoteOff **789/1013 = 77,8875 %**, F1 complet **48,8243 %**.

Gain de 23 vraies attaques, mais +46 faux Onsets et baisse du pourcentage de NoteOff et du F1 complet. **Rejeté.**

https://github.com/Andriamarosoa/note/actions/runs/39063961425

## S690 — tête de qualité uniquement sur événements Onset nouvellement créés

Apprentissage sur **80 vraies prises GuitarSet TRAIN**, étiquettes sur 202 propositions supplémentaires (84 correctes, 118 fausses). Les NoteOn natifs sont conservés tels quels ; la tête supplémentaire n'est qu'un filtre des notes ajoutées.

Politique TRAIN-held : `gate_0.65`.

Sur DEV : **1 000 TP, 586 FP, 580 FN** : F1 Onset **63,1712 %**, Exact-K **54,6089 %**. NoteOff **774/992 = 78,0242 %**, F1 entier **48,8945 %**. Malgré +2 TP/ +1 FP et un gain de F1 complet, la précision conditionnelle NoteOff baisse légèrement. **Rejeté selon les quatre verrous stricts.**

https://github.com/Andriamarosoa/note/actions/runs/39064605908

## S691 — consensus des deux têtes indépendantes

Aucune variante ne franchit les verrous Onset et Exact-K sur les neuf familles TRAIN. Choix automatique `native`. **Rejeté.**

https://github.com/Andriamarosoa/note/actions/runs/39064537342

## S692 — spécialisation grave / médium / aigu

TRAIN : politique `bassmid_0.35_1.2` retenue provisoirement, mais échec du verrou S675/final Note.
DEV : TP **1013**, FP **613**, F1 Onset **63,1940 %**, Exact-K **54,7539 %**, S675 **783/1005 = 77,9104 %**, F1 complet **48,8459 %**. **Rejeté.**

https://github.com/Andriamarosoa/note/actions/runs/39064771981

## Portée, limites et suite

1. Les postérieurs K0..K6 et les têtes natives restent **byte-identiques**; le K décodé peut changer à cause de nouvelles décisions Onset.
2. Le code expérimental **n'a pas été branché au moteur MICRO LIVE Windows** : ni charge CPU, ni overflow, ni délai matériel validés.
3. Un résultat `DEV` inférieur sur un seul des quatre verrous interdit la promotion, même si le F1 Onset s'améliore.
4. S693 étudie la sélection à partir de la **polyphonie prédite K2+**, jamais du vrai K de la référence. Son résultat n'est pas encore inclus ici.
5. L'amélioration durable nécessite probablement une tête supervisée spécifiquement sur les nouvelles fondamentales et une validation sur d'autres sources que GuitarSet DEV.

**Référence officielle LIVE inchangée.**
