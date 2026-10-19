# NOTE — S755 : nouvelles architectures polyphoniques cohérentes en hauteur

**Statut** : branche expérimentale, pas de promotion, aucun nouveau F1
revendiqué avant rapport GitHub Actions final.

## Pourquoi S754 ne pouvait suffire

1. Le rapport S753 a prouvé deux blocages distincts en GuitarSet CAL32 :
   la tête K6 a prédit correctement zéro fois sur 189 décisions,
   et K2–K4 restent sous-comptés même quand le K prédit est suffisant.
2. S754 a entraîné quatre nouvelles têtes K et évalué **12 streams MIDI
   complets**. CI verte mais 0 variante admissible. La variante
   `depth5_natural_blend55` améliore 5 scores mais régresse l'Offset
   77,3292 % → 77,3086 % et perd 4 appariements Offset corrects.
3. Inspection code S754 : pour les `p_top` de dimensions 104D, les
   canaux Onset, Sustain et Release sont triés **indépendamment**
   entre les 49 hauteurs. La représentation conserve leurs valeurs
   marginales mais perd quelle attaque correspond à quel Sustain/
   Release ; mauvaise représentation possible pour distinguer de vrais
   accords des produits harmoniques. Ceci est une **hypothèse de
   conception**, pas une causalité mesurée sur corpus.
4. Le résumé texte S754 choisissait `best_head_k6` par `max((nom,
   nombre_de_K6_justes))`, comparant les **noms** avant les nombres ;
   d'où `depth5_poly_balanced_blend55: 0` affiché comme « best » alors
   que `depth5_natural_blend55` possédait **1/189**. Les résultats
   détaillés étaient corrects. Le résumé S755 classe correctement par
   valeur métrique.

## Expérience S755

- Conserver les 104 features K S754 comme contrôle.
- Ajouter **136 données acoustiques co-alignées** : huit hauteurs
  choisies par un score actuel commun, leurs Onset/Sustain/Release S708
  (24), canaux originaux (24), profils de huit harmoniques sur la grande
  FFT (64), hausse d'Onset au fragment (8), et interactions harmoniques
  ±12/±19 demi-tons (16). Total **240 dimensions**.
- Deux profondeurs d'arbre HGB × deux profils de surpondération
  **distincte** K5/K6, puis mélange 15/35/55 % avec la tête
  originale : **12 combinaisons évaluées en musique MIDI complète**.
- FIT80 uniquement pour les modèles ; CAL32 pour comparaison ;
  annotations lues après émission des 13 flux d'événements complets
  (S708 + 12 nouveaux) sur chaque piste.
- Gardes : même checkpoint SHA S708 et même fermeture S707,
  zéro régression sur les six métriques, aucun ancien Offset
  correct/long Onset perdu ni moins de TP Onset.
- HELD/TEST fermés, aucun changement du LIVE, S80/S708/S707 gelés ;
  aucune distillation étudiant.

## Tests négatifs

CI doit prouver notamment :
- deux scènes de notes ayant les mêmes valeurs Onset/Sustain
  indépendamment triées par canal **diffèrent** dans la
  représentation alignée ;
- modifier le PCM futur ne modifie pas les features du passé ;
- blend=0 reproduit K signé S708 ;
- ABI complète 240D sans NaN ni dimension approximative ;
- six métriques finales dérivent du MIDI causal décodé.

### Incident CI initial S755

Le premier lancement a échoué sur une faute **de dimension
d'assertion** (16 interactions ±12/±19, pas 8 ; la taille réelle
240D, pas 232D). Corrigé sur la branche S755 ; le premier
run ne doit pas être interprété comme une validation modèle.

## Références

- S754 expérimental : https://github.com/Andriamarosoa/note/actions/runs/39210823815
- S755 CI courante : https://github.com/Andriamarosoa/note/actions/runs/39213327892
- S755 PR DRAFT : https://github.com/Andriamarosoa/note/pull/86

**Interdiction** : déclarer « 80 % » sur des têtes isolées ou un
réglage non validé. Le seuil est requis sur les six métriques finales,
simultanément sur le même flux MIDI et hors TEST exploratoire.
