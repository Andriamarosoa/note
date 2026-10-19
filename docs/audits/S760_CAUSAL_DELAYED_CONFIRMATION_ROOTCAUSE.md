# S760 — NoteOn confirmé causalement après 1–3 fragments (23/46/70 ms)

**État : expérimental, non promu.** Aucune mesure CAL32 S760 ne sera
déclarée avant la fin du GitHub Actions
https://github.com/Andriamarosoa/note/actions/runs/39218925531

## Causes prouvées en S758 (et non supposées)

Audit réel S758 https://github.com/Andriamarosoa/note/actions/runs/39218136782 :
checkpoint signé S708 inchangé, candidat S757
`depth5_rare_guard160ms` strictement reproduit sur les 32 pistes
GuitarSet CAL32 (six axes parité). **Trois pertes de références
Offset correctes, toutes sous forme de réattribution Hungarian à un
NoteOn supplémentaire de même hauteur, précédant le NoteOn S708.**

| Morceau | Hauteur | Nouveau NoteOn (s) | NoteOn S708 (s) | Écart | Arrêt précoce du nouveau |
| --- | --- | ---: | ---: | ---: | --- |
| `00_BN2-166-Ab_comp` | MIDI61 | 19.063583 | 19.086803 | 23.22 ms | `s746_source_priority` |
| `03_BN2-166-Ab_comp` | MIDI54 | 21.594558 | 21.617778 | 23.22 ms | `s746_source_priority` |
| `01_Jazz2-110-Bb_comp` | MIDI54 | 21.106939 | 21.153379 | 46.44 ms | `s746_source_priority` |

Ces trois notes supplémentaires occupent temporairement le même
pitch et sont fermées à la prochaine attaque native (toujours
prioritaire). L'appariement `event_score` retient la note
supplémentaire parfois plus proche de la vérité pour le NoteOn et
évalue alors son NoteOff de préemption au lieu de l'ancien NoteOff
S708 correct.

Cette observation **ne signifie pas** que les autres notes
supplémentaires sont des faux positifs : il faut encore mesurer les
six métriques après changement. Les nouveaux NoteOn peuvent améliorer
la cardinalité active ou au contraire être indispensables dans
certains accords.

## Principe causal de S760

Le futur NoteOn S708 est évidemment inconnu au moment de la première
proposition, donc **il est interdit de le regarder pour l'annuler**.
S760 attend un, deux ou trois blocs de décision de 4×256 samples
(≈23,2/46,4/69,7 ms). Pendant cette attente, la proposition reste
**provisoire** et n'a pas encore émis de MIDI NoteOn.

À réception du bloc de confirmation :
- Les seuls événements natifs `available_sample <= now` sont visibles.
- Si une attaque native de la même hauteur est désormais active ou
  vient d'être émise, la proposition est supprimée.
- Si une fermeture du candidat a *déjà eu lieu*, la proposition est
  supprimée ; les fins qui se produiront plus tard ne servent
  pas au calcul de son admission.
- Les probabilités Onset/Sustain/K sont réévaluées **à ce bloc** ;
  le NoteOn confirmé reçoit ce timestamp réel, jamais l'ancien.
- En cas d'admission, le mux S746 conserve tous les événements S708
  inchangés et préempte toujours une voix supplémentaire plutôt que
  la source.
- Une éventuelle vraie nouvelle note est retardée : le score Onset
  peut baisser. **Cette perte ne peut pas être cachée.**

Quatre modèles S755 réellement appris sur **FIT80 uniquement**, trois
retards = **12 flux MIDI GuitarSet CAL32** complets, mesurés sur
exactement les six métriques signées S704 + perte éventuelle d'Offset
correct, long Onset et TP Onset.

## Garde-fous

S708 et S707 signés identiques, S80 officiel distinct et inchangé.
HELD et TEST fermés. Aucun nouveau mode LIVE branché au microphone,
latence matériel non validée. Aucun étudiant distillé avant six
scores >=80% prouvés, aucune promotion sur la seule réussite CI.

## Branches et preuves

- S758 audit des trois pertes :
  https://github.com/Andriamarosoa/note/pull/89
- S760 PR DRAFT :
  https://github.com/Andriamarosoa/note/pull/91
- S760 CI :
  https://github.com/Andriamarosoa/note/actions/runs/39218925531

**Verdict attendu** : une variante doit conserver simultanément
les six métriques S708 et zéro ancienne référence correcte perdue.
Le délai de confirmation n'est pas une preuve de score avant
l'achèvement du CAL32.
