# Renouvellement des trames

Le scheduler conserve uniquement les événements ouverts et les trames demandées
à leurs offsets.

Il reçoit directement les événements live :

```text
onset(A, 100)
onset(B, 200)
offset(A, 700)
offset(B, 820)
```

Pour A fermé à 700 puis B fermé à 820 :

```text
F700 = audio[700:1212)
F820 = audio[820:1332)
```

À 820, `F700` réutilise les 120 échantillons déjà reçus et attend encore 392
échantillons. `F820` attend 512 échantillons. Lorsque le flux atteint 1212,
`F700` est disponible. Lorsqu’il atteint 1332, `F820` est disponible.

L’absence de détection dans `F700` n’annule jamais `F820`.

Une fois une trame analysée, `complete_frame(start)` puis `prune_completed()`
retirent sa demande, l’événement fermé correspondant et l’audio antérieur au
plus ancien besoin restant. `LiveOnsetOffsetPipeline.complete_frame(...)`
effectue ces deux opérations ensemble par défaut.
