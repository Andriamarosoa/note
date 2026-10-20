# S802 — lecture scientifique des collisions synthétiques/natives

Le traitement du [run #39357068590](https://github.com/Andriamarosoa/note/actions/runs/39357068590) est **une étude forensique rétrospective**, pas un composant du décodeur. Il réutilise les mêmes décisions MIDI256 S795/S801 et la vérité terrain GuitarSet FIT80, uniquement après l'émission. Les cinq têtes OOF n'ont pas été réentraînées.

Sur le consensus, parmi 136 attaques synthétiques, 74 sont de vrais positifs et 62 de faux positifs au sens du matching onset officiel ±80 ms. **95** notes synthétiques se sont terminées par la fermeture de priorité native au même pitch : **64 étaient de vrais positifs, 31 de faux positifs**. La suppression indiscriminée des attaques précédant une native serait très dangereuse pour le rappel Onset.

Le profil FIT varie par composition : SS1 13 TP / 33 FP, Jazz1 18 TP / 17 FP et Funk2 22 TP / 3 FP. Une différence de composition peut refléter timbre, motif ou distribution d'attaque, mais aucune étiquette de composition ni connaissance du futur ne doit être utilisée par le routage causal.

Durées des collisions consensus : 84/95 fermées en 1 024 échantillons ou moins, dont 58 TP et 26 FP. Cela signifie que la seule brièveté des notes ou la proximité de la prochaine attaque native ne suffit pas à décider d'un veto.

Medianes consensus des mesures disponibles à l'instant de l'attaque : `on_rise` TP 0,7151 / FP 0,6958 ; `general_p` TP 0,9484 / FP 0,9436 ; `harmonic_parent` TP 0,0079 / FP 0,0168. Les distributions se recouvrent : pas de seuil garanti par ces medianes.

Le prochain travail concerne un classifieur de risque causal très local ou une politique de garde par fragment/hauteur, évaluée par composition sur FIT avant CAL32 et avec les six métriques appariées. La politique doit être rejetée en cas d'une seule régression et aucune latence supplémentaire ne peut être introduite.

S802 six audits GitHub **PASS**, consensus **PASS**, référence officielle **S708 inchangée**. Aucun septième appel GLM pendant cette série ; dernier appel réellement achevé en S799. Aucun des six agents personnalisés Work n'a été invoqué dans GitHub Actions.
