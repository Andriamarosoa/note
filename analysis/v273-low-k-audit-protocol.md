# Audit des surcomptages à vrai K inférieur à 4

Périmètre : les 14 744 groupes du fold externe 3 dont le vrai K vaut 0, 1, 2
ou 3. Un surcomptage signifie K prédit > K vrai, même si la prédiction vaut
4, 5 ou 6. Les quatre modèles déjà entraînés sont figés. Aucun entraînement,
aucun autre fold externe, aucun correcteur ni changement de référence.

1. Vérifier les poids, probabilités et identités des lignes. Comparer les
   décisions à fenêtre et pondération identiques, puis à pondération seule
   différente. Distinguer les nouveaux surcomptages, ceux persistants, ceux
   devenus exacts et ceux devenus des sous-comptages.
2. Recalculer les étiquettes depuis les annotations et les candidats entiers
   complets conservés, avec la règle historique du candidat le plus proche.
   Mesurer les attaques attribuées à un autre groupe, les attaques non
   attribuées, les notes antérieures encore actives et les fins de notes dans
   la fenêtre. K=0 signifie zéro attaque attribuée, pas nécessairement silence.
3. Rejouer l'inférence TensorFlow avec les mêmes poids et entrées. Exiger les
   mêmes argmax et des probabilités numériquement concordantes. Examiner le
   graphe réellement conservé dans le composant natif.
4. Sondes de sensibilité, sans correction des sorties : neutraliser séparément
   les représentations candidates et spectrales avant la tête inchangée ; à
   31 trames, remplacer les huit dernières par zéro ou par répétition de la
   23e trame ; neutraliser les huit premières comme contrôle de localisation.
   Vérifier les effets sur tous les K<4 et sur les nouveaux surcomptages.

Les sondes préservent les poids et le décodage direct argmax. Elles peuvent
produire des entrées hors distribution : elles mesurent une sensibilité du
modèle, pas une méthode de correction ni une preuve acoustique suffisante.
La répétition de la 23e trame intervient sur les trois canaux déjà calculés ;
elle n'est pas présentée comme un signal physique resynthétisé.

Les associations sont rapportées par vrai K pour réduire la confusion due
au mélange des classes. Les résultats gardent les dénominateurs, les
probabilités et les erreurs qui apparaissent après intervention. Aucun seuil
ou hyperparamètre n'est choisi pour améliorer le fold externe. La confusion
d'harmoniques n'est pas affirmée sans vérification acoustique spécifique.

Sources : les sorties de l'exécution 36351028493 et le cache exact à 31 trames
de l'exécution 36139446276. Les archives sont figées par SHA-256 dans le
fichier de lancement ; l'archive d'annotations est contrôlée par son MD5
historique. Les fichiers sources et résultats restent consultables sur GitHub.
