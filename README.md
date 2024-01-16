    I. Problématique
       Nous travaillons en tant que Data Scientist pour une société financière “Prêt à dépenser” qui souhaite pouvoir prédire si un client risque ou non d’être en défaut de paiement. 

	Nos objectifs sont donc de:
    • Construire un modèle de classification qui pourra prédire si le client risque d’être en défaut de paiement ou non.
    • Construire un Dashboard interactif à destination des conseillers clientèles permettant d'interpréter les prédictions faites par le modèle, et d’améliorer la connaissance client.
    • Mettre en production le modèle de prédiction à l’aide d’une API, ainsi que le Dashboard interactif qui appelle l’API pour les prédictions.
      II. Traitement des données
Les différentes étapes jusqu’à l’entraînement du modèle sont :
    • Le Traitement des valeurs manquantes en utilisant SimpleImputer de Scikit-Learn: on remplace les valeurs manquantes par la moyenne pour les variables numériques et par “Unknown” pour les variables catégorielles.
    • Le Traitement des variables catégorielles : pour les variables binaires (avec deux valeurs possibles), je remplace ces deux valeurs par 0 ou 1. Pour les autres variables, je regroupe au maximum les valeurs par similarité pour avoir le moins de valeurs possibles par catégories 
    • Le Traitement du déséquilibre: la table obtenue après nettoyage app_train_bis contient 92% de clients sans défaut de paiement et 8% avec défaut de paiement. Cette table contient plus de 300000 lignes (307511) , je fais donc le choix de traiter ce déséquilibre en prenant un échantillon de cette table pour entraîner mes modèles = Undersampling
    • Utilisation de OneHotEncoder pour finaliser le traitement des variables catégorielles.
    • Réalisation de RFECV (Recursive Feature Elimination avec cross validation) puis RFE pour sélectionner les dix meilleures features à utiliser pour faire la prédiction.
III. Entraînement des modèles
Nous souhaitons prédire si un client risque d’être en défaut de paiement ou non. Nous devons donc utiliser des modèles de classification supervisée. J’ai choisi de tester 4 modèles: RandomForestClassifier, LogisticRegression, XGBoost et CatBoostClassifier.
Pour départager ces modèles, nous allons utiliser plusieurs scores:
    • Score AUC (Area Under the Curve) qui mesure l'aire sous la courbe ROC (Receiver Operating Characteristic). La courbe ROC représente la performance d'un modèle de classification à différents seuils de décision. Plus l'AUC est proche de 1, meilleure est la capacité du modèle à distinguer entre les classes (positives et négatives)
    • Accuracy (Précision) : rapport du nombre d’observations correctement classées sur le nombre total d’observations.
    • Score Business : score donnant 10 fois plus d’impact aux faux négatifs(crédit accordé à un mauvais client : perte de capital) qu’aux faux positifs (refus de crédit pour bon client : manque à gagner)
   Tous ces modèles ainsi que leurs hyperparamètres et les scores associés sont repertoriés sur une plateforme Open source MLFLOW.
MLflow est utilisé pour simplifier le développement, la gestion et le déploiement des modèles d'apprentissage automatique. Il fournit des composants pour l'enregistrement des paramètres et des métriques, la gestion des expérimentations, le suivi des modèles, et le déploiement des modèles dans différents environnements.
