import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, fbeta_score, roc_auc_score, roc_curve
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


## Fonctions pour descriptions des dataframes
list_dataset = []
list_shape = []
list_variable = []
list_duplicates = []
list_unique = []
list_null = []
list_null_ratio = []
list_type = []

dic_var_info = {'Dataset': list_dataset,
                'Shape': list_shape,
                'Variable': list_variable,
                'Duplicates': list_duplicates,
                'Unique': list_unique,
                'Null': list_null,
                'Null_ratio': list_null_ratio,
                'Type': list_type}

def var_info(df,df_name,var,print_=0,n=5):

    list_dataset.append(df_name)
    list_shape.append(df.shape)
    list_variable.append(var)
    list_duplicates.append(df.duplicated(subset=[var]).sum())
    list_unique.append(df[var].unique().shape[0])
    list_null.append(df[var].isna().sum().sum())
    list_null_ratio.append((df[var].isna().sum().sum())/df.shape[0])
    list_type.append(df[var].dtypes)

    df_data_model = pd.DataFrame(dic_var_info)

    if print_ == 1:
        varInfo = df_data_model.tail(n)
        return varInfo


def all_var_info(df,df_name):

    last_value = df.columns.tolist()[len(df.columns.tolist())-1]
    for i in df.columns.tolist():
        if i != last_value:
            var_info(df,df_name,i)
    return var_info(df,df_name,last_value,1,df.shape[1])



# fonction de coût prenant en compte le fait qu'un faux négatif coûte 10 fois plus cher qu'un faux positif
def cost_score(y_true,y_pred,fn_cost=10, fp_cost=1):
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    loss = fn * fn_cost + fp * fp_cost
    score = loss
    return score



# Fonction pour identifier le seuil optimal

def find_optimal_threshold(y_true, y_pred, fn_cost=10, fp_cost=1, threshold_range=(0, 1), threshold_step=0.01):
    thresholds = np.arange(threshold_range[0], threshold_range[1] + threshold_step, threshold_step)
    
    best_score = float('inf')
    best_threshold = None

    for threshold in thresholds:
        y_pred_thresholded = (y_pred >= threshold).astype(int)
        score = cost_score(y_true, y_pred_thresholded, fn_cost, fp_cost)

        if score < best_score:
            best_score = score
            best_threshold = threshold

    print(f"Seuil optimal : {best_threshold}")
    print(f"Meilleur score associé : {best_score}")

    return best_threshold, best_score


#fonction du MLFlow permettant d'enregistrer les expériences 

def train_and_log_model(model, X_train, y_train, X_test, y_test, hyperparameters=None, run_name="Model", metrics=None):
    # Check if there's an active run and end it
    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run(run_name=run_name) as run:
        # Utiliser des hyperparamètres s'ils sont fournis
        if hyperparameters is not None:
            model.set_params(**hyperparameters)
            # Log hyperparamaters
            mlflow.log_params(hyperparameters)
            

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Log the model
        mlflow.sklearn.log_model(model, "Trained_Model")

        # Log metrics
        if metrics is not None:
            for metric_name, metric_func in metrics.items():
                metric_value = metric_func(y_test, y_pred)
                mlflow.log_metric(metric_name, metric_value)

        # Log Business score
        if "Business score" in metrics:
            business_score_value = metrics["Business score"](y_test, y_pred)
            mlflow.log_metric("Business score", business_score_value)

        # Autolog other metrics and parameters
        mlflow.autolog()



# il est important de faire ressortir les FN (Faux négatifs) car perte de capital
# les FP(Faux positifs) sont moins risqués car perte d'interêt seulement
# le coût d’un FN est dix fois supérieur au coût d’un FP

def custom_metric(y, y_pred):
    TP = np.sum( (y==1) & (y_pred==1) )
    FP = np.sum( (y==0) & (y_pred==1) )
    TN = np.sum( (y==0) & (y_pred==0) )
    FN = np.sum( (y==1) & (y_pred==0) )
# Fowlkes–Mallows index
# https://en.wikipedia.org/wiki/Confusion_matrix
# https://en.wikipedia.org/wiki/Fowlkes%E2%80%93Mallows_index
# Positive Predictive Value PPV = Precision
    PPV = TP / (TP + FP)
# True Positive Rate TPR = Recall
    TPR = TP / (TP + FN)
    FMI = np.sqrt( PPV * TPR )
    return FMI   


def custom_metric_f2(y, y_pred):
    """
        https://machinelearningmastery.com/fbeta-measure-for-machine-learning/
        F2-Measure
            The F2-measure is an example of the Fbeta-measure with a beta value of 2.0.

            It has the effect of lowering the importance of precision and increase the importance of recall.

            If maximizing precision minimizes false positives, and maximizing recall minimizes false negatives,

            then the F2-measure puts more attention on minimizing false negatives than minimizing false positives.

    """
    return fbeta_score(y, y_pred,beta=2)


def model_eval_score(model, Xval, yval):
    yval_pred = model.predict(Xval)
    conf_mat = confusion_matrix(yval, yval_pred)

     
    yval_proba = model.predict_proba(Xval)[:, 1]
    optimal_threshold, _ = find_optimal_threshold(yval, yval_proba)
    y_pred_optimal_threshold = (yval_proba >= optimal_threshold).astype(int)

     #Plotting the confusion matrix as a heatmap
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    print(f'Accuracy score    : {accuracy_score(yval, yval_pred):.3f}')
    print(f'precision score   : {precision_score(yval, yval_pred):.3f}')
    print(f'recall score      : {recall_score(yval, yval_pred):.3f}')
    print(f'F1 score          : {f1_score(yval, yval_pred):.3f}')
    print(f'F2 score          : {fbeta_score(yval, yval_pred, beta=2):.3f}')
    print(f'ROCAUC score      : {roc_auc_score(yval, yval_pred):.3f}')
    print(f'custom metric FMI : {custom_metric(yval, yval_pred):.3f}')
    print(f'score business    : {cost_score(y_test, y_pred_optimal_threshold)}')
    print()




def model_eval(model, Xval, yval):
    model_eval_score(model, Xval, yval)

    yval_proba = model.predict_proba(Xval)[:, 1]
    
    fpr, tpr, thresholds = roc_curve(yval, yval_proba)
    roc_auc = roc_auc_score(yval, yval_proba)

    plt.figure(figsize=(8, 6))
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()    


## Fonctions pour descriptions des dataframes
list_dataset = []
list_shape = []
list_variable = []
list_duplicates = []
list_unique = []
list_null = []
list_null_ratio = []
list_type = []

dic_var_info = {'Dataset': list_dataset,
                'Shape': list_shape,
                'Variable': list_variable,
                'Duplicates': list_duplicates,
                'Unique': list_unique,
                'Null': list_null,
                'Null_ratio': list_null_ratio,
                'Type': list_type}

def var_info(df,df_name,var,print_=0,n=5):

    list_dataset.append(df_name)
    list_shape.append(df.shape)
    list_variable.append(var)
    list_duplicates.append(df.duplicated(subset=[var]).sum())
    list_unique.append(df[var].unique().shape[0])
    list_null.append(df[var].isna().sum().sum())
    list_null_ratio.append((df[var].isna().sum().sum())/df.shape[0])
    list_type.append(df[var].dtypes)

    df_data_model = pd.DataFrame(dic_var_info)

    if print_ == 1:
        varInfo = df_data_model.tail(n)
        return varInfo


def all_var_info(df,df_name):

    last_value = df.columns.tolist()[len(df.columns.tolist())-1]
    for i in df.columns.tolist():
        if i != last_value:
            var_info(df,df_name,i)
    return var_info(df,df_name,last_value,1,df.shape[1])



# fonction de coût prenant en compte le fait qu'un faux négatif coûte 10 fois plus cher qu'un faux positif
def cost_score(y_true,y_pred,fn_cost=10, fp_cost=1):
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    loss = fn * fn_cost + fp * fp_cost
    score = loss
    return score



# Fonction pour identifier le seuil optimal

def find_optimal_threshold(y_true, y_pred, fn_cost=10, fp_cost=1, threshold_range=(0, 1), threshold_step=0.01):
    thresholds = np.arange(threshold_range[0], threshold_range[1] + threshold_step, threshold_step)
    
    best_score = float('inf')
    best_threshold = None

    for threshold in thresholds:
        y_pred_thresholded = (y_pred >= threshold).astype(int)
        score = cost_score(y_true, y_pred_thresholded, fn_cost, fp_cost)

        if score < best_score:
            best_score = score
            best_threshold = threshold

    print(f"Seuil optimal : {best_threshold}")
    print(f"Meilleur score associé : {best_score}")

    return best_threshold, best_score


#fonction du MLFlow permettant d'enregistrer les expériences 

def train_and_log_model(model, X_train, y_train, X_test, y_test, hyperparameters=None, run_name="Model", metrics=None):
    # Check if there's an active run and end it
    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run(run_name=run_name) as run:
        # Utiliser des hyperparamètres s'ils sont fournis
        if hyperparameters is not None:
            model.set_params(**hyperparameters)
            # Log hyperparamaters
            mlflow.log_params(hyperparameters)
            

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Log the model
        mlflow.sklearn.log_model(model, "Trained_Model")

        # Log metrics
        if metrics is not None:
            for metric_name, metric_func in metrics.items():
                metric_value = metric_func(y_test, y_pred)
                mlflow.log_metric(metric_name, metric_value)

        # Log Business score
        if "Business score" in metrics:
            business_score_value = metrics["Business score"](y_test, y_pred)
            mlflow.log_metric("Business score", business_score_value)

        # Autolog other metrics and parameters
        mlflow.autolog()


def model_eval_score(model, Xval, yval):
    yval_pred = model.predict(Xval)
    conf_mat = confusion_matrix(yval, yval_pred)

     
    yval_proba = model.predict_proba(Xval)[:, 1]
    optimal_threshold, _ = find_optimal_threshold(yval, yval_proba)
    y_pred_optimal_threshold = (yval_proba >= optimal_threshold).astype(int)

     #Plotting the confusion matrix as a heatmap
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    print(f'Accuracy score    : {accuracy_score(yval, yval_pred):.3f}')
    print(f'precision score   : {precision_score(yval, yval_pred):.3f}')
    print(f'recall score      : {recall_score(yval, yval_pred):.3f}')
    print(f'F1 score          : {f1_score(yval, yval_pred):.3f}')
    print(f'F2 score          : {fbeta_score(yval, yval_pred, beta=2):.3f}')
    print(f'ROCAUC score      : {roc_auc_score(yval, yval_pred):.3f}')
    print(f'custom metric FMI : {custom_metric(yval, yval_pred):.3f}')
    print(f'score business    : {cost_score(y_test, y_pred_optimal_threshold)}')
    print()




def model_eval(model, Xval, yval):
    model_eval_score(model, Xval, yval)

    yval_proba = model.predict_proba(Xval)[:, 1]
    
    fpr, tpr, thresholds = roc_curve(yval, yval_proba)
    roc_auc = roc_auc_score(yval, yval_proba)

    plt.figure(figsize=(8, 6))
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()    


def train_test_data():
    # Générer des données d'entraînement et de test pour l'exemple
    X_train, X_test, y_train, y_test = train_test_split(
        np.random.rand(100, 5),  # caractéristiques
        np.random.randint(2, size=100)  # variable cible binaire
    )
    return X_train, X_test, y_train, y_test