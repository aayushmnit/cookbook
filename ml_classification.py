"""
@author: Aayush Agrawal
@Purpose - Re-usable code in Python 3 for cross-validation and binary classification task in modeling process
"""

## Importing required libraries
import pandas as pd ## For DataFrame operation
import numpy as np ## Numerical python for matrix operations
from sklearn.model_selection import KFold, train_test_split ## Creating cross validation sets
from sklearn import metrics ## For loss functions
import matplotlib.pyplot as plt

## Libraries for Classification algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
import xgboost as xgb 
import lightgbm as lgb 
import lime
import lime.lime_tabular

########### Cross Validation ###########
### 1) Train test split
def holdout_cv(X,y,size = 0.3, seed = 1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size, random_state = seed)
    X_train = X_train.reset_index(drop='index')
    X_test = X_test.reset_index(drop='index')
    return X_train, X_test, y_train, y_test

### 2) Cross-Validation (K-Fold)
def kfold_cv(X,n_folds = 5, seed = 1):
    cv = KFold(n_splits = n_folds, random_state = seed, shuffle = True)
    return cv.split(X)

########### Model Explanation ###########
## Variable Importance plot
def feature_importance(model,X):
    feature_importance = model.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, X.columns[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()

## Functions for explaination using Lime
def make_prediction_function(model):
    predict_fn = lambda x:model.predict_proba(x).astype(float)
    return predict_fn

def make_lime_explainer(df, c_names = [], k_width=3, verbose_val = True):
    explainer = lime.lime_tabular.LimeTabularExplainer(df.values ,class_names=c_names,
                                                   feature_names = list(df.columns),
                                                   kernel_width=3, verbose=verbose_val)
    return explainer

def lime_explain(explainer,predict_fn, df, index = 0, 
                 show_in_notebook = True, filename = None):
    exp = explainer.explain_instance(df.values[index], predict_fn, num_features=df.shape[1])
    
    if show_in_notebook:
        exp.show_in_notebook(show_all=False)
    
    if filename is not None:
        exp.save_to_file(filename)

########### Algorithms For Binary classification ###########
 
### Running Xgboost
def runXGB(train_X, train_y, test_X, test_y=None, test_X2=None, seed_val=0, 
           rounds=500, dep=8, eta=0.05,sub_sample=0.7,col_sample=0.7,
           min_child_weight_val=1, silent_val = 1):
    params = {}
    params["objective"] = "binary:logistic"
    params['eval_metric'] = 'auc'
    params["eta"] = eta
    params["subsample"] = sub_sample
    params["min_child_weight"] = min_child_weight_val
    params["colsample_bytree"] = col_sample
    params["max_depth"] = dep
    params["silent"] = silent_val
    params["seed"] = seed_val
    #params["max_delta_step"] = 2
    #params["gamma"] = 0.5
    num_rounds = rounds

    plst = list(params.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=100, verbose_eval=20)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)
    
    pred_test_y = model.predict(xgtest, ntree_limit=model.best_ntree_limit)
    
    pred_test_y2 = 0
    if test_X2 is not None:
        pred_test_y2 = model.predict(xgb.DMatrix(test_X2), ntree_limit=model.best_ntree_limit)
    
    loss = 0
    if test_y is not None:
        loss = metrics.roc_auc_score(test_y, pred_test_y)
        return pred_test_y, loss, pred_test_y2, model
    else:
        return pred_test_y, loss, pred_test_y2, model

### Running Xgboost classifier for model explaination
def runXGBC(train_X, train_y, test_X, test_y=None, test_X2=None, seed_val=0, 
            rounds=500, dep=8, eta=0.05,sub_sample=0.7,col_sample=0.7,
            min_child_weight_val=1, silent_val = 1):
    model = xgb.XGBClassifier(objective="binary:logistic",
                              learning_rate=eta,
                              subsample=sub_sample,
                              min_child_weight=min_child_weight_val,
                              colsample_bytree=col_sample,
                              max_depth=dep,
                              silent=silent_val,
                              seed=seed_val,
                              n_estimators=rounds)

    model.fit(train_X, train_y)
    train_preds = model.predict_proba(train_X)[:,1]
    test_preds = model.predict_proba(test_X)[:,1]
    
    test_preds2 = 0
    if test_X2 is not None:
        test_preds2 = model.predict_proba(test_X2)[:,1]

    test_loss = 0
    if test_y is not None:
        train_loss = metrics.roc_auc_score(train_y, train_preds)
        test_loss = metrics.roc_auc_score(test_y, test_preds)
        print("Train and Test loss : ", train_loss, test_loss)
    return test_preds, test_loss, test_preds2, model

### Running LightGBM
def runLGB(train_X, train_y, test_X, test_y=None, test_X2=None, feature_names=None, seed_val=0,
           rounds=500, dep=8, eta=0.05,sub_sample=0.7,col_sample=0.7,
           silent_val = 1,min_data_in_leaf_val = 20, bagging_freq = 5):
    params = {}
    params["objective"] = "binary"
    params['metric'] = 'auc'
    params["max_depth"] = dep
    params["min_data_in_leaf"] = min_data_in_leaf_val
    params["learning_rate"] = eta
    params["bagging_fraction"] = sub_sample
    params["feature_fraction"] = col_sample
    params["bagging_freq"] = bagging_freq
    params["bagging_seed"] = seed_val
    params["verbosity"] = silent_val
    num_rounds = rounds
    
    lgtrain = lgb.Dataset(train_X, label=train_y)
    
    if test_y is not None:
        lgtest = lgb.Dataset(test_X, label=test_y)
        model = lgb.train(params, lgtrain, num_rounds, valid_sets=[lgtest],
                          early_stopping_rounds=100, verbose_eval=20)
    else:
        lgtest = lgb.Dataset(test_X)
        model = lgb.train(params, lgtrain, num_rounds)
        
    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    
    pred_test_y2 = 0
    if test_X2 is not None:
        pred_test_y2 = model.predict(test_X2, num_iteration=model.best_iteration)
    
    loss = 0
    if test_y is not None:
        loss = metrics.roc_auc_score(test_y, pred_test_y)
        print(loss)
        return pred_test_y, loss, pred_test_y2, model
    else:
        return pred_test_y, loss, pred_test_y2, model

### Running LightGBM classifier for model explaination
def runLGBC(train_X, train_y, test_X, test_y=None, test_X2=None, seed_val=0, rounds=500,
            dep=8, eta=0.05,sub_sample=0.7,col_sample=0.7,
            silent_val = 1,min_data_in_leaf_val = 20, bagging_freq = 5):
    model = lgb.LGBMClassifier(max_depth=dep,
                   learning_rate=eta,
                   min_data_in_leaf=min_data_in_leaf_val,
                  bagging_fraction = sub_sample,
                  feature_fraction = col_sample,
                  bagging_freq = bagging_freq,
                  bagging_seed = seed_val,
                  verbosity = silent_val,
                  num_rounds = rounds)

    model.fit(train_X, train_y)
    train_preds = model.predict_proba(train_X)[:,1]
    test_preds = model.predict_proba(test_X)[:,1]
    
    test_preds2 = 0
    if test_X2 is not None:
        test_preds2 = model.predict_proba(test_X2)[:,1]

    test_loss = 0
    if test_y is not None:
        train_loss = metrics.roc_auc_score(train_y, train_preds)
        test_loss = metrics.roc_auc_score(test_y, test_preds)
        print("Train and Test loss : ", train_loss, test_loss)
    return test_preds, test_loss, test_preds2, model

### Running Extra Trees  
def runET(train_X, train_y, test_X, test_y=None, test_X2=None, rounds=100, depth=20, 
          leaf=10, feat=0.2,min_data_split_val=2,seed_val=0,job = -1):
	model = ExtraTreesClassifier(
                    n_estimators = rounds,
					max_depth = depth,
					min_samples_split = min_data_split_val,
					min_samples_leaf = leaf,
					max_features =  feat,
					n_jobs = job,
					random_state = seed_val)
	model.fit(train_X, train_y)
	train_preds = model.predict_proba(train_X)[:,1]
	test_preds = model.predict_proba(test_X)[:,1]
	
	test_preds2 = 0
	if test_X2 is not None:
		test_preds2 = model.predict_proba(test_X2)[:,1]
	
	test_loss = 0
	if test_y is not None:
		train_loss = metrics.roc_auc_score(train_y, train_preds)
		test_loss = metrics.roc_auc_score(test_y, test_preds)
		print("Depth, leaf, feat : ", depth, leaf, feat)
		print("Train and Test loss : ", train_loss, test_loss)
	return test_preds, test_loss, test_preds2, model
 
### Running Random Forest
def runRF(train_X, train_y, test_X, test_y=None, test_X2=None, rounds=100, depth=20, leaf=10, 
          feat=0.2,min_data_split_val=2,seed_val=0,job = -1):
    model = RandomForestClassifier(
                    n_estimators = rounds,
                    max_depth = depth,
                    min_samples_split = min_data_split_val,
                    min_samples_leaf = leaf,
                    max_features =  feat,
                    n_jobs = job,
                    random_state = seed_val)
    model.fit(train_X, train_y)
    train_preds = model.predict_proba(train_X)[:,1]
    test_preds = model.predict_proba(test_X)[:,1]
    
    test_preds2 = 0
    if test_X2 is not None:
        test_preds2 = model.predict_proba(test_X2)[:,1]
    
    test_loss = 0
    if test_y is not None:
        train_loss = metrics.roc_auc_score(train_y, train_preds)
        test_loss = metrics.roc_auc_score(test_y, test_preds)
        print("Train and Test loss : ", train_loss, test_loss)
    return test_preds, test_loss, test_preds2, model

### Running Logistic Regression
def runLR(train_X, train_y, test_X, test_y=None, test_X2=None, C=1.0, penalty ='l1'):
    model = LogisticRegression(C=C, penalty=penalty, n_jobs=-1)
    model.fit(train_X, train_y)
    train_preds = model.predict_proba(train_X)[:,1]
    test_preds = model.predict_proba(test_X)[:,1]

    test_preds2 = 0
    if test_X2 is not None:
        test_preds2 = model.predict_proba(test_X2)[:,1]
    test_loss = 0
    
    train_loss = metrics.roc_auc_score(train_y, train_preds)
    test_loss = metrics.roc_auc_score(test_y, test_preds)
    print("Train and Test loss : ", train_loss, test_loss)
    return test_preds, test_loss, test_preds2, model

### Running Decision Tree
def runDT(train_X, train_y, test_X, test_y=None, test_X2=None, criterion='gini', 
          depth=None, min_split=2, min_leaf=1):
    model = DecisionTreeClassifier(
                                    criterion = criterion, 
                                    max_depth = depth, 
                                    min_samples_split = min_split, 
                                    min_samples_leaf=min_leaf)
    model.fit(train_X, train_y)
    train_preds = model.predict_proba(train_X)[:,1]
    test_preds = model.predict_proba(test_X)[:,1]

    test_preds2 = 0
    if test_X2 is not None:
        test_preds2 = model.predict_proba(test_X2)[:,1]
    
    test_loss = 0
    
    train_loss = metrics.roc_auc_score(train_y, train_preds)
    test_loss = metrics.roc_auc_score(test_y, test_preds)
    print("Train and Test loss : ", train_loss, test_loss)
    return test_preds, test_loss, test_preds2, model
    
### Running K-Nearest Neighbour
def runKNN(train_X, train_y, test_X, test_y=None, test_X2=None,
           neighbors=5, job = -1):
    model = KNeighborsClassifier(
                                n_neighbors=neighbors, 
                                n_jobs=job)
    model.fit(train_X, train_y)
    train_preds = model.predict_proba(train_X)[:,1]
    test_preds = model.predict_proba(test_X)[:,1]

    test_preds2 = 0
    if test_X2 is not None:
        test_preds2 = model.predict_proba(test_X2)[:,1]
    
    test_loss = 0
    
    train_loss = metrics.roc_auc_score(train_y, train_preds)
    test_loss = metrics.roc_auc_score(test_y, test_preds)
    print("Train and Test loss : ", train_loss, test_loss)
    return test_preds, test_loss, test_preds2, model

### Running SVM
def runSVC(train_X, train_y, test_X, test_y=None, test_X2=None, C=1.0, 
           kernel_choice = 'rbf'):
    model = SVC(
                C=C, 
                kernel=kernel_choice, 
                probability=True)
    model.fit(train_X, train_y)
    train_preds = model.predict_proba(train_X)[:,1]
    test_preds = model.predict_proba(test_X)[:,1]

    test_preds2 = 0
    if test_X2 is not None:
        test_preds2 = model.predict_proba(test_X2)[:,1]
    
    test_loss = 0
    
    train_loss = metrics.roc_auc_score(train_y, train_preds)
    test_loss = metrics.roc_auc_score(test_y, test_preds)
    print("Train and Test loss : ", train_loss, test_loss)
    return test_preds, test_loss, test_preds2, model
