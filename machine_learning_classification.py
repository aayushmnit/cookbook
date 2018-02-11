"""
Created on Sat Feb 10 11:38:16 2018
@author: Aayush Agrawal
@Purpose - Re-usable code in Python 3 for cross-validation and machine learning in modeling process
"""

## Importing required libraries
import pandas as pd ## For DataFrame operation
import numpy as np ## Numerical python for matrix operations
from sklearn.model_selection import KFold, train_test_split ## Creating cross validation sets
from sklearn import metrics ## For loss functions

## Libraries for Classification algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from ensemble import ExtraTreesClassifier,RandomForestClassifier
import xgboost as xgb ## Xgboost for regression and classfication
import lightgbm as lgb ## Light GBM for regression and classification

####### Algorithms For Binary classification #########
 
### Running Xgboost
def create_feature_map(features):
	outfile = open('xgb.fmap', 'w')
	for i, feat in enumerate(features):
		outfile.write('{0}\t{1}\tq\n'.format(i,feat))
	outfile.close()

def runXGB(train_X, train_y, test_X, test_y=None, test_X2=None, feature_names=None, seed_val=0, rounds=500, dep=8, eta=0.05):
	params = {}
	params["objective"] = "binary:logistic"
	params['eval_metric'] = 'auc'
	params["eta"] = eta
	params["subsample"] = 0.7
	params["min_child_weight"] = 1
	params["colsample_bytree"] = 0.7
	params["max_depth"] = dep

	params["silent"] = 1
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

	if feature_names is not None:
		create_feature_map(feature_names)
		model.dump_model('xgbmodel.txt', 'xgb.fmap', with_stats=True)
		importance = model.get_fscore(fmap='xgb.fmap')
		importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True)
		imp_df = pd.DataFrame(importance, columns=['feature','fscore'])
		imp_df['fscore'] = imp_df['fscore'] / imp_df['fscore'].sum()
		imp_df.to_csv("imp_feat.txt", index=False)

	pred_test_y = model.predict(xgtest, ntree_limit=model.best_ntree_limit)
	pred_test_y2 = model.predict(xgb.DMatrix(test_X2), ntree_limit=model.best_ntree_limit)

	loss = 0
	if test_y is not None:
		loss = metrics.roc_auc_score(test_y, pred_test_y)
		return pred_test_y, loss, pred_test_y2, model
	else:
		return pred_test_y, loss, pred_test_y2, model

### Running LightGBM
def runLGB(train_X, train_y, test_X, test_y=None, test_X2=None, feature_names=None, seed_val=0, rounds=500, dep=8, eta=0.05):
	params = {}
	params["objective"] = "binary"
	params['metric'] = 'auc'
	params["max_depth"] = dep
	params["min_data_in_leaf"] = 20
	params["learning_rate"] = eta
	params["bagging_fraction"] = 0.7
	params["feature_fraction"] = 0.7
	params["bagging_freq"] = 5
	params["bagging_seed"] = seed_val
	params["verbosity"] = 0
	num_rounds = rounds

	plst = list(params.items())
	lgtrain = lgb.Dataset(train_X, label=train_y)

	if test_y is not None:
		lgtest = lgb.Dataset(test_X, label=test_y)
		model = lgb.train(params, lgtrain, num_rounds, valid_sets=[lgtest], early_stopping_rounds=100, verbose_eval=20)
	else:
		lgtest = lgb.DMatrix(test_X)
		model = lgb.train(params, lgtrain, num_rounds)

	pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
	pred_test_y2 = model.predict(test_X2, num_iteration=model.best_iteration)

	loss = 0
	if test_y is not None:
		loss = metrics.roc_auc_score(test_y, pred_test_y)
		print loss
		return pred_test_y, loss, pred_test_y2, model
	else:
		return pred_test_y, loss, pred_test_y2, model

### Running Extra Trees  
def runET(train_X, train_y, test_X, test_y=None, test_X2=None, depth=20, leaf=10, feat=0.2):
	model = ExtraTreesClassifier(
			n_estimators = 100,
					max_depth = depth,
					min_samples_split = 2,
					min_samples_leaf = leaf,
					max_features =  feat,
					n_jobs = 8,
					random_state = 0)
	model.fit(train_X, train_y)
	train_preds = model.predict_proba(train_X)[:,1]
	test_preds = model.predict_proba(test_X)[:,1]
	test_preds2 = model.predict_proba(test_X2)[:,1]
	test_loss = 0
	if test_y is not None:
		train_loss = metrics.roc_auc_score(train_y, train_preds)
		test_loss = metrics.roc_auc_score(test_y, test_preds)
		print "Depth, leaf, feat : ", depth, leaf, feat
		print "Train and Test loss : ", train_loss, test_loss
	return test_preds, test_loss, test_preds2, model
 
 ### Running Random Forest
 def runRF(train_X, train_y, test_X, test_y=None, test_X2=None, depth=20, leaf=10, feat=0.2):
    model = ensemble.RandomForestClassifier(
            n_estimators = 1000,
                    max_depth = depth,
                    min_samples_split = 2,
                    min_samples_leaf = leaf,
                    max_features =  feat,
                    n_jobs = 4,
                    random_state = 0)
    model.fit(train_X, train_y)
    train_preds = model.predict_proba(train_X)[:,1]
    test_preds = model.predict_proba(test_X)[:,1]
    test_preds2 = model.predict_proba(test_X2)[:,1]
    test_loss = 0
    
    train_loss = metrics.log_loss(train_y, train_preds)
    test_loss = metrics.log_loss(test_y, test_preds)
    print "Train and Test loss : ", train_loss, test_loss
    return test_preds, test_loss, test_preds2, model

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

 ### Running Logistic Regression
 def runLR(train_X, train_y, test_X, test_y=None, test_X2=None, C=1.0, penalty ='l1'):
    model = LogisticRegression(C=C, penalty=penalty, n_jobs=-1)
    model.fit(train_X, train_y)
    train_preds = model.predict_proba(train_X)[:,1]
    test_preds = model.predict_proba(test_X)[:,1]
    test_preds2 = model.predict_proba(test_X2)[:,1]
    test_loss = 0
    
    train_loss = metrics.log_loss(train_y, train_preds)
    test_loss = metrics.log_loss(test_y, test_preds)
    print "Train and Test loss : ", train_loss, test_loss
    return test_preds, test_loss, test_preds2, model

### Running Decision Tree
def runDT(train_X, train_y, test_X, test_y=None, test_X2=None, criterion='gini', depth=None, min_split=2, min_leaf=1):
    model = DecisionTreeClassifier(criterion = criterion, max_depth = depth, min_samples_split = min_split, min_samples_leaf=min_leaf)
    model.fit(train_X, train_y)
    train_preds = model.predict_proba(train_X)[:,1]
    test_preds = model.predict_proba(test_X)[:,1]
    test_preds2 = model.predict_proba(test_X2)[:,1]
    test_loss = 0
    
    train_loss = metrics.log_loss(train_y, train_preds)
    test_loss = metrics.log_loss(test_y, test_preds)
    print "Train and Test loss : ", train_loss, test_loss
    return test_preds, test_loss, test_preds2, model
    
### Running K-Nearest Neighbour
def runKNN(train_X, train_y, test_X, test_y=None, test_X2=None, neighbors=5):
    model = KNeighborsClassifier(n_neighbors=neighbors, n_jobs=-1)
    model.fit(train_X, train_y)
    train_preds = model.predict_proba(train_X)[:,1]
    test_preds = model.predict_proba(test_X)[:,1]
    test_preds2 = model.predict_proba(test_X2)[:,1]
    test_loss = 0
    
    train_loss = metrics.log_loss(train_y, train_preds)
    test_loss = metrics.log_loss(test_y, test_preds)
    print "Train and Test loss : ", train_loss, test_loss
    return test_preds, test_loss, test_preds2, model

### Running K-Nearest Neighbour
def runSVC(train_X, train_y, test_X, test_y=None, test_X2=None, C=1.0, kernel_choice = 'rbf'):
    model = SVC(C=C, kernel='kernel_choice', probability=True)
    model.fit(train_X, train_y)
    train_preds = model.predict_proba(train_X)[:,1]
    test_preds = model.predict_proba(test_X)[:,1]
    test_preds2 = model.predict_proba(test_X2)[:,1]
    test_loss = 0
    
    train_loss = metrics.log_loss(train_y, train_preds)
    test_loss = metrics.log_loss(test_y, test_preds)
    print "Train and Test loss : ", train_loss, test_loss
    return test_preds, test_loss, test_preds2, model

########### Cross Validation #######################
### 1) Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=46)
pred_val, loss, pred_test, model = runXGB(train_X = X_train, train_y = y_train,
                                          test_X = X_test, test_y = y_test,
                                          rounds=5000, dep=8)
print("K-Fold Loss = {0}".format(np.mean(loss_list)))

### 2) Cross-Validation (K-Fold)
cv = cross_validation.KFold(len(X), n_folds=5, shuffle=True, random_state=46)
loss_list = []
for traincv, testcv in cv:
    pred_val, loss, pred_test, model = runXGB(train_X = X.ix[traincv,:], train_y = y[traincv],
                                              test_X = X.ix[testcv,:], test_y = y[testcv],
                                              rounds=5000, dep=8)
    loss_list.append(loss)
print("K-Fold Loss = {0}".format(np.mean(loss_list)))

## Variable Importance plot
def feature_importance(model):
    """
    Plots the feature importance for an ensemble model
    """
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

