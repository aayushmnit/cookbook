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
from ensemble import ExtraTreesClassifier  ## ExtraTreesClassifier
import xgboost as xgb ## Xgboost for regression and classfication
import lightgbm as lgb ## Light GBM for regression and classification


## For classification
 
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
		return pred_test_y, loss, pred_test_y2
	else:
		return pred_test_y, loss, pred_test_y2

## Running LightGBM
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
		return pred_test_y, loss, pred_test_y2
	else:
		return pred_test_y, loss, pred_test_y2

## Running Extra Trees  
def runET(train_X, train_y, test_X, test_y=None, test_X2=None, depth=20, leaf=10, feat=0.2):
	model = ensemble.ExtraTreesClassifier(
			n_estimators = 100,
					max_depth = depth,
					min_samples_split = 2,
					min_samples_leaf = leaf,
					max_features =  feat,
					n_jobs = 6,
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
	return test_preds, test_loss, test_preds2