import pandas as pd
import numpy as np
from processors import EmpBayesCluster, Preprocessor, parseInterest

df = pd.read_json('train.json')
df = df.set_index('listing_id')
testdf = pd.read_json('test.json')
testdf = testdf.set_index('listing_id')

preprocessor = Preprocessor()
preprocessor.fit(pd.concat([df.reset_index(), testdf.reset_index()]))

df = preprocessor.transform(df, False)
testdf = preprocessor.transform(testdf, False)



predictor_var = ['bathrooms', 'bedrooms', 'price', 'photo_cnt', 'active', 
                 'month', 'day', 'feature_cnt', 'latitude', 'longitude', 'hour',
                 'num_description_words', 'half_bath',  'manager_id_medium',
                 'manager_id_high', 'building_id', 'manager_id', 
                 'display_address', 'street_address', 'listing_id',
                'price/bed', 'price/bath', 'price/room', 'totalrooms']
outcome_var = 'interest_level'

def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0, num_rounds=2000, classes=3):
    
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.1
    param['max_depth'] = 6
    param['silent'] = 0
    param['num_class'] = classes
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 1
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['seed'] = seed_val

    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=250)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)

    pred_test_y = model.predict(xgtest, ntree_limit=model.best_iteration+1)
    return pred_test_y, model


train_df = df.copy()

dataset_X = df.copy().reset_index()
dataset_y = df['interest_level'].apply(parseInterest)



dataset_X['features_vec'] = dataset_X["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
testdf['features_vec'] = testdf["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
tfidf = CountVectorizer(stop_words='english', max_features=150)
tfidf.fit(list(dataset_X['features_vec']) + list(testdf['features_vec']))



dataset_X_trans = dataset_X.copy()
dataset_X_trans['manager_id_medium'] = 0
dataset_X_trans['manager_id_high'] = 0

means = dataset_X.groupby('interest_level').count()['bathrooms'].rename('count')/float(len(dataset_X))


np.random.seed(1900)
for i in ['medium', 'high']:
    for key in ['manager_id']:
        kf = model_selection.StratifiedKFold(n_splits=5)
        for dev_index, val_index in kf.split(np.zeros(len(dataset_X)), dataset_y):   
            encoder = EmpBayesCluster(key=key, means = means.to_dict())
            encoder.fit(dataset_X.iloc[dev_index], k = 5.0, f = 1.0)
            transformed_X = encoder.transform(dataset_X.iloc[val_index], tgts = ['medium', 'high'], noise=True)
            dataset_X_trans.loc[val_index, key+'_'+i] = transformed_X.loc[val_index,  key+'_'+i]
print '\nManager encoding is done. \n'
                     

dataset_X = dataset_X_trans.copy()
categorical = ['building_id', 'manager_id', 'display_address', 'street_address']

from sklearn.preprocessing import LabelEncoder
for f in categorical:
    encoder = LabelEncoder()
    encoder.fit(list(df[f]) + list(testdf[f])) 
    dataset_X[f] = encoder.transform(df[f].ravel())



from scipy import sparse
import xgboost as xgb
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss

cv_scores = []
kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=2016)
for dev_index, val_index in kf.split(np.zeros(len(dataset_X)), dataset_y):    
        train_X_pre, valid_X_pre = dataset_X.iloc[dev_index], dataset_X.iloc[val_index]
        
        tfidf.fit(train_X_pre['features_vec'])
        tr_sparse = tfidf.transform(train_X_pre['features_vec'])        
        va_sparse = tfidf.transform(valid_X_pre['features_vec'])
        print '\nFeature construction is done. \n'        

        
        train_X = sparse.hstack([train_X_pre[predictor_var], tr_sparse]).tocsr()
        valid_X = sparse.hstack([valid_X_pre[predictor_var], va_sparse]).tocsr()
        
        
        dev_X, val_X = train_X, valid_X
        dev_y, val_y = dataset_y.iloc[dev_index], dataset_y.iloc[val_index]    

        preds, model = runXGB(dev_X, dev_y, val_X, val_y, classes=3, seed_val=2)
        cv_scores.append(log_loss(val_y, preds))
        print(cv_scores)
        break


testdf = pd.read_json('test.json')
preprocessor = Preprocessor()
testdf = preprocessor.transform(testdf, False)
testset = testdf.copy()
testset['features_vec'] = testset["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
test_sparse = tfidf.transform(testset['features_vec'])
encoder = EmpBayesCluster(key='manager_id', means = means)
encoder.fit(dataset_X, k = 5, f = 1)
test_X_pre = encoder.transform(testset, tgts = ['medium', 'high'], noise=True)
encoder = EmpBayesCluster(key='building_id', means = means)
encoder.fit(dataset_X, k = 5, f = 1)
test_X_pre = encoder.transform(test_X_pre, tgts = ['medium', 'high'], noise=True)

test_X = sparse.hstack([test_X_pre[predictor_var], test_sparse]).tocsr()

prediction = model.predict(xgb.DMatrix(test_X), ntree_limit=model.best_iteration+1)

