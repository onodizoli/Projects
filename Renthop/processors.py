
import pandas as pd
import numpy as np

    
            
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin



def parseInterest(interest):
    if interest == 'low':
        return 0
    elif interest == 'medium':
        return 1
    elif interest == 'high':
        return 2
    else:
        return interest
            
class EmpBayesCluster(BaseEstimator, TransformerMixin):
    '''
    Class for encoding high cardinality categorical data, implementing
    the paper: 
    http://helios.mm.di.uoa.gr/~rouvas/ssi/sigkdd/sigkdd.vol3.1/barreca.ps
    It can be used in scikit-learn pipelines
    '''    
    
    def __init__(self, key = 'manager_id', tgts = ['low', 'medium', 'high'], 
                 outcome_var='interest_level', means = None, 
                 exclude_top = False):
        '''
        Arguments:
        ----------    
        key: categorical variable to encode
        tgts: states of outcome variable
        outcome_var: outcome variable of the model
        means: prior distibution of outcome variable, if None, the given dataset
                is used to compute
        exclude_top: if True, the prior is computed by excluding keys with 
                high frequency        
        '''

        self.outcome_var = outcome_var
        
        self.key = key
        self.tgts = tgts
    
        self.means = means
        self.means_dir = {i: 0 for i in self.tgts}
        
        if means != None:
            self.means_dir = means      
        
        self.exclude_top = exclude_top

           
    def fit(self, df, k = 12.0, f = 0.5):
        '''
        Fitting the encoder on the data, calculating the values which should be
            assigned to each instance of the category
        ---------- 
        
        Arguments:
        ----------
        df: dataset    
        k: parameter of weighing function, # of examples needed for an even 
            split between prior and empiric distribution
        f: parameter of weighing function, controlling transition rate
        '''
        
        key_vc = df[self.key].value_counts()
        
        if  self.means == None:
            self.means = df[self.outcome_var].value_counts()
            self.means /= float(len(df))
               
            for i in self.means.index:
                 self.means_dir[i] = self.means[i]        
        
        
            if self.exclude_top:
                ids = pd.Series(key_vc[key_vc < (k-1)].index).rename('ids')
                ids = ids.to_frame()
                ids['dummy']=True
                filtered = pd.merge(left = df[[self.key,self.outcome_var]],
                                    right = ids, how = 'left',
                                    left_on = self.key,
                                    right_on = ids.ids).dropna()
                self.means = filtered[self.outcome_var].value_counts()
                self.means /= float(len(filtered))
                for i in self.means.index:
                    self.means_dir[i] = self.means[i]
                
                
        self.value_table = key_vc.rename('count')
        
        key_remap = df[self.key]
        if self.group_singles:
            remap_singletons = {i:'0' for i in key_vc[key_vc == 1].index}        
            key_remap = df[self.key].replace(remap_singletons)        
            key_rvc = key_remap.value_counts()
            self.value_table = key_rvc.rename('count') 
            

        self.value_table = self.value_table.to_frame()
        for i in self.tgts:            
            self.value_table[self.key + '_' + i] = 0.0

        

        for i in self.value_table.index:
            n = self.value_table.loc[i]['count']
            l = (1.0 / (1.0 + np.exp((-(n - k) / f))))
    
            df_subset = df[key_remap == i]
            
            subset_mean = df_subset[self.outcome_var].value_counts()
            subset_mean /= float(len(df_subset))
            
            subset_means_dir = {m: 0.0 for m in self.tgts}
            
            for m in subset_mean.index:
                subset_means_dir[m] = subset_mean[m]
            
            
            for m in self.tgts:
                x = l * subset_means_dir[m] + (1-l) * self.means_dir[m]
                self.value_table.set_value(i, self.key + '_' + m, x)    

        return self
    
    def transform(self, df, tgts = ['medium', 'high'], r_k=0.01, 
                  noise = False):
        '''
        Return the dataset with the new encoded features
        ----------
        
        Arguments:
        ----------    
        df: dataset
        tgts: values of the outcome variable you want to encode,
            typically len(outcome_var)-1
        noise: if True, uniform noise is added to output for more robustness
        r_k: controls the noise
        '''
        
        
        tgts_rename = [self.key + '_' + i for i in tgts]
        df = pd.merge(left = df, right = self.value_table[tgts_rename],
                      how = 'left', left_on = self.key, right_index = True)
        for i in tgts_rename:
            if self.group_singles:
                df[i].fillna(self.value_table.loc['0'][i], inplace = True)
            else:
                df[i].fillna(self.means_dir[i.split('_')[-1]], inplace = True)
            if noise:
                df[i] = df[i].apply(lambda row:
                    row *(1 + (np.random.uniform() - 0.5) * r_k))        
        return df
        
class Preprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocessing the dataset and creating features. The function is
    usable in scikit-learn pipelines.

    """
    
    def __init__(self):

        self.categorical_features = {}
        
  
    def _parseInterest(self, interest):
        if interest == 'low':
            return 0
        elif interest == 'medium':
            return 1
        else:
            return 2
        
    def fit(self, df, features = ['manager_id', 'building_id']):
        """ 
        Get categorical features with a single occurance for encoding them 
        with -1 in the future
        
        Arguments
        ----------
        df: dataset 
            
        features: list of categorical features
        """

        for feature in features:
            temp = (df.groupby(feature, as_index = True)
                    .count()['bathrooms'].rename('count'))
            table = temp[temp == 1]
            self.categorical_features[feature] = table         
                
        return self
        
    def transform(self, X, train = True):
        """
        Creating new features
        
        Parameters
        ----------
        X: dataset
        train: if True, the outcome variable is note encoded
        """        
        X['photo_cnt'] = X['photos'].apply(len)
        X['created'] = pd.to_datetime(X['created'])
        mindate = min(X['created'].dt.date)
        X['active'] = (X['created'].dt.date - mindate).dt.days
        X['month'] = X['created'].dt.month
        X['day'] = X['created'].dt.day
        X['hour'] = X['created'].dt.hour
        X['minute'] = X['created'].dt.minute
        X['feature_cnt'] = X['features'].apply(len)
        
        X['price/bed'] = X['price']/(X['bedrooms']
            .apply(lambda x: 1 if x ==0 else x))
        X['price/bath'] = X['price']/X['bathrooms']
        X['price/room'] = X['price']/(X['bedrooms']+X['bathrooms'])
        X['totalrooms'] = X['bedrooms']+X['bathrooms']
        X['half_bath'] = (X['bathrooms']
            .apply(lambda x: np.ceil(x)-np.floor(x)))
        
        X['num_description_words'] = (X['description']
            .apply(lambda x: len(x.split(" "))))
        
        X['display_address'] = (X['display_address']
            .apply(lambda x: x.lower().strip()))
        X['street_address'] = (X['street_address']
            .apply(lambda x: x.lower().strip()))
        
        
        # Transform features with a single instance
        for i in self.categorical_features.keys():
            X.loc[X[i].isin(self.categorical_features[i].ravel()), i] = "-1"

        
        if train:
            X['interest_level'] = X['interest_level'].apply(self._parseInterest)
            X['interest_1'] = X['interest_level']
            X['interest_2'] = X['interest_level']        
            X.loc[X['interest_level']==2,'interest_1'] = 1
            X.loc[X['interest_level']==2,'interest_2'] = 0
        
        return X
