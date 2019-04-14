import numpy as np
import random
import pandas as pd
import gzip
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline 

#################################

#load data

n = 40428967  #total number of records in the clickstream data 
sample_size = 100000
skip_values = sorted(random.sample(range(1,n), n-sample_size))
parse_date = lambda val : pd.datetime.strptime(val, '%y%m%d%H')
with gzip.open('avazu/train.gz') as f:
    train = pd.read_csv(f, parse_dates = ['hour'], date_parser = parse_date, skiprows = skip_values)


train = train.rename(columns={'hour': 'date'})

#which hours have the most clicks 
train['hour'] = train.date.apply(lambda x: x.hour)
train['day_of_week'] = train['date'].apply(lambda val: val.weekday_name)

target = train['click']

analysis = train
analysis.drop('date', axis=1, inplace=True)
analysis.drop('id', axis=1, inplace=True)
analysis.drop('click', axis=1, inplace=True)



#Hashing

def convert_obj_to_int(self):
    
    object_list_columns = self.columns
    object_list_dtypes = self.dtypes
    new_col_suffix = '_int'
    for index in range(0,len(object_list_columns)):
        if object_list_dtypes[index] == object :
            self[object_list_columns[index]+new_col_suffix] = self[object_list_columns[index]].map( lambda  x: hash(x))
            self.drop([object_list_columns[index]],inplace=True,axis=1)
    return self
analysis = convert_obj_to_int(analysis)


#split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(analysis, target, test_size = 0.2, random_state =0 )


print('parameters to check')
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [10,50,100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1,20,50]
# Method of selecting samples for training each tree
bootstrap = [True]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)





# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 10 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=3, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)

rf_random.best_params_


print rf_random.best_params_