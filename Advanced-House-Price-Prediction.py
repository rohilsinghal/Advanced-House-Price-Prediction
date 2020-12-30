import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



pd.pandas.set_option('display.max_columns',None)
#this will display all the columns of the dataframe


data = pd.read_csv('train.csv')



data.shape
#this dataset has 1460 rows and 81 columns


data.head(10)


data.drop('Id',axis=1,inplace=True)


#getting the names of all the categorical features in the dataset
data_isnull = [features for features in data.columns if data[features].isnull().sum()>1 and data[features].dtypes=='O']


for features in data_isnull:
    print(features,np.round(data[features].isnull().mean(),4),'%')


for features in data_isnull:
    if np.round(data[features].isnull().mean(),4)>=0.80:
        data.drop([features],axis=1,inplace=True)
    else:
        data[features].fillna(data[features].mode(dropna=True)[0],inplace=True)


#getting the names of all the continues features in the dataset
data_isnull_con = [features for features in data.columns if data[features].isnull().sum()>1 and data[features].dtypes!='O']


for features in data_isnull_con:
    print(features,np.round(data[features].isnull().mean(),4),'%')



for features in data_isnull_con:
    if np.round(data[features].isnull().mean(),4)>=0.80:
        data.drop([features],axis=1,inplace=True)
    else:
        data[features].fillna(data[features].median(),inplace=True)


#checking to see if we have missed any missing value that needs to be handled.
sns.heatmap(data.isnull(),yticklabels=False)

data.head(5)

#Handling date time variables
for feature in ['YearBuilt','YearRemodAdd','GarageYrBlt']:
    data[feature]=data['YrSold']-data[feature]



#getting the names of all the categorical features in the dataset
categorical = [features for features in data.columns if data[features].dtypes=='O']
categorical


for feature in categorical:
    temp=data.groupby(feature)['SalePrice'].count()/len(data)
    temp_df=temp[temp>0.01].index
    data[feature]=np.where(data[feature].isin(temp_df),data[feature],'Rare_var')



for feature in categorical:
    labels_ordered=data.groupby([feature])['SalePrice'].mean().sort_values().index
    labels_ordered={k:i for i,k in enumerate(labels_ordered,0)}
    data[feature]=data[feature].map(labels_ordered)


data.head(10)



#handling numerical variable as some of them are sqewed
num_features=['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']
for feature in num_features:
    data[feature]=np.log(data[feature])


#Feature Scaling
feature_scale=[feature for feature in data.columns if feature not in ['SalePrice']]

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(data[feature_scale])



scaler.transform(data[feature_scale])

data.head()


data = pd.concat([data['SalePrice'].reset_index(drop=True),
                    pd.DataFrame(scaler.transform(data[feature_scale]), columns=feature_scale)],
                    axis=1)


## for feature selection

from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel


#dependent feature
y=data['SalePrice']


#drop dependent feature from dataset
X=data.drop('SalePrice',axis=1)

feature_sel_model = SelectFromModel(Lasso(alpha=0.005, random_state=0))
feature_sel_model.fit(X, y)


selected_feat = X.columns[(feature_sel_model.get_support())]

selected_feat


X=X[selected_feat]
X


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)


from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)
score


from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV

#hyperparameter tuning
xgb1 = XGBRegressor()
parameters = {'nthread':[4],
              'objective':['reg:linear'],
              'learning_rate': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ], #so called `eta` value
              'max_depth': [ 3, 4, 5, 6, 8, 10, 12, 15],
              'min_child_weight': [ 1, 3, 5, 7 ],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [ 0.3, 0.4, 0.5 , 0.7 ],
              'n_estimators': [500]}

xgb_grid = GridSearchCV(xgb1,
                        parameters,
                        cv = 2,
                        n_jobs = 5,
                        verbose=True)

xgb_grid.fit(X_train, y_train)


print(xgb_grid.best_score_)
print(xgb_grid.best_params_)
xgb_grid.best_estimator_


xgb1 = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=0.7, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.05, max_delta_step=0, max_depth=3,
             min_child_weight=1, monotone_constraints='()',
             n_estimators=500, n_jobs=4, nthread=4, num_parallel_tree=1,
             objective='reg:linear', random_state=0, reg_alpha=0, reg_lambda=1,
             scale_pos_weight=1, silent=1, subsample=0.7, tree_method='exact',
             validate_parameters=1, verbosity=None)


xgb1.fit(X_train, y_train)


y_pred = xgb1.predict(X_test)

from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)
print(score)


import pickle
pickle_out = open("xgb1.pkl","wb")
pickle.dump(xgb1,pickle_out)
pickle_out.close()
 