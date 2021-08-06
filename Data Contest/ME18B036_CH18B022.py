import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error,make_scorer
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

# Loading data
dir=os.getcwd()
train_path=dir+'\\train.csv'
test_path=dir+'\\test.csv'
dummy=dir+'\\dummy_submission.csv'
save=dir+'\\save_for_later.csv'
s_labels=dir+'\\song_labels.csv'
so=dir+'\\songs.csv'

train=pd.read_csv(train_path)
test=pd.read_csv(test_path)
#sub=pd.read_csv(dummy)
songs=pd.read_csv(so)
song_labels=pd.read_csv(s_labels)
save_for_later=pd.read_csv(save)

# Preparing Dataset
song_labels=song_labels.loc[song_labels.groupby(['platform_id'])['count'].idxmax()]
save_for_later['Saved']=1
train_merged=pd.merge(train,songs,on='song_id',how='left')
train_merged=pd.merge(train_merged,song_labels,on='platform_id',how='left')
test_merged=pd.merge(test,songs,on='song_id',how='left')
test_merged=pd.merge(test_merged,song_labels,on='platform_id',how='left')
train_merged['Saved']=0
test_merged['Saved']=0
dat=pd.concat([train_merged,test_merged])
dat_m=pd.merge(dat,save_for_later,how='left',on=['customer_id','song_id'])
dat_m['Saved']=(dat_m['Saved_y'].fillna(0)+dat_m['Saved_x']).apply(lambda x:int(x))
dat_m.drop(['Saved_x','Saved_y'],axis=1,inplace=True)


#Preprocessing

cols=['customer_id','song_id','platform_id','language','label_id']
for col in cols:
  enc=LabelEncoder()
  dat_m[col]=enc.fit_transform(dat_m[col].astype(str))
cols=['language','label_id','count','number_of_comments','released_year','platform_id']
for col in cols:
  imp=KNNImputer(n_neighbors=10)
  dat_m[col]=imp.fit_transform(dat_m[[col]])
train_new=dat_m.iloc[:len(train)]
test_new=dat_m.iloc[len(train):]
test_new.drop(['score'],axis=1,inplace=True)

train_new['released_year']=train_new['released_year'].apply(lambda x: np.nan if x<1595 else x)
test_new['released_year']=test_new['released_year'].apply(lambda x: np.nan if x<1595 else x)

#Training and Prediction

X,y=train_new.drop(['score'],axis=1),train_new['score']
scorer=make_scorer(mean_squared_error)
fold=KFold(n_splits=5,random_state=101)

#params2={'reg_alpha': 0.8695646157740582, 'reg_lambda': 0.04000656260214025, 'colsample_bytree': 1.0, 'subsample': 1.0, 'learning_rate': 0.1, 'max_depth': 100, 'num_leaves': 758, 'min_child_samples': 96}
#params3={'reg_alpha': 0.006762373921002385, 'reg_lambda': 1.4229368717799058, 'colsample_bytree': 0.7, 'subsample': 1.0, 'learning_rate': 0.2, 'max_depth': 128, 'num_leaves': 453, 'min_child_samples': 168, 'cat_smooth': 5}
params4={'reg_alpha': 9.456718690047266, 'reg_lambda': 1.6740889682762798, 'colsample_bytree': 0.5, 'subsample': 1.0, 'learning_rate': 0.07442098625426513, 'max_depth': 100, 'num_leaves': 453, 'min_child_samples': 196, 'bagging_freq': 47, 'max_bin': 157, 'cat_smooth': 8}
preds,models=[],[]
for train_idx,test_idx in fold.split(X,y):
  X_train,y_train=X.iloc[train_idx],y[train_idx]
  X_val,y_val=X.iloc[test_idx],y[test_idx]
  model=LGBMRegressor(objective='regression',**params4,random_state=101,min_data_per_group=1,num_boost_round=1000,early_stopping_rounds=30)
  model.fit(X_train,y_train,categorical_feature=['customer_id','song_id','platform_id','language','label_id','Saved'],eval_set=[(X_val,y_val)],verbose=100)
  models.append(model)
p1=np.zeros(len(test_new))
for model in models:
  p=model.predict(test_new)
  p1+=p
preds=p1/5
preds=np.clip(preds,1,5)

#Submission
row_ids = list(range(len(test_new)))
rows = pd.DataFrame(row_ids, columns = ['test_row_id'])
my_sub = pd.DataFrame( preds,  columns = ['score'])
preds_21 = pd.concat([rows, my_sub], axis =1)
preds_21.to_csv('predictions.csv', index= False)
#sub['score']=preds
#sub.to_csv('Predictions.csv',index=False)
