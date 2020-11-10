import pandas as pd
import plotly.express as px 
import plotly.io as pio
pio.renderers.default='browser'
from sklearn.metrics import roc_auc_score
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

# read data train 
train = train = pd.read_csv(
    "C://Users//aigerimb//Desktop//riid_prediction_competition//train.csv",
    usecols=[1, 2, 3, 4, 5, 7, 8, 9],
    dtype={
        'timestamp': 'int64',
        'user_id': 'int32',
        'content_id': 'int16',
        'content_type_id': 'int8',
        'task_container_id': 'int16',
        'answered_correctly':'int8',
        'prior_question_elapsed_time': 'float32',
        'prior_question_had_explanation': 'boolean'
    }
)

questions = pd.read_csv(
    'C://Users//aigerimb//Desktop//riid_prediction_competition//questions.csv',                         
    usecols=[0, 3],
    dtype={
        'question_id': 'int16',
        'part': 'int8'}
)

lectures = pd.read_csv('C://Users//aigerimb//Desktop//riid_prediction_competition//lectures.csv')
# replace strings 
lectures['type_of'].replace('solving question', 'solving_question', inplace=True)

# feature generation from lectures.csv 
#convert categorical variables to one-hot encoding 
lectures = pd.get_dummies(lectures, columns=['part', 'type_of'])
lectures_part_col = [cols for cols in lectures.columns if cols.startswith('part')]
lectures_type_of_col = [cols for cols in lectures.columns if cols.startswith('type_of')]
train_lectures = train[train['content_type_id']==1].merge(lectures, how='left', left_on='content_id', right_on='lecture_id')

# how many times student watched different lectures by part and type 
user_stats_per_lecture = train_lectures.groupby('user_id')[lectures_part_col+lectures_type_of_col].sum()

del(train_lectures)


# drop = True to avoid adding the old index 
train = train[train.content_type_id == False].sort_values('timestamp').reset_index(drop = True)
train[(train['content_type_id']==0)]['task_container_id'].nunique() #saving value to fillna
elapsed_mean = train.prior_question_elapsed_time.mean()


# how many times in average user answerred correctly 
results_u_final = train.loc[(train['content_type_id']==0, ['user_id', 'answered_correctly'])].groupby(['user_id']).agg(['mean'])
results_u_final.columns = ['answerred_correctly_user']
results_u3_final = train.loc[(train['content_type_id']==0, ['user_id', 'answered_correctly'])].groupby(['user_id']).agg(['count'])
results_u3_final.columns = ['total_q_user']

# average of explantions for questions per user 
results_u2_final = train.loc[train.content_type_id == False, ['user_id','prior_question_had_explanation']].groupby(['user_id']).agg(['mean'])
results_u2_final.columns = ['explanation_mean_user']

train = pd.merge(train, questions, left_on = 'content_id', right_on = 'question_id', how = 'left')
results_q_final = train.loc[train.content_type_id == False, ['question_id','answered_correctly']].groupby(['question_id']).agg(['mean'])
results_q_final.columns = ['quest_pct']

question2 = pd.merge(questions, results_q_final, left_on = 'question_id', right_on = 'question_id', how = 'left')

prior_mean_user = results_u2_final.explanation_mean_user.mean()
train.drop(['timestamp', 'content_type_id', 'question_id', 'part'], axis=1, inplace=True)


# creating validation set:
validation = train.groupby('user_id').tail(5)
train = train[~train.index.isin(validation.index)]
len(train) + len(validation)


results_u_val = train[['user_id','answered_correctly']].groupby(['user_id']).agg(['mean'])
results_u_val.columns = ['answered_correctly_user']

results_u2_val = train[['user_id','prior_question_had_explanation']].groupby(['user_id']).agg(['mean'])
results_u2_val.columns = ['explanation_mean_user']

results_u3_val = train[['user_id','answered_correctly']].groupby(['user_id']).agg(['count'])
results_u3_val.columns = ['total_q_user']

X = train.groupby('user_id').tail(18)
train = train[~train.index.isin(X.index)]

results_u_X = train[['user_id','answered_correctly']].groupby(['user_id']).agg(['mean'])
results_u_X.columns = ['answered_correctly_user']

results_u2_X = train[['user_id','prior_question_had_explanation']].groupby(['user_id']).agg(['mean'])
results_u2_X.columns = ['explanation_mean_user']

results_u3_X = train[['user_id','answered_correctly']].groupby(['user_id']).agg(['count'])
results_u3_X.columns = ['total_q_user']

#clearing memory
del(train)

#X = pd.merge(X, group3, left_on=['task_container_id'], right_index= True, how="left")
X = pd.merge(X, results_u_X, on=['user_id'], how="left")
X = pd.merge(X, results_u2_X, on=['user_id'], how="left")
X = pd.merge(X, results_u3_X, on=['user_id'], how="left")
X = pd.merge(X, user_stats_per_lecture, on=['user_id'], how="left")


#validation = pd.merge(validation, group3, left_on=['task_container_id'], right_index= True, how="left")
validation = pd.merge(validation, results_u_val, on=['user_id'], how="left")
validation = pd.merge(validation, results_u2_val, on=['user_id'], how="left")
validation = pd.merge(validation, results_u3_val, on=['user_id'], how="left")
validation = pd.merge(validation, user_stats_per_lecture, on=['user_id'], how="left")


from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()

X.prior_question_had_explanation.fillna(False, inplace = True)
validation.prior_question_had_explanation.fillna(False, inplace = True)

validation["prior_question_had_explanation_enc"] = lb_make.fit_transform(validation["prior_question_had_explanation"])
X["prior_question_had_explanation_enc"] = lb_make.fit_transform(X["prior_question_had_explanation"])

content_mean = question2.quest_pct.mean()

question2.quest_pct.mean()


X = pd.merge(X, question2, left_on = 'content_id', right_on = 'question_id', how = 'left')
validation = pd.merge(validation, question2, left_on = 'content_id', right_on = 'question_id', how = 'left')
X.part = X.part - 1
validation.part = validation.part - 1

y = X['answered_correctly']
X = X.drop(['answered_correctly'], axis=1)
X.head()

y_val = validation['answered_correctly']
X_val = validation.drop(['answered_correctly'], axis=1)


X = X[['answered_correctly_user', 'explanation_mean_user', 'quest_pct', 'total_q_user',
       'prior_question_elapsed_time','prior_question_had_explanation_enc', 'part',
       'part_1', 'part_2', 'part_3', 'part_4', 'part_5', 'part_6', 'part_7',
       'type_of_concept', 'type_of_intention', 'type_of_solving_question', 'type_of_starter']]
X_val = X_val[['answered_correctly_user', 'explanation_mean_user', 'quest_pct', 'total_q_user', 
               'prior_question_elapsed_time','prior_question_had_explanation_enc', 'part',
               'part_1', 'part_2', 'part_3', 'part_4', 'part_5', 'part_6', 'part_7',
               'type_of_concept', 'type_of_intention', 'type_of_solving_question', 'type_of_starter']]


# Filling with 0.5 for simplicity; there could likely be a better value
X['answered_correctly_user'].fillna(0.65,  inplace=True)
X['explanation_mean_user'].fillna(prior_mean_user,  inplace=True)
X['quest_pct'].fillna(content_mean, inplace=True)

X['part'].fillna(4, inplace = True)
X['total_q_user'].fillna(X['total_q_user'].mean(), inplace = True)
X['prior_question_elapsed_time'].fillna(elapsed_mean, inplace = True)
X['prior_question_had_explanation_enc'].fillna(0, inplace = True)

X['part_1'].fillna(0, inplace = True)
X['part_2'].fillna(0, inplace = True)
X['part_3'].fillna(0, inplace = True)
X['part_4'].fillna(0, inplace = True)
X['part_5'].fillna(0, inplace = True)
X['part_6'].fillna(0, inplace = True)
X['part_7'].fillna(0, inplace = True)
X['type_of_concept'].fillna(0, inplace = True)
X['type_of_intention'].fillna(0, inplace = True)
X['type_of_solving_question'].fillna(0, inplace = True)
X['type_of_starter'].fillna(0, inplace = True)

X_val['answered_correctly_user'].fillna(0.65,  inplace=True)
X_val['explanation_mean_user'].fillna(prior_mean_user,  inplace=True)
X_val['quest_pct'].fillna(content_mean,  inplace=True)

X_val['part'].fillna(4, inplace = True)
X_val['total_q_user'].fillna(X['total_q_user'].mean(), inplace = True)
X_val['prior_question_elapsed_time'].fillna(elapsed_mean, inplace = True)
X_val['prior_question_had_explanation_enc'].fillna(0, inplace = True)

X_val['part_1'].fillna(0, inplace = True)
X_val['part_2'].fillna(0, inplace = True)
X_val['part_3'].fillna(0, inplace = True)
X_val['part_4'].fillna(0, inplace = True)
X_val['part_5'].fillna(0, inplace = True)
X_val['part_6'].fillna(0, inplace = True)
X_val['part_7'].fillna(0, inplace = True)
X_val['type_of_concept'].fillna(0, inplace = True)
X_val['type_of_intention'].fillna(0, inplace = True)
X_val['type_of_solving_question'].fillna(0, inplace = True)
X_val['type_of_starter'].fillna(0, inplace = True)

import lightgbm as lgb
params = {
    'objective': 'binary',
    'seed': 42,
    'metric': 'auc',
    'learning_rate': 0.075,
    'max_bin': 800,
    'num_leaves': 80
}

lgb_train = lgb.Dataset(X, y, categorical_feature = ['part', 'prior_question_had_explanation_enc'])
lgb_eval = lgb.Dataset(X_val, y_val, categorical_feature = ['part', 'prior_question_had_explanation_enc'], reference=lgb_train)

model = lgb.train(
    params,
    lgb_train,
    valid_sets=[lgb_train, lgb_eval],
    verbose_eval=100,
    num_boost_round=10000,
    early_stopping_rounds=10
)

y_pred = model.predict(X_val)
y_true = np.array(y_val)
roc_auc_score(y_true, y_pred)