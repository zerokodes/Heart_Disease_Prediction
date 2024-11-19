#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np

import pickle

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# Parameters
C = 0.1
n_splits = 5
output_file = f'model_C={C}.bin'


# Data Preparation

df = pd.read_csv('Heart_Disease_Prediction.csv')

df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

sex_values = {
    0: "female",
    1: "male"
}
df.sex = df.sex.map(sex_values)

fbs_values = {
    0: "false",
    1: "true"
}
df.fbs_over_120 = df.fbs_over_120.map(fbs_values)

exercise_angina_values = {
    0: "no",
    1: "yes"
}
df.exercise_angina = df.exercise_angina.map(exercise_angina_values)

df.heart_disease = (df.heart_disease == "presence").astype(int)



df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)



numerical = ["age","bp","cholesterol","max_hr","st_depression"]

categorical = ['sex', 'chest_pain_type', 'fbs_over_120',
       'ekg_results', 'exercise_angina',
       'slope_of_st', 'number_of_vessels_fluro', 'thallium']


# Training

def train(df_train, y_train, C=0.1):
    dicts = df_train[categorical + numerical].to_dict(orient='records')
    
    dv = DictVectorizer (sparse=False)
    X_train = dv.fit_transform(dicts)
    
    model = LogisticRegression(C=C, max_iter = 1000)
    model.fit(X_train, y_train)
    return dv, model



def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')
    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:,1]
    
    return y_pred


# Validation

print(f'doing validation with C={C}')

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []

fold = 0
for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]

        y_train = df_train.heart_disease.values
        y_val = df_val.heart_disease.values

        dv, model = train(df_train, y_train, C=C)
        y_pred = predict(df_val, dv, model)
        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)

        print(f'auc on fold {fold} is {auc}')
        fold = fold + 1


print('validation results:')

print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))


# Trainig the final model

print('training the final model')

dv, model = train(df_full_train, df_full_train.heart_disease.values, C=0.1)
y_pred = predict(df_test, dv, model)

y_test = df_test.heart_disease.values
auc = roc_auc_score(y_test, y_pred)
auc

print(f'auc={auc}')


# Save the model

with open(output_file, 'wb') as f_out: 
    pickle.dump((dv, model), f_out)


print(f'model is saved to {output_file}')