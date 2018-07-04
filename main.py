# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 16:43:32 2018

@author: Faris Mismar
"""

import keras

from keras.models import Sequential
from keras.layers import Dense

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import os
import sys

os.chdir('/Users/farismismar/Desktop/FIFA')

# Check if tensorflow is used
if (keras.backend.backend() != 'tensorflow' and keras.backend.image_data_format() != 'channels_last' and keras.backend.image_dim_ordering() != 'tf'):
    print('Install tensorflow, configure keras.json to include channels_last for image format and tf for image dimension ordering.')
    print('Program will now exit.')
    sys.exit(1)
    
# Set the random seed
seed = 123
np.random.seed(seed)

# Import the datafile to memory first
dataset = pd.read_csv('./Dataset/results.csv')

# Sanity check. Missing values?
print('Number of missing values: {}'.format(dataset.isnull().sum().sum()))

# Filter what is really needed
df = dataset.loc[dataset['tournament'] == 'FIFA World Cup']

df['date'] = pd.to_datetime(df['date'])
df['Year'] = df['date'].dt.year
df = df.drop(['date', 'tournament', 'city', 'country', 'neutral'], axis=1)

# Generate more features X for the data
# The data is arranged such that the last year set has the final match, so let's find that first
# Replace NaN with zero.  Why?
df['Is_Final'] = df['Year'].diff(periods=-1).fillna(value=0) != 0

condition = (df['home_score'] > df['away_score'])
df.loc[condition, 'Winner'] = df['home_team']
df.loc[~condition, 'Winner'] = df['away_team']

# Drop rows where the winners are one of the losing teams
df = df.loc[(df['Winner'] == 'Uruguay') | (df['Winner'] == 'France') | (df['Winner'] == 'Brazil') | (df['Winner'] == 'Belgium') | (df['Winner'] == 'Russia')
| (df['Winner'] == 'Croatia') | (df['Winner'] == 'Sweden') | (df['Winner'] == 'England') ] 

####################################################################################
#  Approach 0: Just sum and average
####################################################################################
denominator = df['Winner'].value_counts().sum()

winner = df['Winner'].value_counts() / denominator

print(winner.idxmax())

########################################################################################################
#  Approach 1: NN with n-Step Prediction (eight games are left from the dataset including until Jul 3)
########################################################################################################
look_forward = 8

## Let us now drop all data that is not WC finalist
df1 = df[df['Is_Final'] == True].reset_index()

# Now shift the data by one to predict the next season.
df1 = df.drop(['Is_Final'], axis=1).reset_index().drop(['index'], axis=1)
df1['Winner+n'] = df1['Winner'].shift(-look_forward)

df1 = df1.fillna(value='Antarctica') # This is a sentinel.  We now have 9 teams.

# integer encode
teams=df['home_team'].tolist()
teams.append('Antarctica')

for team in df['away_team']:
    teams.append(team)

teams = list(set(teams)) # get rid of duplicates

label_encoder = LabelEncoder()
label_encoder.fit(teams)
integer_encoded = label_encoder.transform(df1['Winner+n'])

df1['Winner+n'] = integer_encoded
df1['Winner'] = label_encoder.transform(df1['Winner'])

df1['dWinner'] = df1['Winner'].diff(-look_forward)

# Impute by last value
df1['dWinner'] = df1['dWinner'].fillna(df1['dWinner'].iloc[-look_forward - 1])

# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(-1, 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
Y = pd.DataFrame(onehot_encoded, dtype=int)

df1['home_team'] = label_encoder.transform(df1['home_team'])
df1['away_team'] = label_encoder.transform(df1['away_team'])

# Now convert the problem to a multi-class problem
X = df1.drop(['Winner+n'], axis=1)

# Now prepare the data for a NN
df1 = pd.concat([X, Y], axis=1)

# Set the index to equal the year
df1.index = df1['Year'].astype(int)
df1 = df1.drop(['Year'], axis=1)

# Perform a split
m, n = df1.shape
rsplit = 0.815 # trial and error to naturally split by end of 2006.

# split into train and test sets
train_size = int(rsplit * m)
test_size = m - train_size

train, test = df1.iloc[0:train_size,:], df1.iloc[train_size:m,:]

X_train = train.iloc[:,:6]
Y_train = train.iloc[:,6:]
X_test = test.iloc[:,:6]
Y_test = test.iloc[:,6:]

mX, nX = X_train.shape
mY, nY = Y_train.shape

# Scale features for NN
ss = StandardScaler()

X_train_sc = ss.fit_transform(X_train)
X_test_sc = ss.transform(X_test)

# create model
def create_mlp(optimizer, output_dim, n_hidden, act):
    mlp = Sequential()
    mlp.add(Dense(units=output_dim, input_dim=nX, activation=act))
    for k in np.arange(n_hidden):
        mlp.add(Dense(output_dim, use_bias=True))

    mlp.add(Dense(units=nY, input_dim=output_dim, activation=act))

    mlp.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])  # metrics really mean nothing here.  Focus on loss.
    return mlp

model = KerasClassifier(build_fn=create_mlp, verbose=1, epochs=10, batch_size=16)

# The hyperparameters
optimizers = ['sgd', 'adam']
output_dims=[1,2,3]
activations = ['relu', 'softmax', 'sigmoid']
n_hiddens = [3,5]

hyperparameters = dict(optimizer=optimizers, output_dim=output_dims, n_hidden=n_hiddens, act=activations)

grid = GridSearchCV(estimator=model, param_grid=hyperparameters, n_jobs=1, cv=3)
grid_result = grid.fit(X_train_sc, Y_train)

# This is the best model: {'act': 'softmax', 'n_hidden': 5, 'optimizer': 'adam', 'output_dim': 2}
best_model_mlp = grid_result.best_params_
print(best_model_mlp)

clf = grid_result.best_estimator_
Y_score_mlp = clf.predict_proba(X_test_sc)
Y_pred_mlp = clf.predict(X_test_sc)  # must look like Y_test

Y_pred_mlp = onehot_encoder.transform(Y_pred_mlp.reshape(-1, 1))
                
def decode_onehot(X):
    teams = []
    # This is obtained through hand derivation.  Quite disappointing we cannot revert with a built-in function.
    t = ['Uruguay', 'Sweden', 'Russia', 'France', 'England', 'Croatia', 'Brazil', 'Belgium']
    for i in range(X.shape[0]):
        X_i = X[i, :]
        if sum(X_i) == 0:
            teams.append('(unknown)')
        else:
            teams.append(t[X_i.argmax()])

    teams = pd.DataFrame(data = {'Team': teams})
    return teams

Y_pred = decode_onehot(Y_pred_mlp)

# Compute ROC curve and ROC area for each class.
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(nY):
    fpr[i], tpr[i], _ = roc_curve(Y_test.iloc[:, i], Y_score_mlp[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

classes = list(set(df1['Winner']))
classes.append(label_encoder.transform(['Antarctica'])[0])
classes = label_encoder.inverse_transform(classes)

plt.figure(figsize=(13,8))
lw = 2
for i in range(nY):
    class_i = classes[i]
    plt.plot(fpr[i], tpr[i], 
             lw=lw, label='ROC curve for class {0} (AUC = {1:.6f})'.format(class_i, roc_auc[i]))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.grid()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc='lower right')
plt.savefig('mlp_roc.pdf', format='pdf')
plt.show()

Y_pred.index = Y_test.index

'''
df1.index[14:], label_encoder.inverse_transform(Y_pred_mlp), 

label_encoder.inverse_transform(Y_test.dot(onehot_encoder.active_features_).astype(int))  # https://stackoverflow.com/questions/22548731/how-to-reverse-sklearn-onehotencoder-transform-to-recover-original-data
'''