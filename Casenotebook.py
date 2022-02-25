#!/usr/bin/env python
# coding: utf-8

# # Case Study

# We will begin by stating the libraries we will use and reading our dataset

# In[629]:


import pandas as pd
import os
import seaborn as sb
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from patsy import dmatrices
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve


# I will just remove the first, unnamed column and have a glipse of how the data look like.

# In[536]:


# Getting to know the data #
df = pd.read_csv("biological_data.csv")
df=df.drop('Unnamed: 0', 1)
df.head()


# In[537]:


df.dtypes


# In[538]:


df.describe()


# Here we notice a couple of things. Our response variable is an integer response variable and we aim to explain it with a mixed of continuous and non-numeric covariates.
# 
# We now proceed to understand how the variable we plan to predict behaves. And couple of things are to be noted:
#  
#  - More than 75% of the sample takes values below 5000
#  - About half of the sample takes values below 1000

# In[539]:


df.hist(column="Y",bins=30)
#df.boxplot(column="Y")


# Now, we focus on understanding the characteristics of the non-numeric explanatory variables.

# In[540]:


df[["X1","X2","X3"]].describe()


# Note:
# 
# - X1 has no variabbility. It is a 1-level factor. It has nothing to offer as a explanatory variable of the variation in Y.
# - X2 and X3 have 10 and 9 levels respectively. Are these too many levels? Do we have a way to learn from them and then group them?

# We now try to understand how both X2 and X3 interact with the response variable. As I do not have any prior knowledge about X2 and X3, I begin by regarding them as categorical variables and renaming the categories.

# In[541]:


number = LabelEncoder()
df['X2'] = number.fit_transform(df['X2'].astype('str'))
df['X3'] = number.fit_transform(df['X3'].astype('str'))
df['X2']=df['X2'].astype('category')
df['X3']=df['X3'].astype('category')


# In[542]:


df['X2'].value_counts()


# In[543]:


df['X3'].value_counts()


# In[544]:


g = sb.catplot(y='X2', x='Y', kind="box", data=df, height=8)
g = sb.catplot(y='X3', x='Y', kind="box", data=df, height=8)


# A quick inspection of the relation between X2 and Y let us see that: 
# 
#  - Level 2 has too few cases and could be assigned into one category
#  - Levels 0,6,7 and 8 span the whole range of Y and could be candidates for being regrupped
#  - Levels 3,5 and 9 behave similarly (low variability, take mostly low values of Y) and could make up another category
#  - Level 2 could be joined with level 1 into another level
#  
# Whereas from the relation between X3 and Y we get:
#  
#  - Levels 0 and 5 are the only levels that span the whole range of Y. Hence, they could be grouped into one category.
#    Level 2 could be possibly included into this category.
#  - Levels 3,4 and 8 can be grouped into one category
#  - Finally, levels 1 and 6 could be included into a third category.
# 
# Now, regarding the continuous explanatory variables, we now plot them in order to find out how they relate with each other and with Y and compute their Pearson coefficient of correlation.
# 

# In[545]:


sb.set_style('whitegrid')
g1 = sb.pairplot(df[['Y','X4','X5','X6']])
plt.show(g1)


# In[546]:


df.corr()


# We can notice some few thing from the figures:
# 
#  - There is perfect positive correlation between X4 and X5. Hence, we can get rid of one of them.
#  - There is negative association between X6 and both X4 and X5
#  - The correlation between X4,X5 and X6 and Y is low.
#  
# From the beginning of this case, we could notice that the behavior of Y is too "clusterized" and better understanding and prectictability of this variable could be reached if we transform it into a binary (or categorical?) variable. We now analyze this new variable in relation with the variables X2-X6

# In[547]:


import numpy as np
df['Ybin'] = np.digitize(df['Y'],bins=[50])


# In[548]:


X2_crosstab = pd.crosstab(df['X2'], 
                            df['Ybin'],
                                margins = False,
                         normalize="index")
X2_crosstab


# In[549]:


X3_crosstab = pd.crosstab(df['X3'], 
                            df['Ybin'],
                                margins = False,
                         normalize="index")
X3_crosstab


# We again plot the relation between the binary response variable and the threee continuous variables. 

# In[550]:


g1 = sb.pairplot(df[['Ybin','X4','X5','X6']])
plt.show(g1)


# We see no clear relation between X4-X6 and Y.
# 
# As we finish the descriptive and exploratory stage there are some actions we will take before we continue:
# 
#  - We will fit models with both Y and Y1 to enrich our discussion and show how differently they can be used.
#  - X1 won't be considered as a candidate variability as it has no variability and therefore will not explain the variation in Y.
#  - X2 and X3 will be grouped to reduce the number of categories and thus gain interpretability.
#  - X4 will be removed from the group of candidate variables as it is almost perfectly correlated with X5. 
#  - Both X5 and X6 will be standardized to avoid issues with the values of the coefficients.

# Based on the knowledge we have so far of our problem we attempt the next models for predicting Y in two scenarios, when Y is assumed numeric and when Y is assumed binary.
# 
# Scenario 1: Discrete response
# 
#  - Negative binomial regression
#  
# Scenario 2: Binary response
# 
#  - Logistic regression
#  - Neural network
# 
# Just before starting, we will create training and testing datasets, scale the continuous explanatory variables and transform the categorical ones.

# In[551]:


df.dtypes
X_1 = df[['X2','X3']]
X_1 = pd.get_dummies(data=X_1, drop_first=True)
X_2 = df[['X5','X6']]
scaler = StandardScaler()
X_2[['X5','X6']] = scaler.fit_transform(X_2[['X5','X6']])
X = pd.concat([X_1, X_2], axis=1)
X = sm.add_constant(X)
Y = df[['Y','Ybin']]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)


# In[ ]:





# In[552]:


NBModel = sm.GLM(y_train['Y'], X_train, family=sm.families.NegativeBinomial()).fit()
print(NBModel.summary())


# In[553]:


nb2_predictions = NBModel.get_prediction(X_test)


# In[554]:


predictions_summary_frame = nb2_predictions.summary_frame()
print(predictions_summary_frame)


# In[555]:


predicted_counts=predictions_summary_frame['mean']
actual_counts = y_test['Y']
fig = plt.figure()
fig.suptitle('Predicted versus actual values of Y')
plt.scatter(predicted_counts, actual_counts)
#actual, = plt.plot(X_test.index, actual_counts, 'ro-', label='Actual counts')
#plt.legend(handles=[predicted, actual])
plt.show()


# In[556]:


print('RMSE: ', np.sqrt(mean_squared_error(predicted_counts,actual_counts))) 


# Now we are ready to perform prediction of Y1 using both logistic regression and Neural Networks

# In[581]:


LRmodel = LogisticRegression(solver='liblinear', random_state=0)
LRmodel.fit(X_train,y_train['Ybin'])


# In[582]:


LRmodel.predict_proba(X_test)


# In[583]:


LRmodel.coef_


# In[584]:


# prepare models
scoring = 'accuracy'
kfold = model_selection.KFold(n_splits=10)
cv_results = model_selection.cross_val_score(LRmodel, X_train, y_train['Ybin'], cv=kfold, scoring=scoring)
#msg = "%s: %f (%f)" % ( cv_results.mean(), cv_results.std())
#print(msg)
# boxplot algorithm comparison
fig = plt.figure()
plt.boxplot(cv_results)
plt.show()


# In[585]:


cv_results


# In[603]:


LRpredict = LRmodel.predict(X_test)


# In[587]:


confusion_matrix(y_test['Ybin'], LRmodel.predict(X_test))


# In[588]:


print(classification_report(y_test['Ybin'], LRmodel.predict(X_test)))


# In[651]:


score = LRmodel.score(X_test, y_test['Ybin'])
score


# In[652]:


score = LRmodel.score(X_train, y_train['Ybin'])
score


# We finalize with a Neural Network model for predicting the output Ybin

# In[589]:


# define the keras model
NNmodel = Sequential()
NNmodel.add(Dense(16, input_dim=20, activation='relu'))
NNmodel.add(Dense(8, activation='relu'))
NNmodel.add(Dense(4, activation='relu'))
NNmodel.add(Dense(1, activation='sigmoid'))


# In[590]:


NNmodel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[591]:


mod_hist = NNmodel.fit(X_train,y_train['Ybin'], validation_split=0.25,epochs=200, batch_size=10)


# In[592]:


accuracy = NNmodel.evaluate(X_train, y_train['Ybin'])


# In[593]:


accuracy = NNmodel.evaluate(X_test, y_test['Ybin'])


# In[594]:


# summarize history for accuracy
plt.plot(mod_hist.history['accuracy'])
plt.plot(mod_hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(mod_hist.history['loss'])
plt.plot(mod_hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[621]:


NNy_pred = NNmodel.predict(X_test).ravel()
LRy_pred = LRmodel.predict_proba(X_test)[:,1]


# In[622]:


NN_fpr, NN_tpr, NN_thresholds = roc_curve(y_test['Ybin'], NNy_pred)
LR_fpr, LR_tpr, LR_thresholds = roc_curve(y_test['Ybin'], LRy_pred)
NNauc_keras = auc(NN_fpr, NN_tpr)
LRauc_keras = auc(LR_fpr, LR_tpr)
# plot the roc curve for the model
plt.plot(NN_fpr, NN_tpr, linestyle='--', label='NN')
plt.plot(LR_fpr, LR_tpr, marker='.', label='Logistic')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()
#plt.plot(nn_fpr_keras, nn_tpr_keras, marker='.', label='Neural Network (auc = %0.3f)' % auc_keras)


# In[626]:


nn_auc = roc_auc_score(y_test['Ybin'], NNy_pred)
lr_auc = roc_auc_score(y_test['Ybin'], LRy_pred)
print(nn_auc);print(lr_auc)


# In[648]:


LR_precision, LR_recall, LR_thresholds = precision_recall_curve(y_test['Ybin'], LRy_pred)
NN_precision, NN_recall, NN_threshold = precision_recall_curve(y_test['Ybin'], NNy_pred)
NNauc = auc(NN_recall, NN_precision)
LRauc = auc(LR_recall, LR_precision)
print(NNauc)
print(LRauc)


# In[647]:


# plot the roc curve for the model
plt.plot(NN_recall, NN_precision, linestyle='--', label='NN')
plt.plot(LR_recall, LR_precision, marker='.', label='Logistic')
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
# show the legend
plt.legend()
# show the plot
plt.show()
#plt.plot(nn_fpr_keras, nn_tpr_keras, marker='.', label='Neural Network (auc = %0.3f)' % auc_keras)


# In[645]:


auc(LR_recall, LR_precision)


# In[ ]:




