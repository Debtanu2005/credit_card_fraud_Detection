#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv("Fraud.csv")
df


# In[4]:


df.describe()


# In[5]:


import re


# In[6]:


df["nameDest"] = df["nameDest"].apply(lambda x: re.split("C", x)[0] )


# In[7]:


df


# In[8]:


df.isnull().mean()*100


# In[22]:


fraud_amounts=df[df["isFraud"] == 0]["amount"]


# In[166]:


sns.histplot(df["isFraud"])


# In[23]:


sns.distplot(df["isFlaggedFraud"])


# In[24]:


fraud_amounts


# In[25]:


df[df["isFraud"] == 0]["newbalanceOrig"]


# In[26]:


df[df["isFraud"] == 0]["oldbalanceOrg"]


# In[27]:


plt.subplots(figsize = (5,5))
plt.scatter( df['newbalanceOrig'],df["isFraud"], color='blue', alpha=0.5)
# plt.xlim(600,850)
# plt.ylim(0,900)
# plt.legend(loc = 1)
plt.show()


# In[28]:


plt.subplots(figsize = (5,5))
plt.scatter( df['amount'],df["isFraud"], color='red', alpha=0.5)
# plt.xlim(600,850)
# plt.ylim(0,900)
# plt.legend(loc = 1)
plt.show()


# In[95]:


df["difInbal"]= df["oldbalanceOrg"]-df["newbalanceOrig"]
df["dif_bal_dest"]= df["oldbalanceDest"]- df["newbalanceDest"]


# In[97]:


plt.subplots(figsize = (5,5))
plt.scatter( df['difInbal'],df["isFraud"], color='red', alpha=0.5)
# plt.xlim(600,850)
# plt.ylim(0,900)
# plt.legend(loc = 1)
plt.show()


# In[98]:


from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
df_new=df.copy()


# In[99]:


y= df["isFraud"]
x = df_new.drop(['step', 'type', 'isFraud','nameDest', 'nameOrig', 'isFlaggedFraud'], axis=1)


# In[100]:


x_train, x_test,y_train,  y_test = train_test_split(x, y)
print(x_train.shape)
print(x_test.shape)


# In[101]:


df


# In[107]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# Parameter grid for hyperparameter tuning
param_grid = {
    'max_depth': [8, 10, 20, 30,35,40, None],
    'min_samples_split': [2,3,4, 5, 10],
}

# Initialize DecisionTreeClassifier
dtree_cls = DecisionTreeClassifier(random_state=42)

# Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=dtree_cls, param_grid=param_grid, 
                           cv=5, n_jobs=-1, verbose=2)

# Fit the grid search with training data and corresponding target
grid_search.fit(x_train, y_train)

# Get the best estimator (model) from grid search
best_dtree_reg = grid_search.best_estimator_

# Make predictions on the test data
y_pred = best_dtree_reg.predict(x_test)

# Get the best parameters from the grid search
params = grid_search.best_params_

# Print the best parameters and the best decision tree model
print(f"Best Parameters: {params}")
print(f"Best Decision Tree Model: {best_dtree_reg}")


# In[108]:


d_tree = DecisionTreeClassifier(
    random_state=42,
    max_depth=30,
    min_samples_split=3
)


# In[109]:


d_tree.fit(x_train, y_train)


# In[135]:


pred =d_tree.predict(x_test)

pred_train = d_tree.predict(x_train)


# In[ ]:





# In[149]:


x_train.loc[0]


# In[151]:


x_train


# In[111]:


from sklearn.metrics import accuracy_score, f1_score , confusion_matrix


# In[112]:


accuracy_score(y_test,pred )


# In[113]:


f1_score(y_test,pred)


# In[121]:


confusion_matrix(y_test , y_pred)


# In[123]:


random_forest = RandomForestClassifier()


# In[125]:


random_forest.fit(x_train,y_train)


# In[126]:


pred1= random_forest.predict(x_test)


# In[127]:


accuracy_score(pred1, y_test)


# In[128]:


f1_score(pred1, y_test )


# In[ ]:


def pred(data):
    


# In[ ]:




